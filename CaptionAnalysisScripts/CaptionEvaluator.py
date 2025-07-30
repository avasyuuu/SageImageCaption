import os
import json
import base64
import anthropic
from pathlib import Path
from typing import Dict
from datetime import datetime
import time
import shutil

class ClaudeCaptionEvaluator:
    def __init__(self, api_key: str):
        """Initialize Claude client with API key"""
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-3-5-sonnet-20241022"
        
    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 string"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def evaluate_all_captions_for_image(self, image_path: str, captions: Dict[str, str]) -> Dict[str, Dict]:
        """Evaluate ALL model captions for a single image in ONE API call"""
        base64_image = self.encode_image(image_path)
        
        file_extension = Path(image_path).suffix.lower()
        media_type_map = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        media_type = media_type_map.get(file_extension, 'image/jpeg')
        
        captions_text = ""
        for model_name, caption in captions.items():
            captions_text += f"\n{model_name}: \"{caption}\""
        
        prompt = f"""Evaluate these AI-generated image captions. For each caption, assess:
1. Accuracy (0-100): How accurately it describes the image
2. Object identification: Are objects correctly identified?
3. Scene understanding: Does it capture the overall context?
4. Missing elements: Important things not mentioned
5. Incorrect elements: Things wrongly described

Captions to evaluate:{captions_text}

Return a JSON object with this exact structure:
{{
    "MODEL_NAME": {{
        "accuracy_score": 0-100,
        "object_identification_score": 0-100,
        "scene_understanding_score": 0-100,
        "missing_elements": ["element1", "element2"],
        "incorrect_elements": ["element1", "element2"],
        "suggested_improvement": "Brief improved caption"
    }},
    // ... repeat for each model
}}

Be concise but thorough. Focus on major issues rather than minor details."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                temperature=0,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": base64_image
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            )
            
            response_text = response.content[0].text
            
            try:
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                json_str = response_text[start_idx:end_idx]
                evaluations_raw = json.loads(json_str)
                
                evaluations = {}
                for model_name, caption in captions.items():
                    if model_name in evaluations_raw:
                        eval_data = evaluations_raw[model_name]
                        eval_data['model_name'] = model_name
                        eval_data['original_caption'] = caption
                        evaluations[model_name] = eval_data
                    else:
                        evaluations[model_name] = {
                            "model_name": model_name,
                            "original_caption": caption,
                            "error": "Not evaluated in batch response",
                            "accuracy_score": -1
                        }
                
                return evaluations
                
            except Exception as e:
                print(f"JSON parsing error: {e}")
                evaluations = {}
                for model_name, caption in captions.items():
                    evaluations[model_name] = {
                        "model_name": model_name,
                        "original_caption": caption,
                        "raw_response": response_text,
                        "error": f"Failed to parse JSON response: {str(e)}",
                        "accuracy_score": -1
                    }
                return evaluations
            
        except Exception as e:
            print(f"API error: {str(e)}")
            evaluations = {}
            for model_name, caption in captions.items():
                evaluations[model_name] = {
                    "model_name": model_name,
                    "original_caption": caption,
                    "error": f"API error: {str(e)}",
                    "accuracy_score": -1
                }
            return evaluations
    
    def process_json_results(self, json_path: str, image_dir: str = None, output_path: str = None):
        """Process the JSON results from MultiModelCaptionAnalyzer"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if output_path is None:
            output_path = f"caption_evaluation_{timestamp}.json"
        
        # Extract directory path from metadata if not provided
        if image_dir is None and 'directory' in data['metadata']:
            image_dir = data['metadata']['directory']
        
        results = {
            "metadata": {
                "evaluation_timestamp": timestamp,
                "original_results_file": json_path,
                "image_directory": image_dir,
                "evaluator_model": self.model,
                "models_evaluated": data['metadata']['models_used'][:-1],  # Exclude YOLO
                "total_images": len(data['images']),
                "api_calls_made": 0,
                "evaluation_method": "batch_per_image"
            },
            "model_performance": {},
            "image_evaluations": []
        }
        
        for model in results['metadata']['models_evaluated']:
            results['model_performance'][model] = {
                "total_evaluations": 0,
                "successful_evaluations": 0,
                "total_score": 0,
                "average_score": 0,
                "score_distribution": {"0-20": 0, "21-40": 0, "41-60": 0, "61-80": 0, "81-100": 0}
            }
        
        for idx, image_data in enumerate(data['images'], 1):
            filename = image_data['filename']
            print(f"\n[{idx}/{len(data['images'])}] Processing: {filename}")
            
            if 'error' in image_data:
                print(f"  Skipping - original processing error: {image_data['error']}")
                continue
            
            # Use full_path if available, otherwise construct from directory
            if 'full_path' in image_data:
                image_path = image_data['full_path']
            elif image_dir:
                image_path = os.path.join(image_dir, filename)
            else:
                image_path = filename
            
            if not os.path.exists(image_path):
                print(f"  Error: Image file not found at {image_path}")
                continue
            
            captions = image_data.get('captions', {})
            
            if not captions:
                print(f"  No captions found for {filename}")
                continue
            
            print(f"  Evaluating {len(captions)} captions in single batch...", end='', flush=True)
            evaluations = self.evaluate_all_captions_for_image(image_path, captions)
            results['metadata']['api_calls_made'] += 1
            
            successful_count = sum(1 for eval_data in evaluations.values() 
                                 if 'accuracy_score' in eval_data and eval_data['accuracy_score'] >= 0)
            print(f" Done! ({successful_count}/{len(captions)} successful)")
            
            image_result = {
                "filename": filename,
                "full_path": image_path,
                "yolo_detection": image_data.get('yolo_detection', {}),
                "model_evaluations": evaluations
            }
            results['image_evaluations'].append(image_result)
            
            for model_name, eval_data in evaluations.items():
                if model_name in results['model_performance']:
                    perf = results['model_performance'][model_name]
                    perf['total_evaluations'] += 1
                    
                    if 'accuracy_score' in eval_data and eval_data['accuracy_score'] >= 0:
                        perf['successful_evaluations'] += 1
                        score = eval_data['accuracy_score']
                        perf['total_score'] += score
                        
                        if score <= 20:
                            perf['score_distribution']['0-20'] += 1
                        elif score <= 40:
                            perf['score_distribution']['21-40'] += 1
                        elif score <= 60:
                            perf['score_distribution']['41-60'] += 1
                        elif score <= 80:
                            perf['score_distribution']['61-80'] += 1
                        else:
                            perf['score_distribution']['81-100'] += 1
            
            time.sleep(0.5)
        
        for model_name, perf in results['model_performance'].items():
            if perf['successful_evaluations'] > 0:
                perf['average_score'] = round(perf['total_score'] / perf['successful_evaluations'], 1)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        self.print_summary(results)
        
        print(f"\n✅ Evaluation complete! Results saved to: {output_path}")
        
        return results
    
    def print_summary(self, results: Dict):
        """Print a summary of the evaluation results"""
        print("\n" + "="*70)
        print("EVALUATION SUMMARY")
        print("="*70)
        
        print(f"\nAPI Usage:")
        print(f"  - Total API calls: {results['metadata']['api_calls_made']}")
        print(f"  - Images processed: {len(results['image_evaluations'])}")
        print(f"  - Evaluation method: {results['metadata']['evaluation_method']}")
        
        print("\nModel Performance Rankings:")
        print("-"*50)
        
        model_scores = [(model, data['average_score']) 
                       for model, data in results['model_performance'].items()]
        model_scores.sort(key=lambda x: x[1], reverse=True)
        
        for rank, (model, avg_score) in enumerate(model_scores, 1):
            perf = results['model_performance'][model]
            print(f"{rank}. {model}: {avg_score:.1f}/100")
            print(f"   - Successful evaluations: {perf['successful_evaluations']}/{perf['total_evaluations']}")
            print(f"   - Score distribution:")
            for range_name, count in perf['score_distribution'].items():
                if count > 0:
                    percentage = (count / perf['successful_evaluations'] * 100) if perf['successful_evaluations'] > 0 else 0
                    print(f"     {range_name}: {count} ({percentage:.1f}%)")
        
        print("\n" + "-"*50)
        print("Image Analysis Highlights:")
        print("-"*50)
        
        best_scores = []
        worst_scores = []
        
        for img_eval in results['image_evaluations']:
            filename = img_eval['filename']
            for model, eval_data in img_eval['model_evaluations'].items():
                if 'accuracy_score' in eval_data and eval_data['accuracy_score'] >= 0:
                    best_scores.append((filename, model, eval_data['accuracy_score']))
                    worst_scores.append((filename, model, eval_data['accuracy_score']))
        
        best_scores.sort(key=lambda x: x[2], reverse=True)
        worst_scores.sort(key=lambda x: x[2])
        
        print("\nTop 3 Best Captions:")
        for filename, model, score in best_scores[:3]:
            print(f"  • {Path(filename).name} - {model}: {score}/100")
        
        print("\nTop 3 Worst Captions:")
        for filename, model, score in worst_scores[:3]:
            print(f"  • {Path(filename).name} - {model}: {score}/100")