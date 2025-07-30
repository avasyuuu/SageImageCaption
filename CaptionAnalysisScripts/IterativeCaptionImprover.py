import json
import os
from datetime import datetime
from typing import Dict, List, Tuple
import anthropic
from pathlib import Path
import shutil
import time

class IterativeCaptionImprover:
    def __init__(self, api_key: str):
        """Initialize the iterative caption improvement system"""
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-3-5-sonnet-20241022"
        
    def analyze_model_weaknesses(self, evaluation_results: Dict) -> Dict[str, Dict]:
        """Analyze each model's common weaknesses from evaluation results"""
        model_analysis = {}
        
        for model_name in evaluation_results['model_performance'].keys():
            weaknesses = {
                'missing_elements': {},
                'incorrect_elements': {},
                'low_object_identification': [],
                'low_scene_understanding': [],
                'average_scores': {
                    'accuracy': 0,
                    'object_identification': 0,
                    'scene_understanding': 0
                }
            }
            
            count = 0
            for img_eval in evaluation_results['image_evaluations']:
                if model_name in img_eval['model_evaluations']:
                    eval_data = img_eval['model_evaluations'][model_name]
                    
                    # Track missing elements
                    for element in eval_data.get('missing_elements', []):
                        weaknesses['missing_elements'][element] = weaknesses['missing_elements'].get(element, 0) + 1
                    
                    # Track incorrect elements
                    for element in eval_data.get('incorrect_elements', []):
                        weaknesses['incorrect_elements'][element] = weaknesses['incorrect_elements'].get(element, 0) + 1
                    
                    # Track low scores
                    if eval_data.get('object_identification_score', 100) < 60:
                        weaknesses['low_object_identification'].append(img_eval['filename'])
                    
                    if eval_data.get('scene_understanding_score', 100) < 60:
                        weaknesses['low_scene_understanding'].append(img_eval['filename'])
                    
                    # Calculate averages
                    if 'accuracy_score' in eval_data and eval_data['accuracy_score'] >= 0:
                        weaknesses['average_scores']['accuracy'] += eval_data['accuracy_score']
                        weaknesses['average_scores']['object_identification'] += eval_data.get('object_identification_score', 0)
                        weaknesses['average_scores']['scene_understanding'] += eval_data.get('scene_understanding_score', 0)
                        count += 1
            
            # Finalize averages
            if count > 0:
                for key in weaknesses['average_scores']:
                    weaknesses['average_scores'][key] /= count
            
            # Sort common issues
            weaknesses['missing_elements'] = dict(sorted(
                weaknesses['missing_elements'].items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5])  # Top 5 commonly missed elements
            
            weaknesses['incorrect_elements'] = dict(sorted(
                weaknesses['incorrect_elements'].items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5])  # Top 5 commonly incorrect elements
            
            model_analysis[model_name] = weaknesses
            
        return model_analysis

    def generate_specialized_prompts(self, model_analysis: Dict[str, Dict], base_prompt: str) -> Dict[str, str]:
        """Generate specialized prompts for each model based on their weaknesses"""
        
        print("\n🤖 Generating specialized prompts for each model...")
        
        # Prepare analysis summary for Claude
        analysis_text = f"Base prompt: '{base_prompt}'\n\nModel Performance Analysis:\n"
        
        for model_name, weaknesses in model_analysis.items():
            analysis_text += f"\n{model_name}:\n"
            analysis_text += f"- Average accuracy: {weaknesses['average_scores']['accuracy']:.1f}%\n"
            analysis_text += f"- Average object identification: {weaknesses['average_scores']['object_identification']:.1f}%\n"
            analysis_text += f"- Average scene understanding: {weaknesses['average_scores']['scene_understanding']:.1f}%\n"
            
            if weaknesses['missing_elements']:
                analysis_text += f"- Commonly misses: {', '.join(list(weaknesses['missing_elements'].keys())[:3])}\n"
            
            if weaknesses['incorrect_elements']:
                analysis_text += f"- Commonly misidentifies: {', '.join(list(weaknesses['incorrect_elements'].keys())[:3])}\n"
        
        prompt = f"""Based on this analysis of different AI models' image captioning performance, generate specialized prompts for each model to improve their specific weaknesses.

{analysis_text}

For each model, create a prompt that:
1. Maintains the intent of the base prompt
2. Addresses their specific weaknesses
3. Emphasizes what they commonly miss or get wrong
4. Is concise and actionable

Return a JSON object with model names as keys and their specialized prompts as values.
Format: {{"MODEL_NAME": "specialized prompt", ...}}"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = response.content[0].text
            
            # Extract JSON
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            json_str = response_text[start_idx:end_idx]
            
            specialized_prompts = json.loads(json_str)
            
            # Ensure all models have prompts
            for model_name in model_analysis.keys():
                if model_name not in specialized_prompts:
                    specialized_prompts[model_name] = base_prompt
            
            return specialized_prompts
            
        except Exception as e:
            print(f"Error generating specialized prompts: {e}")
            # Fallback to base prompt for all models
            return {model: base_prompt for model in model_analysis.keys()}

    def generate_improvement_instructions(self, image_evaluations: List[Dict]) -> Dict[str, List[Dict]]:
        """Generate specific improvement instructions for each model based on feedback"""
        
        model_instructions = {}
        
        for img_eval in image_evaluations:
            filename = img_eval['filename']
            
            for model_name, eval_data in img_eval['model_evaluations'].items():
                if model_name not in model_instructions:
                    model_instructions[model_name] = []
                
                # Skip if no valid evaluation
                if 'accuracy_score' not in eval_data or eval_data['accuracy_score'] < 0:
                    continue
                
                instruction = {
                    'filename': filename,
                    'original_caption': eval_data.get('original_caption', ''),
                    'accuracy_score': eval_data.get('accuracy_score', 0),
                    'suggested_improvement': eval_data.get('suggested_improvement', ''),
                    'missing_elements': eval_data.get('missing_elements', []),
                    'incorrect_elements': eval_data.get('incorrect_elements', [])
                }
                
                # Generate specific guidance
                guidance = []
                
                if eval_data.get('missing_elements'):
                    guidance.append(f"Include: {', '.join(eval_data['missing_elements'][:3])}")
                
                if eval_data.get('incorrect_elements'):
                    guidance.append(f"Correct: {', '.join(eval_data['incorrect_elements'][:3])}")
                
                if eval_data.get('object_identification_score', 100) < 60:
                    guidance.append("Focus more on identifying objects accurately")
                
                if eval_data.get('scene_understanding_score', 100) < 60:
                    guidance.append("Better describe the overall scene context")
                
                instruction['specific_guidance'] = "; ".join(guidance) if guidance else "Maintain current approach"
                
                model_instructions[model_name].append(instruction)
        
        return model_instructions

    def create_model_improvement_config(self, evaluation_json_path: str, original_caption_json_path: str, 
                                       output_dir: str) -> Dict:
        """Create improvement configuration for each model"""
        
        # Load evaluation results
        with open(evaluation_json_path, 'r', encoding='utf-8') as f:
            evaluation_results = json.load(f)
        
        # Load original caption results
        with open(original_caption_json_path, 'r', encoding='utf-8') as f:
            caption_results = json.load(f)
        
        # Get base prompt
        base_prompt = caption_results['metadata'].get('unified_prompt', 'Describe this image in detail')
        
        # Analyze model weaknesses
        print("\n📊 Analyzing model performance patterns...")
        model_analysis = self.analyze_model_weaknesses(evaluation_results)
        
        # Generate specialized prompts
        specialized_prompts = self.generate_specialized_prompts(model_analysis, base_prompt)
        
        # Generate improvement instructions
        print("\n📝 Generating improvement instructions...")
        model_instructions = self.generate_improvement_instructions(
            evaluation_results['image_evaluations']
        )
        
        # Create improvement configuration
        improvement_config = {
            "metadata": {
                "created_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "base_prompt": base_prompt,
                "evaluation_file": evaluation_json_path,
                "caption_file": original_caption_json_path,
                "total_images": len(evaluation_results['image_evaluations'])
            },
            "model_configs": {}
        }
        
        # Build config for each model
        for model_name in model_analysis.keys():
            config = {
                "specialized_prompt": specialized_prompts.get(model_name, base_prompt),
                "performance_summary": {
                    "average_accuracy": model_analysis[model_name]['average_scores']['accuracy'],
                    "average_object_identification": model_analysis[model_name]['average_scores']['object_identification'],
                    "average_scene_understanding": model_analysis[model_name]['average_scores']['scene_understanding']
                },
                "common_issues": {
                    "frequently_missed": list(model_analysis[model_name]['missing_elements'].keys()),
                    "frequently_incorrect": list(model_analysis[model_name]['incorrect_elements'].keys())
                },
                "improvement_examples": []
            }
            
            # Add top 5 worst performing examples with specific guidance
            if model_name in model_instructions:
                sorted_instructions = sorted(
                    model_instructions[model_name], 
                    key=lambda x: x['accuracy_score']
                )[:5]
                
                for inst in sorted_instructions:
                    config['improvement_examples'].append({
                        "filename": inst['filename'],
                        "original_caption": inst['original_caption'],
                        "accuracy_score": inst['accuracy_score'],
                        "suggested_improvement": inst['suggested_improvement'],
                        "specific_guidance": inst['specific_guidance']
                    })
            
            improvement_config['model_configs'][model_name] = config
        
        # Save configuration
        config_path = os.path.join(output_dir, 'model_improvement_config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(improvement_config, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Improvement configuration saved to: {config_path}")
        
        # Print summary
        self.print_improvement_summary(improvement_config)
        
        return improvement_config

    def print_improvement_summary(self, improvement_config: Dict):
        """Print a summary of the improvement configuration"""
        print("\n" + "="*70)
        print("MODEL IMPROVEMENT CONFIGURATION SUMMARY")
        print("="*70)
        
        for model_name, config in improvement_config['model_configs'].items():
            print(f"\n{model_name}:")
            print(f"  Original prompt: \"{improvement_config['metadata']['base_prompt']}\"")
            print(f"  Specialized prompt: \"{config['specialized_prompt']}\"")
            print(f"  Average accuracy: {config['performance_summary']['average_accuracy']:.1f}%")
            
            if config['common_issues']['frequently_missed']:
                print(f"  Frequently misses: {', '.join(config['common_issues']['frequently_missed'][:3])}")
            
            if config['common_issues']['frequently_incorrect']:
                print(f"  Frequently incorrect: {', '.join(config['common_issues']['frequently_incorrect'][:3])}")
            
            if config['improvement_examples']:
                print(f"  Worst performing example: {config['improvement_examples'][0]['filename']} ({config['improvement_examples'][0]['accuracy_score']}%)")

    def generate_feedback_enhanced_caption(self, model_name: str, image_path: str, 
                                         original_caption: str, feedback: Dict, 
                                         specialized_prompt: str) -> str:
        """Generate an improved caption using Claude's feedback"""
        
        # This is a placeholder for the actual implementation
        # In practice, you would integrate this with your caption generation pipeline
        
        guidance_text = ""
        if feedback.get('missing_elements'):
            guidance_text += f"Missing elements to include: {', '.join(feedback['missing_elements'])}\n"
        
        if feedback.get('incorrect_elements'):
            guidance_text += f"Incorrect elements to fix: {', '.join(feedback['incorrect_elements'])}\n"
        
        if feedback.get('suggested_improvement'):
            guidance_text += f"Suggested approach: {feedback['suggested_improvement']}\n"
        
        # This would be passed to your actual model
        enhanced_prompt = f"{specialized_prompt}\n\nPrevious attempt: {original_caption}\n\nGuidance: {guidance_text}"
        
        return enhanced_prompt


def integrate_with_main_pipeline(evaluation_json: str, caption_json: str, api_key: str, output_dir: str):
    """Integrate the improvement system with your existing pipeline"""
    
    print("\n" + "="*70)
    print("GENERATING MODEL IMPROVEMENT CONFIGURATIONS")
    print("="*70)
    
    improver = IterativeCaptionImprover(api_key)
    
    # Create improvement configuration
    improvement_config = improver.create_model_improvement_config(
        evaluation_json_path=evaluation_json,
        original_caption_json_path=caption_json,
        output_dir=output_dir
    )
    
    return improvement_config


# Example usage
if __name__ == "__main__":
    # This would be called after your initial evaluation
    api_key = "your-api-key"
    evaluation_json = "path/to/evaluation_results.json"
    caption_json = "path/to/caption_results.json"
    output_dir = "path/to/output"
    
    improvement_config = integrate_with_main_pipeline(
        evaluation_json, caption_json, api_key, output_dir
    )