from CaptionAnalysis import MultiModelCaptionAnalyzer
from CaptionEvaluator import ClaudeCaptionEvaluator
import torch
import sys
import os
import json
from datetime import datetime

def process_zip_file(zip_path, output_json="caption_results.json", prompt="Describe this image in detail", yolo_output_dir=None):
    """Process 1 zip file of images and generate captions"""
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    print("\n🚀 Initializing Multi-Model Caption Analysis...")
    analyzer = MultiModelCaptionAnalyzer(prompt=prompt)
    
    print(f"\n📁 Processing zip file: {zip_path}")
    results = analyzer.analyze_zip_file(
        zip_path, 
        output_json=output_json, 
        show_yolo=False,
        yolo_output_dir=yolo_output_dir
    )
    
    print("\n✅ Caption generation complete!")
    return results

def evaluate_captions(caption_json_path, zip_path, api_key, output_json="evaluation_results.json"):
    """Evaluate captions using Claude Sonnet"""
    print("\n" + "="*70)
    print("EVALUATING CAPTIONS WITH CLAUDE SONNET")
    print("="*70)
    
    print("\n🔍 Initializing Claude evaluator...")
    evaluator = ClaudeCaptionEvaluator(api_key)
    
    print(f"\n📊 Evaluating captions from: {caption_json_path}")
    evaluation_results = evaluator.process_json_results(
        json_path=caption_json_path,
        zip_path=zip_path,
        output_path=output_json
    )
    
    return evaluation_results

def create_combined_report(caption_results, evaluation_results, output_path="combined_report.json"):
    """Create a combined report with both caption and evaluation data"""
    print("\n" + "="*70)
    print("CREATING COMBINED REPORT")
    print("="*70)
    
    combined = {
        "metadata": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "caption_generation": caption_results['metadata'],
            "evaluation": evaluation_results['metadata']
        },
        "model_rankings": evaluation_results['model_performance'],
        "detailed_results": []
    }
    
    for img_data in caption_results['images']:
        filename = img_data['filename']
        
        eval_data = None
        for eval_img in evaluation_results['image_evaluations']:
            if eval_img['filename'] == filename:
                eval_data = eval_img
                break
        
        if eval_data:
            combined_entry = {
                "filename": filename,
                "yolo_detection": img_data.get('yolo_detection', {}),
                "captions_and_scores": {}
            }
            
            for model_name, caption in img_data.get('captions', {}).items():
                model_eval = eval_data['model_evaluations'].get(model_name, {})
                combined_entry['captions_and_scores'][model_name] = {
                    "caption": caption,
                    "accuracy_score": model_eval.get('accuracy_score', -1),
                    "object_identification_score": model_eval.get('object_identification_score', -1),
                    "scene_understanding_score": model_eval.get('scene_understanding_score', -1),
                    "missing_elements": model_eval.get('missing_elements', []),
                    "incorrect_elements": model_eval.get('incorrect_elements', []),
                    "suggested_improvement": model_eval.get('suggested_improvement', '')
                }
            
            combined['detailed_results'].append(combined_entry)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    
    print(f"\n📋 Combined report saved to: {output_path}")
    
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"Total images processed: {len(combined['detailed_results'])}")
    print("\nModel Rankings by Average Accuracy:")
    
    rankings = [(model, data['average_score']) 
                for model, data in combined['model_rankings'].items()]
    rankings.sort(key=lambda x: x[1], reverse=True)
    
    for rank, (model, score) in enumerate(rankings, 1):
        perf = combined['model_rankings'][model]
        print(f"\n{rank}. {model}: {score:.1f}/100")
        print(f"   - Success rate: {perf['successful_evaluations']}/{perf['total_evaluations']}")
        print(f"   - Score distribution:")
        for range_name, count in perf['score_distribution'].items():
            if count > 0:
                print(f"     {range_name}: {count}")
    
    return combined

def main():
    """Main execution function with three modes"""
    DEFAULT_ZIP_PATH = r"C:\Users\avasy\imagedataset\test_images.zip"
    DEFAULT_CAPTION_JSON = "caption_analysis_results.json"
    DEFAULT_EVALUATION_OUTPUT = "evaluation_results.json"
    DEFAULT_COMBINED_OUTPUT = "combined_report.json"
    DEFAULT_PROMPT = "Describe this image in detail"
    DEFAULT_YOLO_DIR = "yolo_detections"
    
    PROMPT_EXAMPLES = [
        "Describe this image in detail",
        "What objects can you see in this image?",
        "Describe the scene and activities in this image",
        "What is happening in this picture?",
        "Provide a comprehensive description of this image including objects, people, and setting",
        "List all visible objects and describe the overall scene"
    ]
    
    print("="*70)
    print("MULTI-MODEL CAPTION ANALYSIS WITH CLAUDE EVALUATION")
    print("="*70)
    print("\nSelect execution mode:")
    print("1. Generate captions only")
    print("2. Evaluate existing captions only")
    print("3. Generate captions AND evaluate (full pipeline)")
    
    while True:
        mode = input("\nEnter mode (1/2/3): ").strip()
        if mode in ['1', '2', '3']:
            break
        print("Invalid selection. Please enter 1, 2, or 3.")
    
    if mode == '1':
        print("\n" + "="*50)
        print("MODE: CAPTION GENERATION ONLY")
        print("="*50)
        
        zip_path = input(f"\nZip file path [{DEFAULT_ZIP_PATH}]: ").strip() or DEFAULT_ZIP_PATH
        
        print("\n📝 Prompt Selection:")
        print("Choose a prompt or enter your own:")
        for i, prompt_ex in enumerate(PROMPT_EXAMPLES, 1):
            print(f"{i}. {prompt_ex}")
        print(f"{len(PROMPT_EXAMPLES) + 1}. Enter custom prompt")
        
        prompt_choice = input(f"\nChoice (1-{len(PROMPT_EXAMPLES) + 1}) [1]: ").strip() or "1"
        
        if prompt_choice.isdigit() and 1 <= int(prompt_choice) <= len(PROMPT_EXAMPLES):
            prompt = PROMPT_EXAMPLES[int(prompt_choice) - 1]
        else:
            prompt = input("Enter your custom prompt: ").strip() or DEFAULT_PROMPT
        
        print(f"\n✅ Using prompt: '{prompt}'")
        
        output_json = input(f"Output JSON file [{DEFAULT_CAPTION_JSON}]: ").strip() or DEFAULT_CAPTION_JSON
        
        save_yolo = input("\nSave YOLO detection images? (y/n) [y]: ").strip().lower()
        save_yolo = save_yolo != 'n'
        
        yolo_output_dir = None
        if save_yolo:
            yolo_output_dir = input(f"YOLO output directory [{DEFAULT_YOLO_DIR}]: ").strip() or DEFAULT_YOLO_DIR
        
        try:
            caption_results = process_zip_file(zip_path, output_json, prompt, yolo_output_dir)
            print(f"\n✅ Caption generation complete!")
            print(f"   Results saved to: {output_json}")
            print(f"   Prompt used: '{prompt}'")
            if yolo_output_dir:
                print(f"   YOLO detections saved to: {yolo_output_dir}")
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            sys.exit(1)
    
    elif mode == '2':
        print("\n" + "="*50)
        print("MODE: CAPTION EVALUATION ONLY")
        print("="*50)
        
        caption_json = input(f"\nCaption JSON file [{DEFAULT_CAPTION_JSON}]: ").strip() or DEFAULT_CAPTION_JSON
        
        if not os.path.exists(caption_json):
            print(f"\n❌ Error: Caption file '{caption_json}' not found!")
            sys.exit(1)
        
        try:
            with open(caption_json, 'r', encoding='utf-8') as f:
                caption_data = json.load(f)
                used_prompt = caption_data.get('metadata', {}).get('unified_prompt', 'Unknown')
                print(f"\n📝 Original prompt used: '{used_prompt}'")
        except Exception as e:
            print(f"\n⚠️  Could not read prompt from file: {e}")
        
        print("\nDo you have the original zip file? (needed to extract images)")
        has_zip = input("(y/n): ").strip().lower() == 'y'
        
        zip_path = None
        if has_zip:
            zip_path = input(f"Zip file path [{DEFAULT_ZIP_PATH}]: ").strip() or DEFAULT_ZIP_PATH
        else:
            print("\n⚠️  Warning: Without zip file, images must be at their original paths")
        
        api_key = os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            print("\n⚠️  No API key found in environment variables.")
            api_key = input("Enter Anthropic API key: ").strip()
            if not api_key:
                print("\n❌ Error: API key required for evaluation")
                sys.exit(1)
        
        eval_output = input(f"Evaluation output file [{DEFAULT_EVALUATION_OUTPUT}]: ").strip() or DEFAULT_EVALUATION_OUTPUT
        
        try:
            with open(caption_json, 'r', encoding='utf-8') as f:
                caption_results = json.load(f)
            
            evaluation_results = evaluate_captions(caption_json, zip_path, api_key, eval_output)
            
            create_combined = input("\nCreate combined report? (y/n): ").strip().lower() == 'y'
            if create_combined:
                combined_output = input(f"Combined report file [{DEFAULT_COMBINED_OUTPUT}]: ").strip() or DEFAULT_COMBINED_OUTPUT
                create_combined_report(caption_results, evaluation_results, combined_output)
            
            print(f"\n✅ Evaluation complete!")
            print(f"   Results saved to: {eval_output}")
            
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            sys.exit(1)
    
    else:  # mode == '3'
        print("\n" + "="*50)
        print("MODE: FULL PIPELINE (CAPTION + EVALUATION)")
        print("="*50)
        
        zip_path = input(f"\nZip file path [{DEFAULT_ZIP_PATH}]: ").strip() or DEFAULT_ZIP_PATH
        
        print("\n📝 Prompt Selection:")
        print("Choose a prompt or enter your own:")
        for i, prompt_ex in enumerate(PROMPT_EXAMPLES, 1):
            print(f"{i}. {prompt_ex}")
        print(f"{len(PROMPT_EXAMPLES) + 1}. Enter custom prompt")
        
        prompt_choice = input(f"\nChoice (1-{len(PROMPT_EXAMPLES) + 1}) [1]: ").strip() or "1"
        
        if prompt_choice.isdigit() and 1 <= int(prompt_choice) <= len(PROMPT_EXAMPLES):
            prompt = PROMPT_EXAMPLES[int(prompt_choice) - 1]
        else:
            prompt = input("Enter your custom prompt: ").strip() or DEFAULT_PROMPT
        
        print(f"\n✅ Using prompt: '{prompt}'")
        
        caption_output = input(f"Caption output file [{DEFAULT_CAPTION_JSON}]: ").strip() or DEFAULT_CAPTION_JSON
        eval_output = input(f"Evaluation output file [{DEFAULT_EVALUATION_OUTPUT}]: ").strip() or DEFAULT_EVALUATION_OUTPUT
        combined_output = input(f"Combined report file [{DEFAULT_COMBINED_OUTPUT}]: ").strip() or DEFAULT_COMBINED_OUTPUT
        
        save_yolo = input("\nSave YOLO detection images? (y/n) [y]: ").strip().lower()
        save_yolo = save_yolo != 'n'
        
        yolo_output_dir = None
        if save_yolo:
            yolo_output_dir = input(f"YOLO output directory [{DEFAULT_YOLO_DIR}]: ").strip() or DEFAULT_YOLO_DIR
        
        api_key = os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            print("\n⚠️  No API key found in environment variables.")
            api_key = input("Enter Anthropic API key: ").strip()
            if not api_key:
                print("\n❌ Error: API key required for evaluation")
                sys.exit(1)
        
        try:
            print("\n" + "="*50)
            print("STEP 1: GENERATING CAPTIONS")
            print("="*50)
            caption_results = process_zip_file(zip_path, caption_output, prompt, yolo_output_dir)
            
            print("\n" + "="*50)
            print("STEP 2: EVALUATING CAPTIONS")
            print("="*50)
            evaluation_results = evaluate_captions(caption_output, zip_path, api_key, eval_output)
            
            print("\n" + "="*50)
            print("STEP 3: CREATING COMBINED REPORT")
            print("="*50)
            combined_report = create_combined_report(
                caption_results,
                evaluation_results,
                combined_output
            )
            
            print("\n✨ All processing complete!")
            print(f"   • Captions: {caption_output}")
            print(f"   • Evaluations: {eval_output}")
            print(f"   • Combined report: {combined_output}")
            if yolo_output_dir:
                print(f"   • YOLO detections: {yolo_output_dir}")
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Process interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            sys.exit(1)

if __name__ == "__main__":
    main()