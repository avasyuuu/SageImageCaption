from CaptionAnalysis import MultiModelCaptionAnalyzer
from CaptionEvaluator import ClaudeCaptionEvaluator
import torch
import sys
import os
import json
from datetime import datetime
import shutil
import glob 

def create_output_structure(base_name="RESULTS"):
    """Create organized output folder structure with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{base_name}_{timestamp}"
    
    # Create main output directory and subdirectories
    paths = {
        'root': output_dir,
        'json': os.path.join(output_dir, 'json_outputs'),
        'yolo': os.path.join(output_dir, 'yolo_detections'),
        'logs': os.path.join(output_dir, 'logs')
    }
    
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    
    return paths

def process_directory(dir_path, output_paths, prompt="Describe this image in detail", save_yolo=True):
    """Process directory with organized output structure"""
    caption_json = os.path.join(output_paths['json'], 'caption_results.json')
    yolo_dir = output_paths['yolo'] if save_yolo else None
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    print("\n🚀 Initializing Multi-Model Caption Analysis...")
    analyzer = MultiModelCaptionAnalyzer(prompt=prompt)
    
    print(f"\n📁 Processing directory: {dir_path}")
    print(f"📂 Output directory: {output_paths['root']}")
    
    results = analyzer.analyze_directory(
        dir_path, 
        output_json=caption_json, 
        show_yolo=False,
        yolo_output_dir=yolo_dir
    )
    
    # Save processing log
    log_path = os.path.join(output_paths['logs'], 'caption_generation.log')
    with open(log_path, 'w') as f:
        f.write(f"Caption Generation Log\n")
        f.write(f"{'='*50}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Input directory: {dir_path}\n")
        f.write(f"Prompt: {prompt}\n")
        f.write(f"Models used: {', '.join(results['metadata']['models_used'])}\n")
        f.write(f"Images processed: {len(results['images'])}\n")
        f.write(f"Output JSON: {caption_json}\n")
        if yolo_dir:
            f.write(f"YOLO detections: {yolo_dir}\n")
    
    print("\n✅ Caption generation complete!")
    return results, caption_json

def evaluate_captions(caption_json_path, image_dir, api_key, output_paths):
    """Evaluate captions with organized output"""
    eval_json = os.path.join(output_paths['json'], 'evaluation_results.json')
    
    print("\n" + "="*70)
    print("EVALUATING CAPTIONS WITH CLAUDE SONNET")
    print("="*70)
    
    print("\n🔍 Initializing Claude evaluator...")
    evaluator = ClaudeCaptionEvaluator(api_key)
    
    print(f"\n📊 Evaluating captions from: {caption_json_path}")
    evaluation_results = evaluator.process_json_results(
        json_path=caption_json_path,
        image_dir=image_dir,
        output_path=eval_json
    )
    
    # Save evaluation log
    log_path = os.path.join(output_paths['logs'], 'evaluation.log')
    with open(log_path, 'w') as f:
        f.write(f"Caption Evaluation Log\n")
        f.write(f"{'='*50}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Input caption file: {caption_json_path}\n")
        f.write(f"API calls made: {evaluation_results['metadata']['api_calls_made']}\n")
        f.write(f"Images evaluated: {len(evaluation_results['image_evaluations'])}\n")
        f.write(f"Output JSON: {eval_json}\n")
    
    return evaluation_results, eval_json

def create_combined_report(caption_results, evaluation_results, output_paths):
    """Create combined report in organized structure"""
    report_path = os.path.join(output_paths['json'], 'combined_analysis_report.json')
    
    print("\n" + "="*70)
    print("CREATING COMBINED REPORT")
    print("="*70)
    
    combined = {
        "metadata": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "output_directory": output_paths['root'],
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
                "full_path": img_data.get('full_path', ''),
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
    
    # Save combined report
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    
    print(f"\n📋 Combined report saved to: json_outputs/combined_analysis_report.json")
    
    return combined

def archive_previous_outputs(base_dir="RESULTS_*"):
    """Archive previous output directories"""
    archive_dir = "archived_outputs"
    
    # Find all previous output directories
    prev_outputs = glob.glob(base_dir)
    
    if prev_outputs:
        os.makedirs(archive_dir, exist_ok=True)
        print(f"\n📦 Archiving {len(prev_outputs)} previous output(s)...")
        
        for output in prev_outputs:
            if os.path.isdir(output):
                archive_name = os.path.join(archive_dir, os.path.basename(output))
                if os.path.exists(archive_name):
                    shutil.rmtree(archive_name)
                shutil.move(output, archive_dir)
        
        print(f"   Moved to: {archive_dir}/")

def main():
    """Main execution function with organized output structure"""
    DEFAULT_DIR_PATH = r"C:\Users\avasy\SageImageCaptioning\SageImageCaption\test_images"
    DEFAULT_PROMPT = "Describe this image in detail"
    
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
    
    # Ask about archiving
    if len(glob.glob("RESULTS_*")) > 0:
        archive = input("\nArchive previous outputs? (y/n) [y]: ").strip().lower()
        if archive != 'n':
            archive_previous_outputs()
    
    print("\nSelect execution mode:")
    print("1. Generate captions only")
    print("2. Evaluate existing captions only")
    print("3. Generate captions AND evaluate (full pipeline)")
    
    while True:
        mode = input("\nEnter mode (1/2/3): ").strip()
        if mode in ['1', '2', '3']:
            break
        print("Invalid selection. Please enter 1, 2, or 3.")
    
    # Create output structure
    output_paths = create_output_structure()
    print(f"\n📁 Created output directory: {output_paths['root']}/")
    
    if mode == '1':
        print("\n" + "="*50)
        print("MODE: CAPTION GENERATION ONLY")
        print("="*50)
        
        dir_path = input(f"\nImage directory path [{DEFAULT_DIR_PATH}]: ").strip() or DEFAULT_DIR_PATH
        
        if not os.path.exists(dir_path):
            print(f"\n❌ Error: Directory '{dir_path}' not found!")
            sys.exit(1)
        
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
        
        save_yolo = input("\nSave YOLO detection images? (y/n) [y]: ").strip().lower()
        save_yolo = save_yolo != 'n'
        
        try:
            caption_results, caption_json = process_directory(
                dir_path, output_paths, prompt, save_yolo
            )
            
            print(f"\n✅ Caption generation complete!")
            print(f"\n📂 All outputs saved to: {output_paths['root']}/")
            print(f"   • Captions: json_outputs/caption_results.json")
            if save_yolo:
                print(f"   • YOLO detections: yolo_detections/")
            print(f"   • Logs: logs/caption_generation.log")
            
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    elif mode == '2':
        print("\n" + "="*50)
        print("MODE: CAPTION EVALUATION ONLY")
        print("="*50)
        
        caption_json = input(f"\nCaption JSON file path: ").strip()
        
        if not os.path.exists(caption_json):
            print(f"\n❌ Error: Caption file '{caption_json}' not found!")
            sys.exit(1)
        
        # Copy caption file to new output directory
        shutil.copy2(caption_json, os.path.join(output_paths['json'], 'original_captions.json'))
        
        try:
            with open(caption_json, 'r', encoding='utf-8') as f:
                caption_data = json.load(f)
                used_prompt = caption_data.get('metadata', {}).get('unified_prompt', 'Unknown')
                print(f"\n📝 Original prompt used: '{used_prompt}'")
                
                # Get image directory from metadata
                image_dir = caption_data.get('metadata', {}).get('directory', None)
                if image_dir:
                    print(f"📁 Original image directory: {image_dir}")
        except Exception as e:
            print(f"\n⚠️  Could not read metadata from file: {e}")
            image_dir = None
        
        # Ask for directory if not found in metadata or doesn't exist
        if not image_dir or not os.path.exists(image_dir):
            print("\nImage directory not found in metadata or doesn't exist.")
            image_dir = input(f"Enter image directory path [{DEFAULT_DIR_PATH}]: ").strip() or DEFAULT_DIR_PATH
            
            if not os.path.exists(image_dir):
                print(f"\n❌ Error: Directory '{image_dir}' not found!")
                sys.exit(1)
        
        api_key = os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            print("\n⚠️  No API key found in environment variables.")
            api_key = input("Enter Anthropic API key: ").strip()
            if not api_key:
                print("\n❌ Error: API key required for evaluation")
                sys.exit(1)
        
        try:
            with open(caption_json, 'r', encoding='utf-8') as f:
                caption_results = json.load(f)
            
            evaluation_results, eval_json = evaluate_captions(
                caption_json, image_dir, api_key, output_paths
            )
            
            combined_report = create_combined_report(
                caption_results, evaluation_results, output_paths
            )
            
            print(f"\n✅ Evaluation complete!")
            print(f"\n📂 All outputs saved to: {output_paths['root']}/")
            print(f"   • Original captions: json_outputs/original_captions.json")
            print(f"   • Evaluation: json_outputs/evaluation_results.json")
            print(f"   • Combined report: json_outputs/combined_analysis_report.json")
            print(f"   • Logs: logs/evaluation.log")
            
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    else:  # mode == '3'
        print("\n" + "="*50)
        print("MODE: FULL PIPELINE (CAPTION + EVALUATION)")
        print("="*50)
        
        dir_path = input(f"\nImage directory path [{DEFAULT_DIR_PATH}]: ").strip() or DEFAULT_DIR_PATH
        
        if not os.path.exists(dir_path):
            print(f"\n❌ Error: Directory '{dir_path}' not found!")
            sys.exit(1)
        
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
        
        save_yolo = input("\nSave YOLO detection images? (y/n) [y]: ").strip().lower()
        save_yolo = save_yolo != 'n'
        
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
            caption_results, caption_json = process_directory(
                dir_path, output_paths, prompt, save_yolo
            )
            
            print("\n" + "="*50)
            print("STEP 2: EVALUATING CAPTIONS")
            print("="*50)
            evaluation_results, eval_json = evaluate_captions(
                caption_json, dir_path, api_key, output_paths
            )
            
            print("\n" + "="*50)
            print("STEP 3: CREATING COMBINED REPORT")
            print("="*50)
            combined_report = create_combined_report(
                caption_results, evaluation_results, output_paths
            )
            
            print("\n✨ All processing complete!")
            print(f"\n📂 All outputs saved to: {output_paths['root']}/")
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Process interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

if __name__ == "__main__":
    main()