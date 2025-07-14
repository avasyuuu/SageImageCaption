from CaptionAnalysis import MultiModelCaptionAnalyzer
import torch
import sys

def process_zip_file(zip_path, output_json="image_analysis_results.json"):
    """Process a zip file of images"""
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Initialize analyzer
    print("\n🚀 Initializing Multi-Model Caption Analysis...")
    analyzer = MultiModelCaptionAnalyzer()
    
    # Process zip file
    print(f"\n📁 Processing zip file: {zip_path}")
    results = analyzer.analyze_zip_file(zip_path, output_json=output_json, show_yolo=False)
    
    print("\n✅ Processing complete!")
    return results

# Main execution
if __name__ == "__main__":
    # Option 1: Process a zip file
    zip_path = r"C:\Users\avasy\imagedataset\test_images.zip" # Change to your zip path
    output_json = "caption_analysis_results.json"
    
    results = process_zip_file(zip_path, output_json)
    
    # Option 2: Process single image (your existing code)
    # image_path = r"C:\Users\avasy\imagedataset\peoplePlayingBasketball.jpg"
    # analyze_single_image(image_path)