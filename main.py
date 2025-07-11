import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, AutoProcessor,
    BlipProcessor, BlipForConditionalGeneration
)
from ultralytics import YOLO
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

class MultiModelCaptionAnalyzer:
    def __init__(self):
        """Initialize YOLO, BLIP, and Florence-2 models"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        self.models = {}
        
        # Load all models
        self._load_yolo()
        self._load_blip()
        self._load_florence()
        
        print(f"\n✓ Successfully loaded {len(self.models)} models + YOLO")
        print(f"  Active models: {list(self.models.keys())}")
    
    def _load_yolo(self):
        """Load YOLO for object detection"""
        print("\n1. Loading YOLO...")
        try:
            self.yolo = YOLO('yolov8s.pt')
            print("   ✓ YOLO loaded")
        except Exception as e:
            print(f"   ✗ YOLO error: {e}")
            self.yolo = None
    
    def _load_blip(self):
        """Load BLIP model"""
        print("\n2. Loading BLIP...")
        try:
            self.models['BLIP'] = {
                'processor': BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large"),
                'model': BlipForConditionalGeneration.from_pretrained(
                    "Salesforce/blip-image-captioning-large"
                ).to(self.device),
                'type': 'blip'
            }
            print("   ✓ BLIP loaded")
        except Exception as e:
            print(f"   ✗ BLIP error: {e}")
    
    def _load_florence(self):
        """Load Florence-2 model"""
        print("\n3. Loading Florence-2...")
        try:
            model_id = "microsoft/Florence-2-base"
            
            self.models['Florence-2'] = {
                'model': AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float32,
                    trust_remote_code=True
                ).to(self.device).eval(),
                'processor': AutoProcessor.from_pretrained(model_id, trust_remote_code=True),
                'type': 'florence'
            }
            print("   ✓ Florence-2 loaded")
        except Exception as e:
            print(f"   ✗ Florence-2 error: {e}")
    
    def generate_caption(self, image, model_name, model_dict):
        """Generate caption for specific model"""
        try:
            if model_dict['type'] == 'blip':
                inputs = model_dict['processor'](image, return_tensors="pt").to(self.device)
                out = model_dict['model'].generate(**inputs, max_length=50, num_beams=3)
                caption = model_dict['processor'].decode(out[0], skip_special_tokens=True)
                
            elif model_dict['type'] == 'florence':
                task_prompt = "<CAPTION>"
                
                inputs = model_dict['processor'](
                    text=task_prompt, 
                    images=image, 
                    return_tensors="pt"
                ).to(self.device)
                
                generated_ids = model_dict['model'].generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=100,
                    do_sample=False
                )
                
                generated_text = model_dict['processor'].batch_decode(
                    generated_ids, 
                    skip_special_tokens=False
                )[0]
                
                parsed = model_dict['processor'].post_process_generation(
                    generated_text, 
                    task=task_prompt, 
                    image_size=(image.width, image.height)
                )
                caption = parsed[task_prompt]
                
            else:
                caption = "Unknown model type"
                
        except Exception as e:
            print(f"   Error with {model_name}: {str(e)}")
            caption = f"Error generating caption"
            
        return caption
    
    def detect_objects_yolo(self, image):
        """Run YOLO object detection"""
        if self.yolo is None:
            return [], None
        
        results = self.yolo(image, verbose=False)
        objects = []
        
        if results[0].boxes:
            for box in results[0].boxes:
                if float(box.conf) > 0.5:
                    obj_name = self.yolo.names[int(box.cls)]
                    confidence = float(box.conf)
                    objects.append((obj_name, confidence))
        
        return objects, results
    
    def analyze_image(self, image_path, show_yolo=True):
        """Main function to analyze image and display results"""
        
        # Load image
        print(f"\n{'='*70}")
        print(f"Processing: {image_path}")
        print(f"{'='*70}")
        
        image = Image.open(image_path).convert('RGB')
        
        # Run YOLO detection
        print("\n📦 YOLO Object Detection:")
        print("-" * 60)
        objects, yolo_results = self.detect_objects_yolo(image)
        
        if objects:
            print(f"Detected {len(objects)} objects:\n")
            for obj_name, confidence in objects:
                print(f"  • {obj_name}: {confidence:.1%} confidence")
        else:
            print("  No objects detected")
        
        # Generate captions from each model
        print("\n📝 Model Captions:")
        print("-" * 60)
        
        captions = {}
        for model_name, model_dict in self.models.items():
            print(f"\n{model_name}:")
            caption = self.generate_caption(image, model_name, model_dict)
            captions[model_name] = caption
            print(f"  \"{caption}\"")
        
        print("\n" + "="*70)
        
        # Show YOLO detection image if requested
        if show_yolo and yolo_results:
            print("\n🖼️  Displaying YOLO detection image...")
            yolo_results[0].show()  # This opens the default image viewer
        
        return captions, objects


# Simple usage function
def analyze_single_image(image_path):
    """Quick function to analyze a single image"""
    analyzer = MultiModelCaptionAnalyzer()
    captions, objects = analyzer.analyze_image(image_path, show_yolo=True)
    return captions, objects


# Batch processing function
def analyze_multiple_images(image_paths):
    """Analyze multiple images"""
    analyzer = MultiModelCaptionAnalyzer()
    
    all_results = []
    for i, image_path in enumerate(image_paths, 1):
        print(f"\n\n🔍 Image {i}/{len(image_paths)}")
        captions, objects = analyzer.analyze_image(image_path, show_yolo=False)
        all_results.append({
            'path': image_path,
            'captions': captions,
            'objects': objects
        })
    
    return all_results


# Main execution
if __name__ == "__main__":
    # Your image path
    image_path = r'C:\Users\avasy\imagedataset\intersectiondata2.jpg'
    
    # Run the analysis
    print("\n🚀 Starting Multi-Model Caption Analysis...")
    captions, objects = analyze_single_image(image_path)
    
    print("\n✅ Processing complete!")
    
    # Example for batch processing:
    # image_paths = [
    #     r'C:\Users\avasy\imagedataset\13U_R3\APN_S2_13U_R3_IMAG0033.JPG',
    #     r'C:\Users\avasy\imagedataset\13U_R3\APN_S2_13U_R3_IMAG0034.JPG'
    # ]
    # results = analyze_multiple_images(image_paths)