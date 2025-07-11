import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, AutoProcessor,
    BlipProcessor, BlipForConditionalGeneration, LlavaForConditionalGeneration, LlavaProcessor,
    PaliGemmaForConditionalGeneration, PaliGemmaProcessor
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
        self._load_paligemma()
        self._load_llava()
        
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
    
    def _load_llava(self):
        """Load LLaVA model"""
        print("\n4. Loading LLaVA...")
        try:
            model_id = "llava-hf/bakLlava-v1-hf"  # Using smaller variant
            
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            ) if self.device == "cuda" else None
            
            self.models['LLaVA'] = {
                'processor': AutoProcessor.from_pretrained(model_id),
                'model': LlavaForConditionalGeneration.from_pretrained(
                    model_id,
                    quantization_config=quantization_config,
                    device_map="auto" if self.device == "cuda" else None
                ),
                'type': 'llava'
            }
            print("   ✓ LLaVA loaded")
        except Exception as e:
            print(f"   ✗ LLaVA error: {e}")

    def _load_paligemma(self):
        """Load PaliGemma - trying different versions"""
        print("\n5. Loading PaliGemma...")
        
        # List of PaliGemma models to try (from most open to most restricted)
        paligemma_models = [
            "google/paligemma-3b-pt-224",     # Base pre-trained
            "google/paligemma-3b-mix-224",    # Mixed training
            "google/paligemma-3b-mix-448",    # Higher resolution
            "google/paligemma-3b-ft-vqav2-224", # Fine-tuned for VQA
        ]
        
        loaded = False
        for model_id in paligemma_models:
            try:
                print(f"   Trying {model_id}...")
                
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16
                ) if self.device == "cuda" else None
                
                self.models['PaliGemma'] = {
                    'processor': AutoProcessor.from_pretrained(model_id),
                    'model': PaliGemmaForConditionalGeneration.from_pretrained(
                        model_id,
                        quantization_config=quantization_config,
                        device_map="auto" if self.device == "cuda" else None,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                    ),
                    'type': 'paligemma'
                }
                print(f"   ✓ PaliGemma loaded successfully using {model_id}")
                loaded = True
                break
                
            except Exception as e:
                if "restricted" in str(e) or "401" in str(e):
                    continue
                else:
                    print(f"   Error with {model_id}: {str(e)[:50]}...")
        
        if not loaded:
            print("   ✗ All PaliGemma models require authentication")
            print("   To use PaliGemma:")
            print("   1. Run: pip install huggingface_hub")
            print("   2. Run: huggingface-cli login")
            print("   3. Visit https://huggingface.co/google/paligemma-3b-pt-224")
            print("   4. Accept the license agreement")
    
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

            elif model_dict['type'] == 'llava':
                # LLaVA uses a conversation format
                prompt = "USER: <image>\nDescribe this image.\nASSISTANT:"
                
                inputs = model_dict['processor'](
                    text=prompt,
                    images=image,
                    return_tensors="pt"
                ).to(self.device)
                
                generate_ids = model_dict['model'].generate(
                    **inputs,
                    max_new_tokens=22,   # TOKEN SIZE FOR LLAVA
                    do_sample=False
                )
                
                full_output = model_dict['processor'].batch_decode(
                    generate_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0]
                
                caption = full_output.split("ASSISTANT:")[-1].strip()

            elif model_dict['type'] == 'paligemma':
                # PaliGemma uses task prompts
                prompt = "<image>describe"          # For description
               # prompt = "<image>caption en"        # For English caption
               # prompt = "<image>answer en what is in this image?"  # For Q&A
               # prompt = "<image>detect"           # For object detection
                
                inputs = model_dict['processor'](
                    text=prompt,
                    images=image,
                    return_tensors="pt",
                    padding="longest"
                ).to(self.device)
                
                with torch.no_grad():
                    generate_ids = model_dict['model'].generate(
                        **inputs,
                        max_new_tokens=100,
                        do_sample=False
                    )
                
                caption = model_dict['processor'].batch_decode(
                    generate_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )[0]
                
                caption = caption.replace(prompt, "").strip()
                caption = caption.replace("describe", "").strip()
                        
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