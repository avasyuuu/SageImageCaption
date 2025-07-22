import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, AutoProcessor,
    BlipProcessor, BlipForConditionalGeneration, LlavaForConditionalGeneration,
    BitsAndBytesConfig
)
from ultralytics import YOLO
from PIL import Image
import warnings
import zipfile
import json
import os
from datetime import datetime
import numpy as np

warnings.filterwarnings('ignore')

class MultiModelCaptionAnalyzer:
    def __init__(self, prompt="Describe this image in detail"):
        """Initialize YOLO, BLIP, Florence-2, LLava, and Moondream models"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.cuda.empty_cache()
        
        self.prompt = prompt
        print(f"Using unified prompt: '{self.prompt}'")
        
        self.models = {}
        torch.set_grad_enabled(False)
        
        # Load all models
        self._load_yolo()
        self._load_blip()
        self._load_florence()
        self._load_llava()
        self._load_moondream()
        
        print(f"\n✓ Successfully loaded {len(self.models)} models + YOLO")
        print(f"  Active models: {list(self.models.keys())}")
    
    def _load_yolo(self):
        """Load YOLO for object detection"""
        print("\n1. Loading YOLO...")
        try:
            self.yolo = YOLO('yolov8n.pt')
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
            model_id = "llava-hf/bakLlava-v1-hf"
            
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

    def _load_moondream(self):
        """Load Moondream model"""
        print("\n5. Loading Moondream...")
        try:
            model_id = "vikhyatk/moondream2"
            
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            
            if self.device == "cuda":
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    device_map={"": self.device}
                ).to(self.device)
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    torch_dtype=torch.float32
                )
            
            model.eval()
            
            self.models['Moondream'] = {
                'model': model,
                'tokenizer': tokenizer,
                'type': 'moondream'
            }
            print("   ✓ Moondream loaded")
        except Exception as e:
            print(f"   ✗ Moondream error: {e}")

    def generate_caption(self, image, model_name, model_dict):
        """Generate caption for specific model using the unified prompt"""
        try:
            with torch.no_grad():
                if model_dict['type'] == 'blip':
                    inputs = model_dict['processor'](image, return_tensors="pt").to(self.device)
                    out = model_dict['model'].generate(
                        **inputs, 
                        max_length=50, 
                        num_beams=5,
                        min_length=10,
                        repetition_penalty=1.2
                    )
                    caption = model_dict['processor'].decode(out[0], skip_special_tokens=True)
                        
                elif model_dict['type'] == 'florence':
                    task_prompt = "<DETAILED_CAPTION>"
                    
                    inputs = model_dict['processor'](
                        text=task_prompt, 
                        images=image, 
                        return_tensors="pt"
                    ).to(self.device)
                    
                    generated_ids = model_dict['model'].generate(
                        input_ids=inputs["input_ids"],
                        pixel_values=inputs["pixel_values"],
                        max_new_tokens=50,
                        do_sample=False,
                        num_beams=2
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
                    llava_prompt = f"USER: <image>\n{self.prompt}\nASSISTANT:"
                    
                    inputs = model_dict['processor'](
                        text=llava_prompt,
                        images=image,
                        return_tensors="pt"
                    ).to(self.device)
                    
                    generate_ids = model_dict['model'].generate(
                        **inputs,
                        max_new_tokens=20,
                        min_new_tokens=10,
                        do_sample=False,
                        use_cache=True
                    )
                    
                    full_output = model_dict['processor'].batch_decode(
                        generate_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False
                    )[0]
                    
                    caption = full_output.split("ASSISTANT:")[-1].strip()
                    if caption and not caption[-1] in '.!?':
                        last_space = caption.rfind('.')
                        if last_space > 20:
                            caption = caption[:last_space] + '.'

                elif model_dict['type'] == 'moondream':
                    model = model_dict['model']
                    tokenizer = model_dict['tokenizer']
                    
                    if not isinstance(image, Image.Image):
                        image = Image.fromarray(image)
                    
                    try:
                        enc_image = model.encode_image(image)
                        caption = model.answer_question(
                            enc_image,
                            self.prompt,
                            tokenizer,
                            max_new_tokens=100
                        )
                    except AttributeError:
                        prompt_text = f"<image>\n\nQuestion: {self.prompt}\n\nAnswer:"
                        
                        processor = AutoProcessor.from_pretrained("vikhyatk/moondream2", trust_remote_code=True)
                        inputs = processor(images=image, text=prompt_text, return_tensors="pt")
                        inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
                        
                        output_ids = model.generate(
                            **inputs,
                            max_new_tokens=100,
                            do_sample=False,
                            temperature=0
                        )
                        
                        caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                        
                        if "Answer:" in caption:
                            caption = caption.split("Answer:")[-1].strip()
                        else:
                            caption = caption.replace(prompt_text, "").strip()
                        
                        first_period = caption.rfind('.')
                        if first_period > 0:
                            caption = caption[:first_period] + "."
                else:
                    caption = "Unknown model type"
                
        except Exception as e:
            print(f"   Error with {model_name}: {str(e)}")
            caption = f"Error generating caption"
            
        return caption
    
    def process_yolo_detection(self, image, save_path=None):
        """Run YOLO object detection and optionally save annotated image"""
        if self.yolo is None:
            return [], None, False
        
        # Resize for faster processing
        max_size = 640
        if image.width > max_size or image.height > max_size:
            ratio = max_size / max(image.width, image.height)
            new_width = int(image.width * ratio)
            new_height = int(image.height * ratio)
            image_resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        else:
            image_resized = image
        
        results = self.yolo(image_resized, verbose=False)
        objects = []
        
        if results[0].boxes:
            for box in results[0].boxes:
                if float(box.conf) > 0.5:
                    obj_name = self.yolo.names[int(box.cls)]
                    confidence = float(box.conf)
                    objects.append((obj_name, confidence))
        
        saved = False
        if save_path and results[0].boxes:
            try:
                annotated_img = results[0].plot()
                
                if isinstance(annotated_img, np.ndarray):
                    if len(annotated_img.shape) == 3 and annotated_img.shape[2] == 3:
                        annotated_img = annotated_img[:, :, ::-1]
                    annotated_pil = Image.fromarray(annotated_img)
                else:
                    annotated_pil = annotated_img
                
                annotated_pil.save(save_path)
                saved = True
            except Exception as e:
                print(f"   Error saving YOLO image: {e}")
        
        return objects, results, saved

    def analyze_zip_file(self, zip_path, output_json="results.json", show_yolo=False, yolo_output_dir=None):
        """Process all images in a zip file and save results to JSON"""
        results = {
            "metadata": {
                "zip_file": zip_path,
                "processed_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "models_used": list(self.models.keys()) + ["YOLO"],
                "device": self.device,
                "unified_prompt": self.prompt,
                "yolo_output_directory": yolo_output_dir if yolo_output_dir else None
            },
            "images": []
        }
        
        if yolo_output_dir:
            os.makedirs(yolo_output_dir, exist_ok=True)
            print(f"\n📁 YOLO detection images will be saved to: {yolo_output_dir}")
        
        temp_dir = "temp_extracted_images"
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
                image_files = [f for f in zip_ref.namelist()
                            if f.lower().endswith(image_extensions) and not f.startswith('__MACOSX')]
                
                print(f"\nFound {len(image_files)} images in zip file")
                print("="*70)
                
                for idx, image_file in enumerate(image_files, 1):
                    print(f"\n[{idx}/{len(image_files)}] Processing: {image_file}")
                    print("-"*50)
                    
                    try:
                        zip_ref.extract(image_file, temp_dir)
                        image_path = os.path.join(temp_dir, image_file)
                        
                        image = Image.open(image_path).convert('RGB')
                        
                        yolo_save_path = None
                        if yolo_output_dir:
                            base_name = os.path.basename(image_file)
                            name_without_ext = os.path.splitext(base_name)[0]
                            yolo_save_path = os.path.join(yolo_output_dir, f"{name_without_ext}_yolo.jpg")
                        
                        objects, yolo_results, yolo_saved = self.process_yolo_detection(image, yolo_save_path)
                        
                        if yolo_saved:
                            print(f"  ✓ YOLO detection saved: {os.path.basename(yolo_save_path)}")
                        
                        captions = {}
                        for model_name, model_dict in self.models.items():
                            caption = self.generate_caption(image, model_name, model_dict)
                            captions[model_name] = caption
                            print(f"  {model_name}: {caption[:50]}...")
                        
                        yolo_data = {
                            "object_count": len(objects),
                            "objects": [{"name": obj, "confidence": conf} for obj, conf in objects],
                            "detection_image_saved": yolo_saved
                        }
                        
                        results["images"].append({
                            "filename": image_file,
                            "captions": captions,
                            "yolo_detection": yolo_data
                        })
                        
                        os.remove(image_path)
                        
                    except Exception as e:
                        print(f"  Error processing {image_file}: {str(e)}")
                        results["images"].append({
                            "filename": image_file,
                            "error": str(e)
                        })
            
            with open(output_json, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f"\n✅ Results saved to: {output_json}")
            print(f"   Processed {len(results['images'])} images")
            print(f"   Using prompt: '{self.prompt}'")
            if yolo_output_dir:
                print(f"   YOLO detections saved to: {yolo_output_dir}")
            
        finally:
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        
        return results