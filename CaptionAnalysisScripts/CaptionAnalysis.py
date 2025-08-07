import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, AutoProcessor,
    BlipProcessor, BlipForConditionalGeneration, LlavaForConditionalGeneration,
    BitsAndBytesConfig
)
from ultralytics import YOLO
from PIL import Image
import warnings
import json
import os
from datetime import datetime
import numpy as np
import gc
from tqdm import tqdm
from pathlib import Path

warnings.filterwarnings('ignore')

class MultiModelCaptionAnalyzer:
    def __init__(self, prompt="Describe this image in one complete sentence with detail"):
        """Initialize analyzer with prompt only - models loaded on demand"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.cuda.empty_cache()
        
        self.prompt = prompt
        print(f"Using unified prompt: '{self.prompt}'")
        
        torch.set_grad_enabled(False)
        
        self.model_configs = {
            'BLIP': {
                'loader': self._load_blip,
                'generator': self._generate_blip,
                'type': 'blip'
            },
            'Florence-2': {
                'loader': self._load_florence,
                'generator': self._generate_florence,
                'type': 'florence'
            },
            'LLaVA': {
                'loader': self._load_llava,
                'generator': self._generate_llava,
                'type': 'llava'
            },
            'Moondream': {
                'loader': self._load_moondream,
                'generator': self._generate_moondream,
                'type': 'moondream'
            }
        }
        
        self._load_yolo()
        print(f"\n✓ Analyzer initialized with {len(self.model_configs)} models available")
    
    def _clear_gpu_memory(self):
        """Clear GPU memory between models"""
        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
    
    def _load_yolo(self):
        """Load YOLO for object detection"""
        print("\nLoading YOLO for object detection...")
        try:
            self.yolo = YOLO('yolov8n.pt')
            print("   ✓ YOLO loaded")
        except Exception as e:
            print(f"   ✗ YOLO error: {e}")
            self.yolo = None
    
    def _load_blip(self):
        """Load BLIP model"""
        try:
            processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
            model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-large"
            ).to(self.device)
            return {'processor': processor, 'model': model}
        except Exception as e:
            print(f"   ✗ BLIP loading error: {e}")
            return None
    
    def _load_florence(self):
        """Load Florence-2 model"""
        try:
            model_id = "microsoft/Florence-2-base"
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                trust_remote_code=True
            ).to(self.device).eval()
            processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            return {'model': model, 'processor': processor}
        except Exception as e:
            print(f"   ✗ Florence-2 loading error: {e}")
            return None
    
    def _load_llava(self):
        """Load LLaVA model"""
        try:
            model_id = "llava-hf/bakLlava-v1-hf"
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            ) if self.device == "cuda" else None
            
            processor = AutoProcessor.from_pretrained(model_id)
            model = LlavaForConditionalGeneration.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                device_map="auto" if self.device == "cuda" else None
            )
            return {'processor': processor, 'model': model}
        except Exception as e:
            print(f"   ✗ LLaVA loading error: {e}")
            return None

    def _load_moondream(self):
        """Load Moondream model"""
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
            return {'model': model, 'tokenizer': tokenizer}
        except Exception as e:
            print(f"   ✗ Moondream loading error: {e}")
            return None

    def _generate_blip(self, image, model_dict):
        """Generate BLIP caption - ensure complete sentence"""
        inputs = model_dict['processor'](image, return_tensors="pt").to(self.device)
        out = model_dict['model'].generate(
            **inputs, 
            max_length=100,  # Increased for complete sentences
            min_length=15,
            num_beams=5,
            repetition_penalty=1.2,
            early_stopping=False  # Don't stop early
        )
        caption = model_dict['processor'].decode(out[0], skip_special_tokens=True)
        
        # Ensure caption ends with proper punctuation
        if caption and not caption[-1] in '.!?':
            caption += '.'
        
        return caption
    
    def _generate_florence(self, image, model_dict):
        """Generate Florence-2 caption - ensure complete sentence"""
        task_prompt = "<DETAILED_CAPTION>"
        
        inputs = model_dict['processor'](
            text=task_prompt, 
            images=image, 
            return_tensors="pt"
        ).to(self.device)
        
        generated_ids = model_dict['model'].generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=100,  # Increased for complete sentences
            min_new_tokens=15,
            do_sample=False,
            num_beams=3,
            early_stopping=False
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
        
        # Ensure caption ends with proper punctuation
        if caption and not caption[-1] in '.!?':
            caption += '.'
        
        return caption
    
    def _generate_llava(self, image, model_dict):
        """Generate LLaVA caption - ensure complete sentence"""
        llava_prompt = f"USER: <image>\n{self.prompt}\nASSISTANT:"
        
        inputs = model_dict['processor'](
            text=llava_prompt,
            images=image,
            return_tensors="pt"
        ).to(self.device)
        
        generate_ids = model_dict['model'].generate(
            **inputs,
            max_new_tokens=100,  # Increased significantly
            min_new_tokens=20,
            do_sample=False,
            use_cache=True,
            pad_token_id=model_dict['processor'].tokenizer.pad_token_id,
            eos_token_id=model_dict['processor'].tokenizer.eos_token_id
        )
        
        full_output = model_dict['processor'].batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        caption = full_output.split("ASSISTANT:")[-1].strip()
        
        # Clean up incomplete sentences
        if caption:
            # Find the last complete sentence
            last_period = caption.rfind('.')
            last_exclaim = caption.rfind('!')
            last_question = caption.rfind('?')
            
            last_punct = max(last_period, last_exclaim, last_question)
            
            if last_punct > 0:
                caption = caption[:last_punct + 1]
            elif not caption[-1] in '.!?':
                caption += '.'
        
        return caption
    
    def _generate_moondream(self, image, model_dict):
        """Generate Moondream caption - ensure complete sentence"""
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
                max_new_tokens=150  # Increased for complete sentences
            )
        except AttributeError:
            prompt_text = f"<image>\n\nQuestion: {self.prompt}\n\nAnswer:"
            
            processor = AutoProcessor.from_pretrained("vikhyatk/moondream2", trust_remote_code=True)
            inputs = processor(images=image, text=prompt_text, return_tensors="pt")
            inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            
            output_ids = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
                temperature=0
            )
            
            caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            if "Answer:" in caption:
                caption = caption.split("Answer:")[-1].strip()
            else:
                caption = caption.replace(prompt_text, "").strip()
        
        # Ensure complete sentence
        if caption:
            last_period = caption.rfind('.')
            last_exclaim = caption.rfind('!')
            last_question = caption.rfind('?')
            
            last_punct = max(last_period, last_exclaim, last_question)
            
            if last_punct > 0:
                caption = caption[:last_punct + 1]
            elif not caption[-1] in '.!?':
                caption += '.'
        
        return caption
    
    def process_yolo_detection(self, image, save_path=None):
        """Run YOLO object detection and optionally save annotated image"""
        if self.yolo is None:
            return [], None, False
        
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

    def analyze_directory(self, dir_path, output_json="results.json", show_yolo=False, yolo_output_dir=None):
        """Process all images in a directory - model by model for efficiency"""
        results = {
            "metadata": {
                "directory": dir_path,
                "processed_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "models_used": list(self.model_configs.keys()) + ["YOLO"],
                "device": self.device,
                "unified_prompt": self.prompt,
                "yolo_output_directory": yolo_output_dir if yolo_output_dir else None,
                "processing_method": "model-first (optimized)"
            },
            "images": []
        }
        
        if yolo_output_dir:
            os.makedirs(yolo_output_dir, exist_ok=True)
            print(f"\n📁 YOLO detection images will be saved to: {yolo_output_dir}")
        
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.gif')
        image_files = []
        
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file.lower().endswith(image_extensions):
                    full_path = os.path.join(root, file)
                    relative_path = os.path.relpath(full_path, dir_path)
                    image_files.append((full_path, relative_path))
        
        print(f"\nFound {len(image_files)} images in directory")
        print("="*70)
        
        for full_path, relative_path in image_files:
            results["images"].append({
                "filename": relative_path,
                "full_path": full_path,
                "captions": {},
                "yolo_detection": {}
            })
        
        print("\n📸 Running YOLO object detection on all images...")
        for idx, (image_path, relative_path) in enumerate(tqdm(image_files, desc="YOLO Detection")):
            try:
                image = Image.open(image_path).convert('RGB')
                
                yolo_save_path = None
                if yolo_output_dir:
                    relative_dir = os.path.dirname(relative_path)
                    if relative_dir:
                        yolo_subdir = os.path.join(yolo_output_dir, relative_dir)
                        os.makedirs(yolo_subdir, exist_ok=True)
                    
                    base_name = os.path.basename(image_path)
                    name_without_ext = os.path.splitext(base_name)[0]
                    yolo_filename = f"{name_without_ext}_yolo.jpg"
                    
                    if relative_dir:
                        yolo_save_path = os.path.join(yolo_output_dir, relative_dir, yolo_filename)
                    else:
                        yolo_save_path = os.path.join(yolo_output_dir, yolo_filename)
                
                objects, yolo_results, yolo_saved = self.process_yolo_detection(image, yolo_save_path)
                
                results["images"][idx]["yolo_detection"] = {
                    "object_count": len(objects),
                    "objects": [{"name": obj, "confidence": conf} for obj, conf in objects],
                    "detection_image_saved": yolo_saved
                }
                
            except Exception as e:
                print(f"\n  Error processing YOLO for {relative_path}: {str(e)}")
                results["images"][idx]["yolo_detection"] = {"error": str(e)}
        
        for model_name, model_config in self.model_configs.items():
            print(f"\n🤖 Processing all images with {model_name}...")
            
            print(f"   Loading {model_name}...")
            model_dict = model_config['loader']()
            
            if model_dict is None:
                print(f"   ✗ Failed to load {model_name}, skipping...")
                for idx in range(len(image_files)):
                    results["images"][idx]["captions"][model_name] = "Model failed to load"
                continue
            
            print(f"   ✓ {model_name} loaded successfully")
            
            for idx, (image_path, relative_path) in enumerate(tqdm(image_files, desc=f"{model_name} Captions")):
                try:
                    image = Image.open(image_path).convert('RGB')
                    
                    with torch.no_grad():
                        caption = model_config['generator'](image, model_dict)
                    
                    results["images"][idx]["captions"][model_name] = caption
                    
                except Exception as e:
                    print(f"\n  Error processing {relative_path} with {model_name}: {str(e)}")
                    results["images"][idx]["captions"][model_name] = f"Error: {str(e)}"
            
            print(f"   Unloading {model_name} and clearing memory...")
            del model_dict
            self._clear_gpu_memory()
        
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Results saved to: {output_json}")
        print(f"   Processed {len(results['images'])} images")
        print(f"   Using prompt: '{self.prompt}'")
        print(f"   Processing method: Model-first (optimized)")
        if yolo_output_dir:
            print(f"   YOLO detections saved to: {yolo_output_dir}")
        
        print("\n📊 Processing Summary:")
        print("-"*50)
        for model_name in self.model_configs.keys():
            successful = sum(1 for img in results["images"] 
                           if model_name in img["captions"] 
                           and not img["captions"][model_name].startswith("Error:"))
            print(f"{model_name}: {successful}/{len(image_files)} successful")
        
        return results