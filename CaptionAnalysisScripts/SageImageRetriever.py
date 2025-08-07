from gradio_client import Client
import requests
import json
import os
from datetime import datetime
from pathlib import Path
import time
from typing import List, Dict, Optional
from PIL import Image
from io import BytesIO
import logging

class SageImageRetriever:
    def __init__(self, base_url: str = "http://localhost:7860/", 
                 sage_user: str = None, sage_token: str = None):
        """
        Initialize Sage image retriever
        NOTE: Requires port forwarding: ssh h100-node2 -L 7860:localhost:7860
        """
        self.base_url = base_url
        self.sage_user = sage_user or os.environ.get('SAGE_USER', 'FILL_IN')
        self.sage_token = sage_token or os.environ.get('SAGE_TOKEN', 'FILL_IN')
        self.client = None
        
        try:
            self.client = Client(base_url)
            print(f"✅ Connected to Sage image search at {base_url}")
        except Exception as e:
            print(f"⚠️  Using fallback method: {e}")
    
    def search_with_gradio_client(self, query: str) -> Dict:
        """Search using gradio client"""
        if not self.client:
            return {"data": []}
        
        try:
            result = self.client.predict(query=query, api_name="/search")
            return json.loads(result) if isinstance(result, str) else result
        except Exception as e:
            print(f"Gradio client error: {e}")
            return {"data": []}
    
    def get_image(self, url: str) -> Optional[Image.Image]:
        """Retrieve image from Sage storage with authentication"""
        auth = (self.sage_user, self.sage_token)
        
        try:
            response = requests.get(url, auth=auth, timeout=30)
            response.raise_for_status()
            image_data = response.content
            
            image = Image.open(BytesIO(image_data))
            image = image.convert("RGB")
            
            return image
            
        except requests.exceptions.HTTPError as e:
            logging.debug(f"Image skipped, HTTPError for URL {url}: {e}")
            if e.response.status_code == 401:
                print(f"   ❌ Authentication failed - check your SAGE_USER and SAGE_TOKEN")
            return None
        except Exception as e:
            logging.debug(f"Image skipped, error for URL {url}: {e}")
            return None
    
    def search_images(self, query: str) -> Dict:
        """Search for images using the specified query"""
        print(f"\n🔍 Searching for: '{query}'")
        
        results = self.search_with_gradio_client(query)
        
        if isinstance(results, dict) and "error" in results:
            print(f"❌ Search error: {results['error']}")
            return {"data": []}
        
        if isinstance(results, dict) and 'data' in results:
            images = results['data']
        elif isinstance(results, list):
            images = results
            results = {"data": images}
        else:
            print(f"⚠️  Unexpected result format: {type(results)}")
            results = {"data": []}
        
        print(f"✅ Found {len(results.get('data', []))} images")
        return results
    
    def download_sage_images(self, query: str, output_dir: str, max_images: int = 50) -> Dict:
        """Search and download Sage images to local directory"""
        os.makedirs(output_dir, exist_ok=True)
        metadata_file = os.path.join(output_dir, "sage_metadata.json")
        
        search_results = self.search_images(query)
        images = search_results.get('data', [])
        
        if not images:
            return {"error": "No images found", "query": query}
        
        images = images[:max_images]
        
        download_results = {
            "query": query,
            "search_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_found": len(images),
            "downloaded": 0,
            "failed": 0,
            "images": []
        }
        
        print(f"\n📥 Downloading {len(images)} images to {output_dir}")
        
        for idx, img_data in enumerate(images):
            try:
                if isinstance(img_data, list) and len(img_data) > 9:
                    filename = img_data[0] if len(img_data) > 0 else f"image_{idx}.jpg"
                    caption = img_data[1] if len(img_data) > 1 else ""
                    score = img_data[2] if len(img_data) > 2 else 0.0
                    image_url = img_data[9]
                    
                    node_id = "unknown"
                    if len(img_data) > 3:
                        import re
                        for item in img_data[3:]:
                            if isinstance(item, str):
                                node_match = re.search(r'000048b02d[a-f0-9]+', item)
                                if node_match:
                                    node_id = node_match.group()
                                    break
                else:
                    print(f"  ⚠️  Unexpected data format for image {idx}")
                    download_results["failed"] += 1
                    continue
                
                if '/' in filename:
                    filename = filename.split('/')[-1]
                
                base_name = Path(filename).stem
                extension = Path(filename).suffix or '.jpg'
                filename = f"{base_name}_{node_id}_{idx}{extension}"
                filepath = os.path.join(output_dir, filename)
                
                print(f"  [{idx+1}/{len(images)}] Downloading {filename}...", end='', flush=True)
                
                image = self.get_image(image_url)
                
                if image:
                    image.save(filepath)
                    print(" ✓")
                    download_results["downloaded"] += 1
                    
                    caption_text = ""
                    keywords = ""
                    if isinstance(caption, str):
                        if 'caption:' in caption and 'keywords:' in caption:
                            parts = caption.split('keywords:')
                            caption_text = parts[0].replace('caption:', '').strip()
                            keywords = parts[1].strip() if len(parts) > 1 else ""
                        else:
                            caption_text = caption
                    
                    img_metadata = {
                        "filename": filename,
                        "original_url": image_url,
                        "sage_id": base_name,
                        "node_id": node_id,
                        "score": score,
                        "sage_caption": caption_text,
                        "sage_keywords": keywords,
                        "query": query,
                        "download_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "original_metadata": img_data
                    }
                    download_results["images"].append(img_metadata)
                else:
                    print(" ✗")
                    download_results["failed"] += 1
                
                time.sleep(0.1)
                
            except Exception as e:
                print(f" ✗ Error: {str(e)[:100]}...")
                download_results["failed"] += 1
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(download_results, f, indent=2)
        
        print(f"\n📊 Download Summary:")
        print(f"   • Successfully downloaded: {download_results['downloaded']}")
        print(f"   • Failed: {download_results['failed']}")
        print(f"   • Metadata saved to: {metadata_file}")
        
        return download_results
    
    def download_multiple_queries(self, queries: List[str], base_output_dir: str, 
                                images_per_query: int = 20) -> Dict:
        """Download images for multiple search queries"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        main_output_dir = os.path.join(base_output_dir, f"sage_images_{timestamp}")
        os.makedirs(main_output_dir, exist_ok=True)
        
        all_results = {
            "timestamp": timestamp,
            "output_directory": main_output_dir,
            "sage_credentials": {
                "user": self.sage_user,
                "authenticated": self.sage_user != 'FILL_IN'
            },
            "queries": {},
            "total_downloaded": 0,
            "total_failed": 0
        }
        
        for query in queries:
            print(f"\n{'='*60}")
            print(f"Processing query: '{query}'")
            print('='*60)
            
            query_dir = os.path.join(main_output_dir, query.replace(' ', '_'))
            
            results = self.download_sage_images(query, query_dir, max_images=images_per_query)
            
            all_results["queries"][query] = results
            all_results["total_downloaded"] += results.get("downloaded", 0)
            all_results["total_failed"] += results.get("failed", 0)
            
            time.sleep(1)
        
        combined_metadata_file = os.path.join(main_output_dir, "all_queries_metadata.json")
        with open(combined_metadata_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"✅ All downloads complete!")
        print(f"   • Total images downloaded: {all_results['total_downloaded']}")
        print(f"   • Total failures: {all_results['total_failed']}")
        print(f"   • Output directory: {main_output_dir}")
        print(f"   • Combined metadata: {combined_metadata_file}")
        
        return all_results
    