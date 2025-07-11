from CaptionAnalysis import MultiModelCaptionAnalyzer
    
 # Your image path
image_path = r"C:\Users\avasy\imagedataset\peoplePlayingBasketball.jpg"

# Simple usage function
def analyze_single_image(image_path):
    """Quick function to analyze a single image"""
    analyzer = MultiModelCaptionAnalyzer()
    captions, objects = analyzer.analyze_image(image_path, show_yolo=True)
    return captions, objects

# Run the analysis
print("\n🚀 Starting Multi-Model Caption Analysis...")
captions, objects = analyze_single_image(image_path)


print("\n✅ Processing complete!")
