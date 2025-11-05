import os
import cv2
import numpy as np
import glob
from craft_text_detector import Craft

# ----- Paths -----
IMG_DIR = "data/icdar2015/test_images"
SAVE_DIR = "outputs/craft_results"
os.makedirs(SAVE_DIR, exist_ok=True)

def load_image(image_path):
    """Load image using cv2 to bypass the problematic image_utils"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    return img

def save_detection_results(image, boxes, scores, image_path, save_dir):
    """Save detection results with bounding boxes drawn"""
    # Draw boxes on image
    result_img = image.copy()
    for box in boxes:
        if box is not None:
            box = np.array(box, dtype=np.int32)
            cv2.polylines(result_img, [box], True, (0, 255, 0), 2)
    
    # Save result
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    result_path = os.path.join(save_dir, f"{base_name}_craft_result.jpg")
    cv2.imwrite(result_path, result_img)
    
    # Save boxes coordinates with scores
    txt_path = os.path.join(save_dir, f"{base_name}_craft_boxes.txt")
    with open(txt_path, 'w') as f:
        for i, box in enumerate(boxes):
            if box is not None:
                # Convert box coordinates to string
                coords = ','.join([f"{int(x)},{int(y)}" for x, y in box])
                # Get score (fallback to 0.9 if not available)
                score = float(scores[i]) if scores is not None and i < len(scores) else 0.90
                f.write(f"{coords},{score:.4f}\n")

def process_dataset(img_dir, save_dir, craft_model):
    """Process all images in the dataset"""
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(img_dir, ext)))
    
    if not image_files:
        print(f"No images found in {img_dir}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    processed_count = 0
    failed_count = 0
    
    for i, image_path in enumerate(image_files, 1):
        try:
            print(f"Processing image {i}/{len(image_files)}: {os.path.basename(image_path)}")
            
            # Load image
            image = load_image(image_path)
            
            # Use the internal prediction function directly
            from craft_text_detector.predict import get_prediction
            
            # Get prediction
            prediction_result = get_prediction(
                image=image,
                craft_net=craft_model.craft_net,
                refine_net=craft_model.refine_net,
                text_threshold=craft_model.text_threshold,
                link_threshold=craft_model.link_threshold,
                low_text=craft_model.low_text,
                cuda=craft_model.cuda,
                long_size=craft_model.long_size,
                poly=False  # Use boxes instead of polygons
            )
            
            # Extract boxes and scores
            boxes = prediction_result["boxes"]
            scores = prediction_result.get("boxes_scores")  # Get scores if available
            
            # Save results
            save_detection_results(image, boxes, scores, image_path, save_dir)
            
            processed_count += 1
            print(f"  ✅ Detected {len(boxes)} text regions")
            
        except Exception as e:
            failed_count += 1
            print(f"  ❌ Failed to process {os.path.basename(image_path)}: {str(e)}")
    
    print(f"\n🎉 Dataset processing completed!")
    print(f"✅ Successfully processed: {processed_count} images")
    print(f"❌ Failed: {failed_count} images")
    print(f"📁 Results saved to: {save_dir}")

try:
    # Initialize CRAFT model
    print("Initializing CRAFT model...")
    craft = Craft(
        output_dir=SAVE_DIR,
        crop_type="box",  # Use box instead of poly to avoid polygon issues
        cuda=False,  # Set to True if you have CUDA available
        rectify=True,
        weight_path_craft_net="models/craft_mlt_25k.pth"
    )
    print("CRAFT model initialized successfully!")
    
    # Process the entire dataset
    print(f"Starting dataset processing from: {IMG_DIR}")
    process_dataset(IMG_DIR, SAVE_DIR, craft)
    
    # Release model
    craft.unload_craftnet_model()
    craft.unload_refinenet_model()
    
except Exception as e:
    print(f"Error occurred: {str(e)}")
    import traceback
    traceback.print_exc()