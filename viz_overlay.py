#!/usr/bin/env python3
"""
Step 8: Visualization - Create comparison panels showing EAST, CRAFT, and FUSED results
Creates side-by-side visualizations with color-coded bounding boxes and legends.
"""

import os
import cv2
import numpy as np
from glob import glob
from shapely.geometry import Polygon

def read_polys(txt_path):
    """Read polygon coordinates from text file"""
    polys = []
    if not os.path.exists(txt_path):
        print(f"Warning: {txt_path} not found")
        return polys
    
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Split by comma or space
            vals = line.split(',') if ',' in line else line.split()
            if len(vals) < 8:
                continue
            
            try:
                # Extract first 8 coordinates (x1,y1,x2,y2,x3,y3,x4,y4)
                coords = list(map(float, vals[:8]))
                # Reshape to 4 points
                pts = np.array(coords).reshape(4, 2)
                polys.append(pts)
            except (ValueError, IndexError):
                continue
    
    return polys

def draw_polygons(img, polys, color, label=None, thickness=2):
    """Draw polygons on image with specified color and optional label"""
    result = img.copy()
    
    # Draw polygons
    for poly in polys:
        pts = poly.astype(int).reshape(-1, 1, 2)
        cv2.polylines(result, [pts], True, color, thickness)
    
    # Add label with background
    if label:
        label_bg_color = color
        text_color = (255, 255, 255)
        
        # Calculate text size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        text_thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
        
        # Draw background rectangle
        cv2.rectangle(result, (10, 10), (10 + text_w + 20, 10 + text_h + 20), label_bg_color, -1)
        
        # Draw text
        cv2.putText(result, label, (20, 10 + text_h + 5), font, font_scale, text_color, text_thickness, cv2.LINE_AA)
    
    return result

def create_comparison_panel(img_path, east_txt, craft_txt, fused_txt, save_path):
    """Create side-by-side comparison panel for one image"""
    
    # Load original image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not load image {img_path}")
        return False
    
    h, w = img.shape[:2]
    
    # Read polygon coordinates
    east_polys = read_polys(east_txt)
    craft_polys = read_polys(craft_txt)
    fused_polys = read_polys(fused_txt)
    
    # Create visualizations with different colors
    # EAST = Red, CRAFT = Blue, FUSED = Green
    img_east = draw_polygons(img, east_polys, (0, 0, 255), f"EAST ({len(east_polys)})")
    img_craft = draw_polygons(img, craft_polys, (255, 0, 0), f"CRAFT ({len(craft_polys)})")
    img_fused = draw_polygons(img, fused_polys, (0, 200, 0), f"FUSED ({len(fused_polys)})")
    
    # Create side-by-side panel
    panel = np.concatenate([img_east, img_craft, img_fused], axis=1)
    
    # Add overall title
    panel_h, panel_w = panel.shape[:2]
    title_img = np.zeros((50, panel_w, 3), dtype=np.uint8)
    
    # Get image name for title
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    title = f"Text Detection Comparison: {img_name}"
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    (title_w, title_h), _ = cv2.getTextSize(title, font, font_scale, thickness)
    
    # Center the title
    x = (panel_w - title_w) // 2
    y = (50 + title_h) // 2
    cv2.putText(title_img, title, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    
    # Combine title and panel
    final_panel = np.concatenate([title_img, panel], axis=0)
    
    # Save result
    cv2.imwrite(save_path, final_panel)
    return True

def create_single_visualization(name="img_103"):
    """Create visualization for a single image (for testing)"""
    
    print(f"🎨 Creating single visualization for {name}")
    
    # Define paths
    base_dir = "data/icdar2015/test_images"
    east_dir = "outputs/east_eval_ready"
    craft_dir = "outputs/craft_eval_ready"
    fused_dir = "outputs/ensemble_eval_ready"
    out_dir = "outputs/visualizations"
    
    # Create output directory
    os.makedirs(out_dir, exist_ok=True)
    
    # Create panel
    success = create_comparison_panel(
        img_path=os.path.join(base_dir, f"{name}.jpg"),
        east_txt=os.path.join(east_dir, f"{name}.txt"),
        craft_txt=os.path.join(craft_dir, f"{name}.txt"),
        fused_txt=os.path.join(fused_dir, f"{name}.txt"),
        save_path=os.path.join(out_dir, f"{name}_comparison.jpg")
    )
    
    if success:
        print(f"✅ Saved: {out_dir}/{name}_comparison.jpg")
    else:
        print(f"❌ Failed to create visualization for {name}")

def batch_visualizations(img_dir="data/icdar2015/test_images", 
                        east_dir="outputs/east_eval_ready",
                        craft_dir="outputs/craft_eval_ready", 
                        fused_dir="outputs/ensemble_eval_ready",
                        out_dir="outputs/visualizations/batch",
                        limit=20):
    """Create comparison panels for multiple images"""
    
    print(f"🎨 Creating batch visualizations (limit: {limit if limit else 'all'})")
    
    # Create output directory
    os.makedirs(out_dir, exist_ok=True)
    
    # Get list of image names
    img_pattern = os.path.join(img_dir, "img_*.jpg")
    img_paths = sorted(glob(img_pattern))
    
    if not img_paths:
        print(f"❌ No images found in {img_dir}")
        return
    
    # Extract image names
    names = [os.path.splitext(os.path.basename(p))[0] for p in img_paths]
    
    # Apply limit if specified
    if limit:
        names = names[:limit]
    
    print(f"📊 Processing {len(names)} images...")
    
    success_count = 0
    
    for i, name in enumerate(names, 1):
        print(f"[{i:3d}/{len(names)}] Processing {name}...", end=" ")
        
        success = create_comparison_panel(
            img_path=os.path.join(img_dir, f"{name}.jpg"),
            east_txt=os.path.join(east_dir, f"{name}.txt"),
            craft_txt=os.path.join(craft_dir, f"{name}.txt"),
            fused_txt=os.path.join(fused_dir, f"{name}.txt"),
            save_path=os.path.join(out_dir, f"{name}_comparison.jpg")
        )
        
        if success:
            print("✅")
            success_count += 1
        else:
            print("❌")
    
    print(f"\n🎯 Batch visualization complete!")
    print(f"   Successful: {success_count}/{len(names)}")
    print(f"   Output directory: {out_dir}")

def create_overlay_visualization(name="img_103"):
    """Create single image with all three models overlaid"""
    
    print(f"🎨 Creating overlay visualization for {name}")
    
    # Define paths
    base_dir = "data/icdar2015/test_images"
    east_dir = "outputs/east_eval_ready"
    craft_dir = "outputs/craft_eval_ready"
    fused_dir = "outputs/ensemble_eval_ready"
    out_dir = "outputs/visualizations"
    
    os.makedirs(out_dir, exist_ok=True)
    
    # Load image and polygons
    img_path = os.path.join(base_dir, f"{name}.jpg")
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"❌ Could not load {img_path}")
        return
    
    east_polys = read_polys(os.path.join(east_dir, f"{name}.txt"))
    craft_polys = read_polys(os.path.join(craft_dir, f"{name}.txt"))
    fused_polys = read_polys(os.path.join(fused_dir, f"{name}.txt"))
    
    # Draw all polygons on same image
    result = img.copy()
    
    # Draw EAST (red, thin)
    for poly in east_polys:
        pts = poly.astype(int).reshape(-1, 1, 2)
        cv2.polylines(result, [pts], True, (0, 0, 255), 1)
    
    # Draw CRAFT (blue, medium)
    for poly in craft_polys:
        pts = poly.astype(int).reshape(-1, 1, 2)
        cv2.polylines(result, [pts], True, (255, 0, 0), 2)
    
    # Draw FUSED (green, thick)
    for poly in fused_polys:
        pts = poly.astype(int).reshape(-1, 1, 2)
        cv2.polylines(result, [pts], True, (0, 200, 0), 3)
    
    # Add legend
    legend_height = 120
    legend = np.zeros((legend_height, result.shape[1], 3), dtype=np.uint8)
    
    # Legend text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    
    cv2.putText(legend, f"EAST: {len(east_polys)} boxes", (20, 30), font, font_scale, (0, 0, 255), thickness)
    cv2.putText(legend, f"CRAFT: {len(craft_polys)} boxes", (20, 60), font, font_scale, (255, 0, 0), thickness)
    cv2.putText(legend, f"FUSED: {len(fused_polys)} boxes", (20, 90), font, font_scale, (0, 200, 0), thickness)
    
    # Combine image and legend
    final_result = np.concatenate([result, legend], axis=0)
    
    # Save
    save_path = os.path.join(out_dir, f"{name}_overlay.jpg")
    cv2.imwrite(save_path, final_result)
    print(f"✅ Saved overlay: {save_path}")

def main():
    """Main visualization function with menu"""
    
    print("🎨 STEP 8: VISUALIZATION")
    print("=" * 50)
    
    # Check if required directories exist
    required_dirs = [
        "data/icdar2015/test_images",
        "outputs/east_eval_ready", 
        "outputs/craft_eval_ready",
        "outputs/ensemble_eval_ready"
    ]
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            print(f"❌ Required directory not found: {dir_path}")
            print("Please run the evaluation preparation first.")
            return
    
    print("📁 All required directories found!")
    
    # Create visualizations
    print("\n1️⃣ Creating single image comparison (img_103)...")
    create_single_visualization("img_103")
    
    print("\n2️⃣ Creating overlay visualization (img_103)...")
    create_overlay_visualization("img_103")
    
    print("\n3️⃣ Creating batch comparisons (first 20 images)...")
    batch_visualizations(limit=20)
    
    print("\n🎯 VISUALIZATION COMPLETE!")
    print("=" * 50)
    print("📁 Generated visualizations:")
    print("   - outputs/visualizations/img_103_comparison.jpg (side-by-side)")
    print("   - outputs/visualizations/img_103_overlay.jpg (overlaid)")
    print("   - outputs/visualizations/batch/ (20 comparison panels)")
    
    print("\n💡 Visualization Legend:")
    print("   🔴 Red boxes: EAST detections")
    print("   🔵 Blue boxes: CRAFT detections")
    print("   🟢 Green boxes: FUSED (ensemble) detections")

if __name__ == "__main__":
    main()