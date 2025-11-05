import cv2, os, numpy as np
from glob import glob

MODEL_PATH = "models/frozen_east_text_detection.pb"
IMG_DIR    = "data/icdar2015/test_images"
SAVE_DIR   = "outputs/east_results"
os.makedirs(SAVE_DIR, exist_ok=True)

# Tunables
INPUT_SIZE   = 640        # try 320 first; 640 if text is small
SCORE_THRESH = 0.50       # 0.5 is a good starting point
NMS_THRESH   = 0.40
MIN_HEIGHT   = 10         # pixels after rescale to original
MIN_AREA     = 80         # pixels^2 after rescale
AR_MIN, AR_MAX = 0.1, 15  # width/height ratio bounds

def decode(scores, geometry, scoreThresh=SCORE_THRESH):
    # Standard EAST axis-aligned decode (OpenCV-style)
    (numRows, numCols) = scores.shape[2:4]
    rects, confidences = [], []
    for y in range(numRows):
        scoresData = scores[0, 0, y]
        x0 = geometry[0, 0, y]
        x1 = geometry[0, 1, y]
        x2 = geometry[0, 2, y]
        x3 = geometry[0, 3, y]
        angles = geometry[0, 4, y]
        for x in range(numCols):
            score = float(scoresData[x])
            if score < scoreThresh:
                continue
            angle = angles[x]
            cos, sin = np.cos(angle), np.sin(angle)
            h = x0[x] + x2[x]
            w = x1[x] + x3[x]
            offsetX, offsetY = x * 4.0, y * 4.0
            endX = int(offsetX + (cos * x1[x]) + (sin * x2[x]))
            endY = int(offsetY - (sin * x1[x]) + (cos * x2[x]))
            startX, startY = int(endX - w), int(endY - h)
            rects.append((startX, startY, endX, endY))
            confidences.append(score)
    return rects, confidences

def clamp_box(x1, y1, x2, y2, W, H):
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W-1, x2), min(H-1, y2)
    return x1, y1, x2, y2

def box_ok(x1, y1, x2, y2):
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0: return False
    if w*h < MIN_AREA:   return False
    if h < MIN_HEIGHT:   return False
    ar = w / float(h)
    return (AR_MIN <= ar <= AR_MAX)

print("Loading EAST model…")
net = cv2.dnn.readNet(MODEL_PATH)
layer_names = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

image_paths = sorted(glob(os.path.join(IMG_DIR, "*.jpg")))
print(f"Found {len(image_paths)} images")

processed_count = 0
failed_count = 0

for idx, p in enumerate(image_paths, 1):
    try:
        print(f"Processing image {idx}/{len(image_paths)}: {os.path.basename(p)}")
        
        orig = cv2.imread(p)
        if orig is None: 
            failed_count += 1
            print(f"  ❌ Failed to load image")
            continue
            
        H, W = orig.shape[:2]

        newW = (INPUT_SIZE // 32) * 32
        newH = (INPUT_SIZE // 32) * 32
        rW, rH = W / float(newW), H / float(newH)

        image = cv2.resize(orig, (newW, newH))
        blob = cv2.dnn.blobFromImage(image, 1.0, (newW, newH),
                                     (123.68,116.78,103.94), swapRB=True, crop=False)
        net.setInput(blob)
        scores, geometry = net.forward(layer_names)

        rects, confs = decode(scores, geometry, SCORE_THRESH)
        idxs = cv2.dnn.NMSBoxes(rects, confs, SCORE_THRESH, NMS_THRESH)

        drawn = orig.copy()
        base  = os.path.splitext(os.path.basename(p))[0]
        txt_path = os.path.join(SAVE_DIR, f"{base}_east_boxes.txt")
        
        valid_boxes = 0
        with open(txt_path, "w") as f:
            if len(idxs) > 0:
                for i in np.array(idxs).flatten():
                    x1, y1, x2, y2 = rects[i]
                    # scale back to original
                    x1, y1 = int(x1 * rW), int(y1 * rH)
                    x2, y2 = int(x2 * rW), int(y2 * rH)
                    x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, W, H)
                    if not box_ok(x1, y1, x2, y2): 
                        continue
                    # draw & save as 4-point polygon + score
                    cv2.rectangle(drawn, (x1,y1), (x2,y2), (0,255,0), 2)
                    score = confs[i]
                    poly = [x1,y1, x2,y1, x2,y2, x1,y2]
                    f.write(",".join(map(str, poly + [score])) + "\n")
                    valid_boxes += 1

        cv2.imwrite(os.path.join(SAVE_DIR, f"{base}_east_result.jpg"), drawn)
        processed_count += 1
        print(f"  ✅ Detected {valid_boxes} text regions")
        
    except Exception as e:
        failed_count += 1
        print(f"  ❌ Error processing {os.path.basename(p)}: {str(e)}")

print(f"\n🎉 EAST processing completed!")
print(f"✅ Successfully processed: {processed_count} images")
print(f"❌ Failed: {failed_count} images")
print(f"📁 Results saved to: {SAVE_DIR}")
print(f"[EAST] Cleaned results saved to {SAVE_DIR}")