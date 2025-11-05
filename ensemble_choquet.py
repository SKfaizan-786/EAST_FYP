import os, argparse, glob, numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
import cv2

def read_boxes(txt_path):
    boxes = []
    if not os.path.exists(txt_path):
        return boxes
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) not in (8,9):
                continue
            pts = list(map(float, parts[:8]))
            poly = np.array(pts, dtype=np.float32).reshape(4,2)
            score = float(parts[8]) if len(parts) == 9 else 0.5
            boxes.append((poly, score))
    return boxes

def poly_iou(p1, p2):
    try:
        P1, P2 = Polygon(p1), Polygon(p2)
        if not P1.is_valid or not P2.is_valid:
            return 0.0
        inter = P1.intersection(P2).area
        union = unary_union([P1, P2]).area
        return 0.0 if union == 0 else float(inter / union)
    except:
        return 0.0

def choquet_two(x1, x2, a, b, c):
    # a = g({EAST}), b = g({CRAFT}), c = g({EAST,CRAFT}), with max(a,b) <= c <= 1
    x = np.array([x1, x2], dtype=np.float32)
    order = np.argsort(x)  # ascending
    x_sorted = x[order]
    # map index to measure
    # subset for smaller is the set of top2 indices (both)
    # subset for larger is the singleton of the larger one
    if order[0] == 0: m1 = c       # A_(1) = {0,1}
    else:             m1 = c
    m2 = b if order[1] == 1 else a
    return (x_sorted[0] - 0.0) * m1 + (x_sorted[1] - x_sorted[0]) * m2

def fuse_poly(p1, s1, p2, s2):
    w1 = float(s1); w2 = float(s2)
    if w1 + w2 == 0: 
        return p1
    return (w1 * p1 + w2 * p2) / (w1 + w2)

def draw_polys(img, polys, color, thick=2):
    out = img.copy()
    for p in polys:
        pts = p.astype(int).reshape(-1,1,2)
        cv2.polylines(out, [pts], True, color, thick)
    return out

def main(args):
    os.makedirs(args.out, exist_ok=True)
    east_files = sorted(glob.glob(os.path.join(args.east, "*_east_boxes.txt")))
    craft_files = sorted(glob.glob(os.path.join(args.craft, "*_craft_boxes.txt")))
    # build index by image base name
    idx_e = {os.path.basename(p).replace("_east_boxes.txt",""): p for p in east_files}
    idx_c = {os.path.basename(p).replace("_craft_boxes.txt",""): p for p in craft_files}
    names = sorted(set(idx_e.keys()) | set(idx_c.keys()))
    print(f"Found {len(names)} images to fuse")

    a, b, c = args.a, args.b, max(args.c, max(args.a, args.b))  # enforce monotonicity lightly
    fused_count = 0

    for name in names:
        east_boxes = read_boxes(idx_e.get(name, ""))
        craft_boxes = read_boxes(idx_c.get(name, ""))

        used_e = set()
        used_c = set()
        fused = []

        # Greedy matching by IoU
        for ei, (ep, es) in enumerate(east_boxes):
            best_iou, best_cj = 0.0, -1
            for cj, (cp, cs) in enumerate(craft_boxes):
                if cj in used_c: 
                    continue
                iou = poly_iou(ep, cp)
                if iou > best_iou:
                    best_iou, best_cj = iou, cj
            if best_iou >= args.iou:
                used_e.add(ei); used_c.add(best_cj)
                cp, cs = craft_boxes[best_cj]
                fused_score = choquet_two(es, cs, a, b, c)
                fused_poly  = fuse_poly(ep, es, cp, cs)
                fused.append((fused_poly, fused_score))

        # Add unmatched EAST and CRAFT (optionally discount their scores)
        for ei, (ep, es) in enumerate(east_boxes):
            if ei in used_e: continue
            fused.append((ep, es * args.singleton_discount))
        for cj, (cp, cs) in enumerate(craft_boxes):
            if cj in used_c: continue
            fused.append((cp, cs * args.singleton_discount))

        # Optional final NMS (polygon IoU)
        keep = []
        for i in range(len(fused)):
            keep_it = True
            for j in keep:
                if poly_iou(fused[i][0], fused[j][0]) > args.final_nms:
                    # keep higher score
                    if fused[i][1] <= fused[j][1]:
                        keep_it = False
                        break
                    else:
                        keep.remove(j)
                        break
            if keep_it: keep.append(i)

        fused = [fused[i] for i in keep]

        # Save text file
        out_txt = os.path.join(args.out, f"{name}_fused.txt")
        with open(out_txt, "w") as f:
            for poly, sc in fused:
                flat = poly.reshape(-1)
                line = ",".join([f"{int(v)}" for v in flat]) + f",{float(sc):.4f}\n"
                f.write(line)

        # Optional visualization
        if args.draw:
            # try to find any original image location
            # prefer craft result image; fall back to east input dir image
            img_path = None
            for root in [args.imgs, os.path.dirname(idx_c.get(name,"")), os.path.dirname(idx_e.get(name,""))]:
                candidate = os.path.join(root, f"{name}.jpg")
                if os.path.exists(candidate): 
                    img_path = candidate; break
            if img_path:
                img = cv2.imread(img_path)
                vis = draw_polys(img, [p for p,_ in fused], (0,255,0), 2)
                cv2.imwrite(os.path.join(args.out, f"{name}_fused.jpg"), vis)

        fused_count += 1

    print(f"Done. Fused {fused_count} images -> {args.out}")
    print(f"Choquet params: a={a:.2f} (EAST), b={b:.2f} (CRAFT), c={c:.2f} (joint), IoU match={args.iou}, singleton_discount={args.singleton_discount}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--east", required=True, help="dir with *_east_boxes.txt")
    ap.add_argument("--craft", required=True, help="dir with *_craft_boxes.txt")
    ap.add_argument("--imgs", default="data/icdar2015/test_images", help="original images dir (for drawing)")
    ap.add_argument("--out", default="outputs/ensemble_results")
    ap.add_argument("--iou", type=float, default=0.5, help="IoU to match EAST/CRAFT boxes")
    ap.add_argument("--final_nms", type=float, default=0.3, help="final NMS IoU on fused set")
    ap.add_argument("--singleton_discount", type=float, default=0.9, help="weight for unmatched boxes")
    ap.add_argument("--a", type=float, default=0.6, help="g({EAST})")
    ap.add_argument("--b", type=float, default=0.7, help="g({CRAFT})")
    ap.add_argument("--c", type=float, default=0.9, help="g({EAST,CRAFT}) must be >= max(a,b)")
    ap.add_argument("--draw", action="store_true")
    args = ap.parse_args()
    main(args)