
import os
import cv2
import sys
import time
import math
import argparse
import numpy as np
from pathlib import Path
from collections import deque, defaultdict

# ---- Try to import user's BRNN model (Important Object Selector) from Step6 file ----
BRNN = None
def _import_brnn():
    global BRNN
    try:
        from Step6_train_and_evaluate_Important_Object_Selector import BRNN as _BRNN
        BRNN = _BRNN
        return True
    except Exception as e:
        print("[WARN] Could not import BRNN from Step6 file:", e)
        return False

# ---- Minimal SORT-like tracking using simple IoU + persistence ----
class SimpleTracker:
    def __init__(self, max_lost=30):
        self.next_id = 1
        self.tracks = {}        # id -> {'bbox':(x1,y1,x2,y2), 'lost':0, 'history':deque, 'cls':int}
        self.max_lost = max_lost

    @staticmethod
    def iou(a, b):
        xA = max(a[0], b[0]); yA = max(a[1], b[1])
        xB = min(a[2], b[2]); yB = min(a[3], b[3])
        inter = max(0, xB-xA) * max(0, yB-yA)
        if inter <= 0: return 0.0
        areaA = max(0,(a[2]-a[0]))*max(0,(a[3]-a[1]))
        areaB = max(0,(b[2]-b[0]))*max(0,(b[3]-b[1]))
        return inter / max(1e-6, areaA + areaB - inter)

    def update(self, detections, classes):
        assigned = set()
        # Match existing tracks with detections by IoU
        for tid, tr in list(self.tracks.items()):
            best_iou, best_j = 0.0, -1
            for j, box in enumerate(detections):
                if j in assigned: continue
                iou = self.iou(tr['bbox'], box)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou > 0.3:
                self.tracks[tid]['bbox'] = detections[best_j]
                self.tracks[tid]['lost'] = 0
                self.tracks[tid]['cls'] = classes[best_j]
                self.tracks[tid]['history'].append(detections[best_j])
                assigned.add(best_j)
            else:
                self.tracks[tid]['lost'] += 1

        # Create new tracks
        for j, box in enumerate(detections):
            if j in assigned: continue
            tid = self.next_id; self.next_id += 1
            self.tracks[tid] = {
                'bbox': box,
                'lost': 0,
                'cls': classes[j],
                'history': deque([box], maxlen=120)
            }

        # Remove lost
        for tid in list(self.tracks.keys()):
            if self.tracks[tid]['lost'] > self.max_lost:
                del self.tracks[tid]

        return self.tracks

# ---- YOLOv4-tiny OpenCV DNN ----
def load_yolov4_tiny(model_dir):
    cfg = str(Path(model_dir)/"yolov4-tiny.cfg")
    weights = str(Path(model_dir)/"yolov4-tiny.weights")
    names = str(Path(model_dir)/"coco.names")
    if not (os.path.exists(cfg) and os.path.exists(weights) and os.path.exists(names)):
        raise FileNotFoundError("Missing YOLOv4-tiny cfg/weights/names in: " + model_dir)
    net = cv2.dnn.readNetFromDarknet(cfg, weights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    with open(names, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    ln = net.getLayerNames()
    ln = [ln[i[0]-1] for i in net.getUnconnectedOutLayers()]
    return net, ln, class_names

def yolo_detect(net, ln, frame, conf_thr=0.3, nms_thr=0.4, input_size=416):
    (H, W) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (input_size, input_size), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    boxes, confidences, classIDs = [], [], []
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = int(np.argmax(scores))
            confidence = float(scores[classID])
            if confidence > conf_thr:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(confidence)
                classIDs.append(classID)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_thr, nms_thr)
    dets, clses = [], []
    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y, w, h = boxes[i]
            dets.append((max(0,x), max(0,y), min(W-1,x+w), min(H-1,y+h)))
            clses.append(classIDs[i])
    return dets, clses

def track_to_features(track_history, imgW, imgH, cls_id_norm):
    feats = []
    for (x1,y1,x2,y2) in track_history:
        xc = (x1+x2)/2.0; yc = (y1+y2)/2.0
        wd = (x2-x1); ht = (y2-y1)
        feats.append([xc/imgW, yc/imgH, wd/imgW, ht/imgH, cls_id_norm])
    return np.asarray(feats, dtype=np.float32)

def estimate_behavior(prev_gray, gray):
    if prev_gray is None: return '...'
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5,3,15,3,5,1.2,0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1], angleInDegrees=True)
    mean_mag = float(np.mean(mag))
    horiz = float(np.mean(np.abs(flow[...,0])))
    vert  = float(np.mean(np.abs(flow[...,1])))
    if mean_mag < 0.6:
        return 'Slowing Down'
    if horiz > 0.7*vert:
        return 'Deviate'
    return 'Turning and Slowing Down'

def explain(io_track_hist, all_tracks_framecount, imgW, imgH):
    if len(io_track_hist) < 2: return 'Obstruction'
    last = io_track_hist[-1]; prev = io_track_hist[0]
    area_last = (last[2]-last[0])*(last[3]-last[1]) + 1e-3
    area_prev = (prev[2]-prev[0])*(prev[3]-prev[1]) + 1e-3
    growth = area_last/area_prev
    x_last = (last[0]+last[2])/2.0; x_prev = (prev[0]+prev[2])/2.0
    dx = (x_last - x_prev)/imgW
    y_last = (last[1]+last[3])/2.0
    ahead = y_last < (imgH*0.55)
    density = all_tracks_framecount
    if ahead and growth > 1.35 and abs(dx) < 0.03:
        return 'Overtake'
    if density >= 6 and growth < 1.2:
        return 'Congestion'
    if ahead and growth < 1.05:
        return 'Obstruction'
    if abs(dx) >= 0.06:
        return 'Cut-in'
    return 'Avoid Obstruction'

def main():
    ok = _import_brnn()
    p = argparse.ArgumentParser()
    p.add_argument('-i','--input', required=True, help='Input .mp4 path')
    p.add_argument('-o','--output', default='demo_outputs\\full_demo.mp4', help='Output .mp4 path')
    p.add_argument('--yolo_dir', default='yolo', help='Folder with yolov4-tiny.cfg/weights and coco.names')
    p.add_argument('--stride', type=int, default=1, help='Process every Nth frame')
    p.add_argument('--maxframes', type=int, default=600, help='Limit frames for speed')
    p.add_argument('--brnn_ckpt', default='models\\classConditionedBiGRU_ImportantObjectSelector.ckpt', help='BRNN checkpoint')
    args = p.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print("Input not found:", in_path); sys.exit(1)

    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        print("Cannot open video."); sys.exit(1)

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 20.0

    out_dir = Path(os.path.dirname(args.output) or '.'); out_dir.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))

    print("[INFO] Loading YOLOv4-tiny from", args.yolo_dir)
    net, ln, class_names = load_yolov4_tiny(args.yolo_dir)

    brnn_model = None
    if ok and os.path.exists(args.brnn_ckpt):
        import torch
        brnn_model = BRNN(input_size=5, hidden_size=5, num_layers=1, num_classes=2)
        state = torch.load(args.brnn_ckpt, map_location='cpu')
        key = 'state_dict' if 'state_dict' in state else None
        if key:
            brnn_model.load_state_dict(state[key])
        else:
            brnn_model.load_state_dict(state)
        brnn_model.eval()
        print("[INFO] BRNN loaded.")
    else:
        print("[WARN] BRNN not available; will mark IO by largest area.")

    tracker = SimpleTracker(max_lost=30)
    prev_gray = None
    frame_idx = 0
    per_track_hist = defaultdict(lambda: deque(maxlen=60))

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1
        if frame_idx % args.stride != 0: continue
        if frame_idx > args.maxframes: break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        dets, clses = yolo_detect(net, ln, frame, conf_thr=0.35, nms_thr=0.45, input_size=416)
        tracks = tracker.update(dets, clses)

        for tid, tr in tracks.items():
            per_track_hist[tid].append(tr['bbox'])

        io_tid = None
        if brnn_model is not None and len(tracks)>0:
            scores = {}
            for tid, tr in tracks.items():
                hist = list(per_track_hist[tid])
                if len(hist) < 4: continue
                cls_norm = (tr['cls'] % 80 + 1) / 80.0
                feats = track_to_features(hist, W, H, cls_norm)
                import torch
                x = torch.from_numpy(feats).unsqueeze(0).float()
                with torch.no_grad():
                    out = brnn_model(x, None, None, h0=None)
                    prob = torch.softmax(out, dim=1)[0,1].item()
                scores[tid] = prob
            if scores:
                io_tid = max(scores, key=scores.get)
        else:
            areas = {tid: (tr['bbox'][2]-tr['bbox'][0])*(tr['bbox'][3]-tr['bbox'][1]) for tid,tr in tracks.items()}
            if areas:
                io_tid = max(areas, key=areas.get)

        behavior = estimate_behavior(prev_gray, gray)
        prev_gray = gray

        important_text = 'None'
        explanation = '...'
        for tid, tr in tracks.items():
            (x1,y1,x2,y2) = tr['bbox']
            color = (60, 60, 255)
            if tid == io_tid:
                color = (60, 220, 60)
                important_text = f'ID {tid}'
                hist = list(per_track_hist[tid])
                explanation = explain(hist, len(tracks), W, H)
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, f'#{tid}', (x1, max(20,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.putText(frame, f'Ego Behavior: {behavior}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        cv2.putText(frame, f'Important Obj: {important_text}', (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        cv2.putText(frame, f'Explanation: {explanation}', (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0), 2)

        out.write(frame)

    cap.release(); out.release()
    print("✅ Demo saved to", args.output)

if __name__ == "__main__":
    main()
