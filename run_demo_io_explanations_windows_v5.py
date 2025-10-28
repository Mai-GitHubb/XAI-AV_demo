import os
import sys
import types
import argparse
import json
from pathlib import Path
from collections import deque, defaultdict

import numpy as np
import cv2
import torch
import torch.nn as nn

# ---------- Try to import user's BRNN from Step6, mocking mmaction/mmcv if missing ----------
BRNN = None

def import_brnn():
    """Try (1) native import; (2) with mocked mmaction/mmcv; else we'll use MinimalBRNN fallback."""
    global BRNN
    try:
        from Step6_train_and_evaluate_Important_Object_Selector import BRNN as _BRNN
        BRNN = _BRNN
        return "native"
    except Exception:
        try:
            # Mock modules so Step6 can be imported without mmaction/mmcv installed.
            sys.modules['mmaction'] = types.ModuleType('mmaction')
            sys.modules['mmcv'] = types.ModuleType('mmcv')
            from Step6_train_and_evaluate_Important_Object_Selector import BRNN as _BRNN
            BRNN = _BRNN
            return "mocked"
        except Exception:
            return "minimal"

# ---------- Ultralytics YOLOv8 detector ----------
def load_yolov8(model_name='yolov8n.pt', device='cpu'):
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError("Ultralytics not installed. Run: pip install ultralytics") from e
    model = YOLO(model_name)   # auto-downloads yolov8n.pt if missing
    model.to(device if device.startswith('cuda') else 'cpu')
    return model

def yolo_detect_ultralytics(model, frame):
    """Returns: dets=[(x1,y1,x2,y2), ...], clses=[int,...]"""
    res = model.predict(source=frame, verbose=False)[0]
    dets, clses = [], []
    if res.boxes is None:
        return dets, clses
    for b in res.boxes:
        xyxy = b.xyxy[0].cpu().numpy().astype(int)
        cls = int(b.cls.item())
        dets.append((int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])))
        clses.append(cls)
    return dets, clses

# ---------- Simple IoU tracker ----------
class SimpleTracker:
    def __init__(self, max_lost=30):
        self.next_id = 1
        self.tracks = {}  # id -> {'bbox':(x1,y1,x2,y2), 'lost':0, 'history':deque, 'cls':int}
        self.max_lost = max_lost

    @staticmethod
    def iou(a, b):
        xA = max(a[0], b[0]); yA = max(a[1], b[1])
        xB = min(a[2], b[2]); yB = min(a[3], b[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        if inter <= 0: return 0.0
        areaA = max(0, (a[2]-a[0])) * max(0, (a[3]-a[1]))
        areaB = max(0, (b[2]-b[0])) * max(0, (b[3]-b[1]))
        return inter / max(1e-6, areaA + areaB - inter)

    def update(self, dets, clses):
        assigned = set()
        # match tracks to dets by IoU
        for tid, tr in list(self.tracks.items()):
            best_iou, best_j = 0.0, -1
            for j, box in enumerate(dets):
                if j in assigned: continue
                iou = self.iou(tr['bbox'], box)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou > 0.3:
                self.tracks[tid]['bbox'] = dets[best_j]
                self.tracks[tid]['lost'] = 0
                self.tracks[tid]['cls']  = clses[best_j]
                self.tracks[tid]['history'].append(dets[best_j])
                assigned.add(best_j)
            else:
                self.tracks[tid]['lost'] += 1
        # new tracks
        for j, box in enumerate(dets):
            if j in assigned: continue
            tid = self.next_id; self.next_id += 1
            self.tracks[tid] = {'bbox': box, 'lost': 0, 'cls': clses[j],
                                'history': deque([box], maxlen=60)}
        # drop lost
        for tid in list(self.tracks.keys()):
            if self.tracks[tid]['lost'] > self.max_lost:
                del self.tracks[tid]
        return self.tracks

# ---------- Behavior (optical flow heuristic) ----------
def estimate_behavior(prev_gray, gray):
    if prev_gray is None: return '...'
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1], angleInDegrees=True)
    mean_mag = float(np.mean(mag))
    horiz = float(np.mean(np.abs(flow[...,0])))
    vert  = float(np.mean(np.abs(flow[...,1])))
    if mean_mag < 0.6: return 'Slowing Down'
    if horiz > 0.7 * vert: return 'Deviate'
    return 'Turning and Slowing Down'

# ---------- Explanation (rule-based) ----------
def explain(io_hist, n_tracks, W, H):
    if len(io_hist) < 2: return 'Obstruction'
    last = io_hist[-1]; prev = io_hist[0]
    area_last = (last[2]-last[0]) * (last[3]-last[1]) + 1e-3
    area_prev = (prev[2]-prev[0]) * (prev[3]-prev[1]) + 1e-3
    growth = area_last / area_prev
    x_last = (last[0]+last[2]) / 2.0; x_prev = (prev[0]+prev[2]) / 2.0
    dx = (x_last - x_prev) / W
    y_last = (last[1]+last[3]) / 2.0
    ahead = y_last < (H * 0.55)
    if ahead and growth > 1.35 and abs(dx) < 0.03: return 'Overtake'
    if n_tracks >= 6 and growth < 1.2: return 'Congestion'
    if ahead and growth < 1.05: return 'Obstruction'
    if abs(dx) >= 0.06: return 'Cut-in'
    return 'Avoid Obstruction'

# ---------- MinimalBRNN fallback that matches checkpoint dims ----------
class MinimalBRNN(nn.Module):
    def __init__(self, input_size=5, hidden_size=5, num_layers=1, num_classes=2, bidirectional=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        fc_in = hidden_size * (2 if bidirectional else 1)  # 10 if H=5 and bi=True
        self.fc = nn.Linear(fc_in, num_classes)

    def forward(self, x, *args, **kwargs):
        # x: [B, T, 5]
        _, h_n = self.gru(x)   # [num_layers*(1+bi), B, H]
        if self.bidirectional:
            h_f = h_n[-2]      # forward last layer
            h_b = h_n[-1]      # backward last layer
            h = torch.cat([h_f, h_b], dim=1)  # [B, 2H]
        else:
            h = h_n[-1]        # [B, H]
        return self.fc(h)      # [B, C]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i','--input', required=True, help='Input video path')
    ap.add_argument('-o','--output', default='demo_outputs\\full_demo.mp4', help='Output video path')
    ap.add_argument('--device', default='cpu', help='cpu or cuda:0 (if GPU)')
    ap.add_argument('--stride', type=int, default=1, help='process every Nth frame')
    ap.add_argument('--maxframes', type=int, default=600, help='limit frames for speed')
    ap.add_argument('--flow_downscale', type=int, default=1, help='downscale factor for optical flow (1 = no downscale)')
    ap.add_argument('--brnn_ckpt', default='models\\classConditionedBiGRU_ImportantObjectSelector.ckpt')
    ap.add_argument('--yolo_model', default='yolov8n.pt', help='Ultralytics model name/path')
    args = ap.parse_args()

    # Torch device
    torch_device = args.device if args.device.startswith('cuda') else 'cpu'

    # Detector (YOLOv8 uses same device)
    yolo = load_yolov8(args.yolo_model, args.device)

    # Load BRNN (user's class or MinimalBRNN) and match checkpoint dims
    mode = import_brnn()
    brnn_model = None
    if mode != "none":
        try:
            state = torch.load(args.brnn_ckpt, map_location='cpu')
            sd = state['state_dict'] if isinstance(state, dict) and 'state_dict' in state else state

            # Infer dims from checkpoint tensors
            ih = sd.get('gru.weight_ih_l0', None)
            if ih is None:
                raise RuntimeError("gru.weight_ih_l0 not found in ckpt")
            hidden_size = ih.shape[0] // 3          # (3*H) x input_size  -> H
            input_size = ih.shape[1]                # -> 5
            bidir = 'gru.weight_ih_l0_reverse' in sd

            BRNNClass = BRNN if BRNN is not None and mode in ("native", "mocked") else MinimalBRNN
            # Instantiate with inferred sizes
            if BRNNClass is MinimalBRNN:
                brnn_model = BRNNClass(input_size=input_size, hidden_size=hidden_size,
                                       num_layers=1, num_classes=2, bidirectional=bidir)
            else:
                # User BRNN signature: BRNN(input_size, hidden_size, num_layers, num_classes)
                brnn_model = BRNNClass(input_size=input_size, hidden_size=hidden_size,
                                       num_layers=1, num_classes=2)

            # Load state dict (non-strict to tolerate minor key diffs)
            brnn_model.load_state_dict(sd, strict=False)
            brnn_model.to(torch_device).eval()
            print(f"[INFO] BRNN loaded (mode={mode}, hidden_size={hidden_size}, input_size={input_size}, bidirectional={bidir}) on {torch_device}.")
        except Exception as e:
            print("[WARN] Failed to load BRNN ckpt; fallback to largest-area IO. Error:", e)
            brnn_model = None
    else:
        print("[WARN] BRNN unavailable; fallback to largest-area IO.")

    # I/O
    inp = Path(args.input)
    if not inp.exists():
        print("Input not found:", inp); sys.exit(1)
    cap = cv2.VideoCapture(str(inp))
    if not cap.isOpened():
        print("Cannot open video."); sys.exit(1)

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 20.0
    out_dir = Path(os.path.dirname(args.output) or '.'); out_dir.mkdir(parents=True, exist_ok=True)
    video_out = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))

    tracker = SimpleTracker(max_lost=30)
    prev_gray_flow = None
    frame_idx = 0
    per_track_hist = defaultdict(lambda: deque(maxlen=60))

    # Per-frame JSON log
    log_data = []

    # Flow downscale handling
    flow_scale = max(1, int(args.flow_downscale))

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1
        if frame_idx % args.stride != 0: continue
        if frame_idx > args.maxframes: break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect + Track
        dets, clses = yolo_detect_ultralytics(yolo, frame)
        tracks = tracker.update(dets, clses)
        for tid, tr in tracks.items():
            per_track_hist[tid].append(tr['bbox'])

        # Choose Important object
        io_tid = None
        if brnn_model is not None and len(tracks) > 0:
            scores = {}
            for tid, tr in tracks.items():
                hist = list(per_track_hist[tid])
                if len(hist) < 4: continue
                cls_norm = (tr['cls'] % 80 + 1) / 80.0
                feats = []
                for (x1,y1,x2,y2) in hist:
                    xc = (x1+x2)/2.0; yc = (y1+y2)/2.0
                    wd = (x2-x1); ht = (y2-y1)
                    feats.append([xc/W, yc/H, wd/W, ht/H, cls_norm])
                x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(torch_device)
                with torch.no_grad():
                    logits = brnn_model(x)
                    prob = torch.softmax(logits, dim=1)[0,1].item()
                scores[tid] = prob
            if scores:
                io_tid = max(scores, key=scores.get)
        else:
            # fallback: largest box area
            areas = {tid: (tr['bbox'][2]-tr['bbox'][0])*(tr['bbox'][3]-tr['bbox'][1]) for tid,tr in tracks.items()}
            if areas: io_tid = max(areas, key=areas.get)

        # Behavior (optional downscale for speed)
        if flow_scale > 1:
            gray_small = cv2.resize(gray, (W//flow_scale, H//flow_scale))
            if prev_gray_flow is None:
                behavior = '...'
            else:
                behavior = estimate_behavior(prev_gray_flow, gray_small)
            prev_gray_flow = gray_small
        else:
            if prev_gray_flow is None:
                behavior = '...'
            else:
                behavior = estimate_behavior(prev_gray_flow, gray)
            prev_gray_flow = gray

        # Explanation (rule-based)
        important_text = 'None'
        explanation = '...'
        for tid, tr in tracks.items():
            (x1,y1,x2,y2) = tr['bbox']
            color = (60, 60, 255)
            if tid == io_tid:
                color = (60, 220, 60)
                important_text = f'ID {tid}'
                explanation = explain(list(per_track_hist[tid]), len(tracks), W, H)
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, f'#{tid}', (x1, max(20,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # --------- NEW: single sentence overlay "behavior because explanation" ----------
        sentence = f"{behavior} because {explanation}"
        cv2.putText(frame, sentence, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 0), 2, cv2.LINE_AA)

        # Keep the important object line (helpful for panel)
        cv2.putText(frame, f'Important Obj: {important_text}', (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

        # Write frame & log
        video_out.write(frame)
        log_data.append({
            "frame": frame_idx,
            "important_obj_id": int(io_tid) if io_tid is not None else None,
            "behavior": behavior,
            "explanation": explanation,
            "reason_text": sentence,
            "tracked_objects": len(tracks)
        })

    cap.release(); video_out.release()

    # Save JSON
    json_path = Path(args.output).with_suffix('.json')
    with open(json_path, 'w') as f:
        json.dump(log_data, f, indent=2)

    print(f"✅ Demo saved to {args.output}")
    print(f"🧾 Frame-level logs saved to {json_path}")

if __name__ == '__main__':
    main()
