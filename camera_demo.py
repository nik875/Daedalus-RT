"""
Real-time camera demo: YOLO26n detection with toggleable universal adversarial
perturbation.

Controls:
  a  toggle the universal attack on / off
  q  quit

Usage:
    python camera_demo.py
    python camera_demo.py --model yolo26n.pt --mlx      # native MLX backend
    python camera_demo.py --delta adv_examples/yolo26_universal/delta_final.npy
"""

import argparse
import os
import time

import cv2
import numpy as np
from ultralytics.utils.plotting import Colors

DELTA_PATH = "adv_examples/yolo26_universal/delta_final.npy"
MODEL_PATH = "yolo26n.pt"
IMG_SIZE = 640

_COLORS = Colors()

COCO80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


def _resolve_name(names, cls_id):
    cand = None
    if isinstance(names, dict):
        cand = names.get(cls_id, names.get(str(cls_id)))
    elif isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
        cand = names[cls_id]
    if cand is not None:
        s = str(cand).strip().lower()
        if not (s.startswith("class") or s.startswith("cls")):
            return str(cand)
    return COCO80[cls_id] if 0 <= cls_id < len(COCO80) else f"class{cls_id}"


def _annotate(result, is_mlx):
    """Draw per-class coloured boxes on result.orig_img; returns BGR numpy array."""
    base = result.orig_img
    img = cv2.cvtColor(base, cv2.COLOR_RGB2BGR) if is_mlx else base.copy()
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return img
    lw = max(round(sum(img.shape[:2]) / 2 * 0.003), 2)
    font, fs, tf = cv2.FONT_HERSHEY_SIMPLEX, lw / 3.0, max(lw - 1, 1)
    for i in range(len(boxes)):
        x1, y1, x2, y2 = (int(v) for v in boxes.xyxy[i])
        cls_id = int(boxes.cls[i])
        color = _COLORS(cls_id, bgr=True)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, lw, cv2.LINE_AA)
        label = f"{_resolve_name(result.names, cls_id)} {float(boxes.conf[i]):.2f}"
        (tw, th), _ = cv2.getTextSize(label, font, fs, tf)
        outside = y1 - th - 3 >= 0
        cv2.rectangle(img, (x1, y1 - th - 3 if outside else y1),
                      (x1 + tw, y1 if outside else y1 + th + 3), color, -1, cv2.LINE_AA)
        txt_color = (255, 255, 255) if sum(color) < 384 else (0, 0, 0)
        cv2.putText(img, label, (x1, y1 - 2 if outside else y1 + th + 2),
                    font, fs, txt_color, tf, cv2.LINE_AA)
    return img


def _load_model(path, force_mlx=False):
    if force_mlx or path.lower().endswith((".npz", ".safetensors")):
        from yolo26mlx import YOLO
        if path.lower().endswith(".pt"):
            npz_path = os.path.splitext(path)[0] + ".npz"
            if not os.path.exists(npz_path):
                from yolo26mlx.converters.convert import convert_yolo26_weights
                convert_yolo26_weights(path, npz_path, verbose=False)
            path = npz_path
        return YOLO(path), True
    from ultralytics import YOLO
    return YOLO(path), False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--delta", default=DELTA_PATH,
                        help="universal delta .npy to apply when attack is toggled on")
    parser.add_argument("--model", default=MODEL_PATH,
                        help="YOLO weights (e.g. yolo26n.pt, yolo26n.npz)")
    parser.add_argument("--mlx", action="store_true",
                        help="force the native MLX backend (auto-converts .pt to .npz)")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--cam", type=int, default=0, help="camera device index")
    args = parser.parse_args()

    print(f"Loading delta: {args.delta}")
    delta = np.load(args.delta).astype(np.float32)   # (640, 640, 3) RGB, signed

    print(f"Loading model: {args.model}")
    model, is_mlx = _load_model(args.model, force_mlx=args.mlx)
    predict_kwargs = {"imgsz": IMG_SIZE, "conf": args.conf}
    if not is_mlx:
        predict_kwargs["verbose"] = False

    # Warm up — first inference allocates GPU memory and is always slow.
    print("Warming up ...")
    dummy = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    model.predict(dummy, **predict_kwargs)

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {args.cam}")

    attack_on = False
    fps = 0.0
    ema = 0.2  # smoothing factor for FPS

    print("Running. Press 'a' to toggle attack, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Always work at 640×640 so the delta lines up exactly.
        frame_sq = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))

        if attack_on:
            rgb = cv2.cvtColor(frame_sq, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            adv = np.clip(rgb + delta, 0.0, 1.0)
            inp = cv2.cvtColor((adv * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        else:
            inp = frame_sq

        t0 = time.perf_counter()
        result = model.predict(inp, **predict_kwargs)[0]
        dt = time.perf_counter() - t0
        fps = (1.0 - ema) * fps + ema * (1.0 / max(dt, 1e-6))

        ann = cv2.resize(_annotate(result, is_mlx), (IMG_SIZE, IMG_SIZE))

        # Status bar
        n = len(result.boxes)
        if attack_on:
            bar = f"ATTACK ON  [a]  |  {n} det  |  {fps:.1f} fps"
            bar_color = (0, 60, 220)
        else:
            bar = f"attack off [a]  |  {n} det  |  {fps:.1f} fps"
            bar_color = (40, 40, 40)
        cv2.rectangle(ann, (0, 0), (ann.shape[1], 30), bar_color, -1)
        cv2.putText(ann, bar, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Daedalus demo", ann)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("a"):
            attack_on = not attack_on
            print(f"Attack {'ON' if attack_on else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
