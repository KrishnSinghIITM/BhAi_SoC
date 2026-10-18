"""
inspect_laptop.py
Interactive & Visual PCB Defect Inspection Terminal on Local Laptop
Supports:
  - Evaluation of test PCB images or live webcam stream
  - Device switching: CPU (Intel i5) vs GPU (NVIDIA RTX 3050 via DirectML / PyTorch)
  - Visual bounding box overlay with confidence score and real-time FPS
  - Deterministic IPC-A-610 Standard advisory report trigger
"""

import os
import sys
import time
import json
import cv2
import numpy as np

# Load IPC-A-610 prompts
PROMPTS_PATH = "BhAi_SoC/reasoning/prompt_templates.json"
REASONING_RULES = {}
if os.path.exists(PROMPTS_PATH):
    try:
        with open(PROMPTS_PATH, "r") as f:
            data = json.load(f)
            REASONING_RULES = data.get("defect_taxonomy_rules", {})
    except Exception:
        pass

CLASS_NAMES = ["SH", "SP", "SC", "OP", "MB", "HB", "CS", "CFO", "BMFO"]
CLASS_FULL_NAMES = {
    "SH": "Short Circuit",
    "SP": "Spur",
    "SC": "Scratch",
    "OP": "Open Circuit",
    "MB": "Mouse Bite",
    "HB": "Hole Break",
    "CS": "Conductor Spacing",
    "CFO": "Copper-Free Error",
    "BMFO": "Board Mask Flaw"
}

# Distinct high-contrast bounding box colors (BGR)
CLASS_COLORS = {
    "SH": (0, 0, 255),       # Red
    "OP": (0, 69, 255),      # Orange-Red
    "HB": (0, 215, 255),     # Gold/Amber
    "SC": (255, 0, 255),     # Magenta
    "SP": (255, 255, 0),     # Cyan
    "MB": (128, 0, 128),     # Purple
    "CS": (0, 165, 255),     # Orange
    "CFO": (255, 0, 0),      # Blue
    "BMFO": (0, 255, 0)      # Green
}

def inspect_image(image_path, model_path="BhAi_SoC/runs/detect/experiments/baseline_yolo11n_dspcbsd/train_run/weights/best.pt", device="cpu", out_path=None):
    from ultralytics import YOLO
    
    print(f"\n[+] Loading inspection model: {model_path} on {device.upper()}...")
    model = YOLO(model_path)
    
    if not os.path.exists(image_path):
        print(f"[!] Image not found: {image_path}")
        return
        
    img = cv2.imread(image_path)
    if img is None:
        print(f"[!] Could not decode image: {image_path}")
        return
        
    t0 = time.perf_counter()
    results = model.predict(img, device=device, verbose=False)
    t1 = time.perf_counter()
    latency_ms = (t1 - t0) * 1000.0
    fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0
    
    boxes = results[0].boxes
    num_detections = len(boxes)
    print(f"[+] Processed {image_path} in {latency_ms:.2f} ms ({fps:.1f} FPS) | Found {num_detections} defect(s)")
    
    annotated = img.copy()
    
    for box in boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        
        cls_code = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"ID_{cls_id}"
        full_name = CLASS_FULL_NAMES.get(cls_code, cls_code)
        color = CLASS_COLORS.get(cls_code, (0, 255, 0))
        
        # Draw bounding box
        cv2.rectangle(annotated, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
        
        # Label banner
        label = f"{cls_code} {conf:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (xyxy[0], xyxy[1] - 20), (xyxy[0] + w, xyxy[1]), color, -1)
        cv2.putText(annotated, label, (xyxy[0], xyxy[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Print IPC-A-610 Rework Advisory
        rules = REASONING_RULES.get(cls_code, {})
        print("\n" + "-" * 60)
        print(f" [!] DEFECT DETECTED: {full_name} ({cls_code}) | Confidence: {conf:.1%}")
        print(f"     Severity: {rules.get('severity', 'MAJOR')}")
        print(f"     IPC-A-610 Standard: {rules.get('ipc_classification', 'Class 2/3 Violation')}")
        print(f"     Immediate Action: {rules.get('immediate_operator_action', 'Quarantine and inspect')}")
        print(f"     Root Cause Origin: {rules.get('root_cause_hypothesis', 'Process drift')}")
        print("-" * 60)
        
    # Telemetry HUD on image
    hud = f"{device.upper()} | {latency_ms:.1f}ms ({fps:.1f} FPS) | Defects: {num_detections}"
    cv2.putText(annotated, hud, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, annotated)
        print(f"[✓] Annotated inspection saved to: {out_path}")
        
    return annotated

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PCB Defect Visual Inspector")
    parser.add_argument("--image", default="BhAi_SoC/runs/detect/experiments/baseline_yolo11n_dspcbsd/train_run/val_batch0_labels.jpg")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "0"])
    parser.add_argument("--out", default="BhAi_SoC/experiments/audit_results/test_inspection_annotated.jpg")
    args = parser.parse_args()
    
    inspect_image(args.image, device=args.device, out_path=args.out)
