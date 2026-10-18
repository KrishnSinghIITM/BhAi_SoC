"""
train_vision.py
Automated Training Pipeline for YOLO11-Nano on DsPCBSD+
Optimized for NVIDIA RTX 6000 Ada / Blackwell Server execution.
Outputs all evaluation curves, confusion matrices, and audit-proof logs.
"""

import os
import sys
import json
import time
import platform
import argparse
from pathlib import Path
import torch
from ultralytics import YOLO

def log_environment(output_dir, device_id):
    """Logs complete server hardware & software environment manifest for research audit."""
    os.makedirs(output_dir, exist_ok=True)
    dev_int = int(device_id) if str(device_id).isdigit() else 0
    env_info = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "os": platform.platform(),
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_name": torch.cuda.get_device_name(dev_int) if torch.cuda.is_available() else "CPU",
        "gpu_vram_gb": round(torch.cuda.get_device_properties(dev_int).total_memory / (1024**3), 2) if torch.cuda.is_available() else 0,
    }
    env_path = os.path.join(output_dir, "env_metadata.json")
    with open(env_path, "w") as f:
        json.dump(env_info, f, indent=4)
    print(f"[✓] Hardware & Software environment manifest logged: {env_path}")
    return env_info

def train_pcb_vision(args):
    exp_dir = os.path.join("experiments", args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # 1. Log environment
    log_environment(exp_dir, args.device)
    
    # 2. Initialize Model (YOLO11-Nano or YOLO11-Small)
    print(f"[+] Initializing architecture: {args.model}")
    model = YOLO(args.model)
    
    # 3. Train with research-grade logging & mobile-specific augmentations
    print(f"[+] Starting training for {args.epochs} epochs on device {args.device}...")
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        optimizer=args.optimizer,
        lr0=args.lr0,
        seed=42,
        deterministic=True,
        amp=True,
        cache=args.cache,
        
        # Mobile handheld / rig camera augmentations
        degrees=15.0,
        translate=0.1,
        scale=0.2,
        fliplr=0.5,
        flipud=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        mosaic=1.0,
        mixup=0.1,
        
        project=exp_dir,
        name="train_run",
        save=True,
        plots=True,
        val=True
    )
    
    # 4. Comprehensive Validation on Validation/Test Set
    print("[+] Running detailed validation evaluation...")
    metrics = model.val(data=args.data, imgsz=args.imgsz, split="val")
    
    # 5. Extract and format per-class metrics and model complexity
    class_names = model.names
    per_class_results = {}
    for i, name in class_names.items():
        if i < len(metrics.box.maps):
            per_class_results[name] = {
                "mAP50": round(float(metrics.box.ap50[i]), 4),
                "mAP50_95": round(float(metrics.box.maps[i]), 4)
            }
            
    summary = {
        "model_architecture": args.model,
        "input_resolution": f"{args.imgsz}x{args.imgsz}",
        "total_parameters_million": round(sum(p.numel() for p in model.model.parameters()) / 1e6, 2),
        "overall_mAP50": round(float(metrics.box.map50), 4),
        "overall_mAP50_95": round(float(metrics.box.map), 4),
        "overall_precision": round(float(metrics.box.mp), 4),
        "overall_recall": round(float(metrics.box.mr), 4),
        "speed_ms": {
            "preprocess": round(float(metrics.speed["preprocess"]), 2),
            "inference": round(float(metrics.speed["inference"]), 2),
            "postprocess": round(float(metrics.speed["postprocess"]), 2)
        },
        "per_class_metrics": per_class_results
    }
    
    summary_path = os.path.join(exp_dir, "metrics_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
        
    print(f"\n==========================================")
    print(f" TRAINING COMPLETE - EXPERIMENT SAVED")
    print(f" Best Checkpoint: {os.path.join(exp_dir, 'train_run', 'weights', 'best.pt')}")
    print(f" Overall mAP@0.5: {summary['overall_mAP50']}")
    print(f" Overall mAP@0.5:0.95: {summary['overall_mAP50_95']}")
    print(f" Summary JSON: {summary_path}")
    print(f"==========================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO11 on DsPCBSD+ for edge deployment.")
    parser.add_argument("--model", default="yolo11n.pt", help="Base model weights (e.g. yolo11n.pt, yolo11s.pt)")
    parser.add_argument("--data", default="data.yaml", help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=64, help="Batch size (e.g. 64 on RTX 6000 Ada)")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image resolution")
    parser.add_argument("--device", default="1", help="CUDA device index or 'cpu'")
    parser.add_argument("--workers", type=int, default=8, help="DataLoader CPU workers")
    parser.add_argument("--optimizer", default="AdamW", help="Optimizer")
    parser.add_argument("--lr0", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--cache", default="ram", help="Dataset caching strategy ('ram', 'disk', False)")
    parser.add_argument("--exp-name", default="baseline_yolo11n_dspcbsd", help="Experiment name")
    args = parser.parse_args()
    
    train_pcb_vision(args)
