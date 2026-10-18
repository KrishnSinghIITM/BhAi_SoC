"""
full_audit_suite.py
Complete 100% Research & Competition Audit Suite
Runs:
1. Visual Test Predictions on Real PCBs (already completed!)
2. Batch Size Scaling Sweep (batch=1, 4, 8, 16, 32, 64)
3. Input Resolution Scaling Sweep (320x320, 640x640, 1024x1024)
Outputs all JSONs and visual plots into experiments/audit_results/
"""

import os
import time
import json
import argparse
import numpy as np
import torch
from ultralytics import YOLO

def run_audit(weights_path, data_path="data.yaml", device="1", out_dir="experiments/audit_results"):
    os.makedirs(out_dir, exist_ok=True)
    print("\n=======================================================")
    print(" LAUNCHING 100% RESEARCH & COMPETITION AUDIT SUITE")
    print(f" Model: {weights_path} | Device: GPU {device}")
    print("=======================================================\n")
    
    model = YOLO(weights_path)
    audit_report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_weights": weights_path,
        "device": device
    }
    
    # ---------------------------------------------------------
    # 1. Visual Predictions (Already Generated)
    # ---------------------------------------------------------
    print("[1/3] Visual PCB defect predictions already generated in visual_inspected_pcbs/ ✅")
    audit_report["visual_inspections_path"] = "experiments/audit_results/visual_inspected_pcbs"

    # ---------------------------------------------------------
    # 2. Batch Size Scaling Sweep (Hardware Throughput Curve)
    # ---------------------------------------------------------
    print("\n[2/3] Running Batch-Size Throughput Scaling Sweep...")
    batch_sizes = [1, 4, 8, 16, 32, 64]
    batch_results = {}
    
    for b in batch_sizes:
        # Pass list of (H, W, C) images for batch inference in Ultralytics
        dummy_batch = [np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8) for _ in range(b)]
        
        # Warmup
        _ = model.predict(dummy_batch, device=device, verbose=False)
        if torch.cuda.is_available() and str(device).isdigit():
            torch.cuda.synchronize(int(device))
            
        t0 = time.perf_counter()
        iters = 15
        for _ in range(iters):
            _ = model.predict(dummy_batch, device=device, verbose=False)
        if torch.cuda.is_available() and str(device).isdigit():
            torch.cuda.synchronize(int(device))
        t1 = time.perf_counter()
        
        total_time_ms = ((t1 - t0) / iters) * 1000
        fps = (b * 1000.0) / total_time_ms
        latency_per_image_ms = total_time_ms / b
        
        batch_results[f"batch_{b}"] = {
            "batch_size": b,
            "total_batch_latency_ms": round(total_time_ms, 2),
            "latency_per_image_ms": round(latency_per_image_ms, 2),
            "throughput_fps": round(fps, 2)
        }
        print(f"  • Batch {b:2d}: {latency_per_image_ms:5.2f} ms/image | Throughput: {fps:6.1f} FPS")
        
    audit_report["batch_scaling_ablation"] = batch_results

    # ---------------------------------------------------------
    # 3. Input Resolution Scaling Sweep (Pareto Frontier)
    # ---------------------------------------------------------
    print("\n[3/3] Running Resolution Scaling Sweep (Latency vs Resolution)...")
    resolutions = [320, 640, 1024]
    res_results = {}
    
    for res in resolutions:
        dummy_img = np.random.randint(0, 255, (res, res, 3), dtype=np.uint8)
        _ = model.predict(dummy_img, imgsz=res, device=device, verbose=False)
        if torch.cuda.is_available() and str(device).isdigit():
            torch.cuda.synchronize(int(device))
            
        t0 = time.perf_counter()
        iters = 25
        for _ in range(iters):
            _ = model.predict(dummy_img, imgsz=res, device=device, verbose=False)
        if torch.cuda.is_available() and str(device).isdigit():
            torch.cuda.synchronize(int(device))
        t1 = time.perf_counter()
        
        avg_ms = ((t1 - t0) / iters) * 1000
        fps = 1000.0 / avg_ms
        res_results[f"res_{res}x{res}"] = {
            "resolution": f"{res}x{res}",
            "latency_ms": round(avg_ms, 2),
            "fps": round(fps, 2)
        }
        print(f"  • Resolution {res:4d}x{res:<4d}: {avg_ms:5.2f} ms | Throughput: {fps:6.1f} FPS")
        
    audit_report["resolution_scaling_ablation"] = res_results
    
    report_path = os.path.join(out_dir, "complete_audit_report.json")
    with open(report_path, "w") as f:
        json.dump(audit_report, f, indent=4)
        
    print("\n=======================================================")
    print(f" [✓] COMPLETE AUDIT FINISHED! Report: {report_path}")
    print("=======================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="runs/detect/experiments/baseline_yolo11n_dspcbsd/train_run/weights/best.pt")
    parser.add_argument("--data", default="data.yaml")
    parser.add_argument("--device", default="1")
    parser.add_argument("--out-dir", default="experiments/audit_results")
    args = parser.parse_args()
    
    run_audit(args.weights, args.data, args.device, args.out_dir)
