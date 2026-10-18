"""
benchmark_pipeline.py
Automated Cross-Platform Benchmarking Suite
Measures:
1. Statistical Latency (Mean, Std, Median, P95, P99)
2. Throughput (FPS)
3. Memory Footprint (Peak RAM RSS, Peak VRAM)
4. Thermal Delta (Starting Temp, Ending Temp in °C)
5. Power Draw (Watts) & Energy per Inference (Joules)
"""

import os
import sys
import time
import json
import psutil
import argparse
import platform
import subprocess
import numpy as np
import torch
from ultralytics import YOLO

def get_process_memory_mb():
    """Returns current process RAM usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)

def get_gpu_telemetry(device_id):
    """Queries NVIDIA hardware sensors for temperature (°C) and power draw (Watts)."""
    try:
        dev_idx = int(device_id) if str(device_id).isdigit() else 0
        cmd = f"nvidia-smi --id={dev_idx} --query-gpu=temperature.gpu,power.draw --format=csv,noheader,nounits"
        out = subprocess.check_output(cmd, shell=True).decode().strip().split(",")
        temp_c = float(out[0].strip())
        power_w = float(out[1].strip())
        return temp_c, power_w
    except Exception:
        return None, None

def benchmark_vision_model(weights_path, device="cpu", imgsz=640, num_warmup=10, num_runs=100):
    print(f"\n[+] Benchmarking Vision: {weights_path} on {str(device).upper()} ({num_runs} runs)...")
    
    is_cuda = "cuda" in str(device).lower() or str(device).isdigit()
    if is_cuda and not torch.cuda.is_available():
        print("[!] CUDA requested but not available. Falling back to CPU.")
        device = "cpu"
        is_cuda = False
        
    dev_idx = int(device) if str(device).isdigit() else 0
    model = YOLO(weights_path)
    
    # Create dummy frame representing 640x640 industrial PCB crop
    dummy_input = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
    
    # Warmup runs
    print(f"[+] Executing {num_warmup} warm-up iterations...")
    for _ in range(num_warmup):
        _ = model.predict(dummy_input, device=device, verbose=False)
        
    if is_cuda:
        torch.cuda.synchronize(dev_idx)
        torch.cuda.reset_peak_memory_stats(dev_idx)
        
    # Initial hardware telemetry
    start_temp_c, start_power_w = get_gpu_telemetry(device) if is_cuda else (None, None)
    
    latencies = []
    preprocess_times = []
    inference_times = []
    postprocess_times = []
    power_readings = []
    
    start_ram = get_process_memory_mb()
    
    print(f"[+] Recording {num_runs} timed iterations with thermal & power logging...")
    for i in range(num_runs):
        t0 = time.perf_counter()
        results = model.predict(dummy_input, device=device, verbose=False)
        if is_cuda:
            torch.cuda.synchronize(dev_idx)
        t1 = time.perf_counter()
        
        total_time_ms = (t1 - t0) * 1000
        latencies.append(total_time_ms)
        
        speed = results[0].speed
        preprocess_times.append(speed.get("preprocess", 0.0))
        inference_times.append(speed.get("inference", 0.0))
        postprocess_times.append(speed.get("postprocess", 0.0))
        
        # Periodic power sampling
        if is_cuda and i % 25 == 0:
            _, cur_p = get_gpu_telemetry(device)
            if cur_p is not None:
                power_readings.append(cur_p)
                
    end_temp_c, end_power_w = get_gpu_telemetry(device) if is_cuda else (None, None)
    if end_power_w is not None:
        power_readings.append(end_power_w)
        
    lat_arr = np.array(latencies)
    mean_lat_ms = float(np.mean(lat_arr))
    fps = round(1000.0 / mean_lat_ms, 2)
    peak_ram = get_process_memory_mb()
    peak_vram = torch.cuda.max_memory_allocated(dev_idx) / (1024 ** 2) if is_cuda else 0.0
    
    avg_power_w = round(float(np.mean(power_readings)), 2) if power_readings else None
    energy_per_frame_joules = round((avg_power_w * (mean_lat_ms / 1000.0)), 6) if avg_power_w is not None else None
    fps_per_watt = round(fps / avg_power_w, 2) if avg_power_w and avg_power_w > 0 else None
    
    stats = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(dev_idx) if is_cuda else platform.processor(),
        "input_resolution": f"{imgsz}x{imgsz}",
        "iterations": num_runs,
        "throughput_fps": fps,
        "total_latency_ms": {
            "mean": round(mean_lat_ms, 2),
            "std": round(float(np.std(lat_arr)), 2),
            "median": round(float(np.median(lat_arr)), 2),
            "p95": round(float(np.percentile(lat_arr, 95)), 2),
            "p99": round(float(np.percentile(lat_arr, 99)), 2)
        },
        "breakdown_ms": {
            "preprocess_mean": round(float(np.mean(preprocess_times)), 2),
            "inference_mean": round(float(np.mean(inference_times)), 2),
            "postprocess_mean": round(float(np.mean(postprocess_times)), 2)
        },
        "memory": {
            "peak_ram_mb": round(peak_ram, 2),
            "peak_vram_mb": round(peak_vram, 2)
        },
        "thermals_and_energy": {
            "starting_temp_c": start_temp_c,
            "ending_temp_c": end_temp_c,
            "temp_delta_c": round(end_temp_c - start_temp_c, 1) if (start_temp_c is not None and end_temp_c is not None) else None,
            "avg_power_draw_watts": avg_power_w,
            "energy_per_inference_joules": energy_per_frame_joules,
            "silicon_efficiency_fps_per_watt": fps_per_watt
        }
    }
    return stats

def run_benchmark(args):
    os.makedirs(args.out_dir, exist_ok=True)
    
    vision_stats = benchmark_vision_model(
        weights_path=args.weights,
        device=args.device,
        imgsz=args.imgsz,
        num_warmup=args.warmup,
        num_runs=args.runs
    )
    
    target_tag = f"{args.platform}_{args.device}".lower()
    out_file = os.path.join(args.out_dir, f"{target_tag}_benchmark.json")
    
    full_log = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform_tier": args.platform,
        "os": platform.platform(),
        "vision_benchmarks": vision_stats
    }
    
    with open(out_file, "w") as f:
        json.dump(full_log, f, indent=4)
        
    te = vision_stats["thermals_and_energy"]
    print("\n=======================================================")
    print(f" BENCHMARK RESULTS: {args.platform.upper()} ({str(args.device).upper()})")
    print(f" Mean Latency : {vision_stats['total_latency_ms']['mean']} ms (P95: {vision_stats['total_latency_ms']['p95']} ms)")
    print(f" Throughput   : {vision_stats['throughput_fps']} FPS")
    print(f" Peak Memory  : RAM {vision_stats['memory']['peak_ram_mb']} MB | VRAM {vision_stats['memory']['peak_vram_mb']} MB")
    if te["avg_power_draw_watts"] is not None:
        print(f" Thermals     : {te['starting_temp_c']}°C -> {te['ending_temp_c']}°C (Delta: +{te['temp_delta_c']}°C)")
        print(f" Power Draw   : {te['avg_power_draw_watts']} Watts")
        print(f" Energy/Frame : {te['energy_per_inference_joules']} Joules")
        print(f" Efficiency   : {te['silicon_efficiency_fps_per_watt']} FPS / Watt")
    print(f" Log Saved To : {out_file}")
    print("=======================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-platform benchmark harness with thermals & power.")
    parser.add_argument("--weights", default="runs/detect/experiments/baseline_yolo11n_dspcbsd/train_run/weights/best.pt", help="Model weights")
    parser.add_argument("--platform", choices=["server", "laptop", "mobile"], default="server", help="Hardware tier label")
    parser.add_argument("--device", default="1", help="Compute device (e.g. 1, 0, cpu)")
    parser.add_argument("--imgsz", type=int, default=640, help="Resolution")
    parser.add_argument("--runs", type=int, default=100, help="Timed iterations")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--out-dir", default="experiments/benchmark_logs", help="Output directory")
    args = parser.parse_args()
    
    run_benchmark(args)
