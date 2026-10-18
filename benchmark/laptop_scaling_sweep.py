"""
laptop_scaling_sweep.py
Runs Batch-Size Scaling & Resolution Pareto Sweep on Laptop GPU (RTX 3050 DirectML)
Matches the exact audit ablations performed on the server.
"""

import os
import time
import json
import numpy as np
import onnxruntime as ort

def run_sweep(onnx_path="BhAi_SoC/experiments/exported_models/best.onnx", runs=20):
    print("=" * 65)
    print(" LAPTOP GPU (RTX 3050) ABLATION SCALING SWEEP")
    print("=" * 65)
    
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(onnx_path, sess_options, providers=["DmlExecutionProvider"])
    input_name = session.get_inputs()[0].name
    
    # [1] Resolution Scaling Sweep
    print("\n[1/2] Running Resolution Scaling Sweep (Latency vs Resolution)...")
    resolutions = [320, 640, 1024]
    res_results = {}
    
    for r in resolutions:
        dummy = np.random.randn(1, 3, r, r).astype(np.float32)
        # Warmup
        try:
            for _ in range(5):
                _ = session.run(None, {input_name: dummy})
            
            latencies = []
            for _ in range(runs):
                t0 = time.perf_counter()
                _ = session.run(None, {input_name: dummy})
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000.0)
                
            mean_lat = float(np.mean(latencies))
            fps = 1000.0 / mean_lat if mean_lat > 0 else 0.0
            res_results[f"{r}x{r}"] = {
                "latency_ms": round(mean_lat, 2),
                "fps": round(fps, 1)
            }
            print(f"  • Resolution {r}x{r:4d} : {mean_lat:6.2f} ms | Throughput: {fps:6.1f} FPS")
        except Exception as e:
            print(f"  • Resolution {r}x{r:4d} : Dynamic shape not supported by static graph.")
            break

    # [2] Batch Size Scaling Sweep
    print("\n[2/2] Running Batch-Size Throughput Scaling Sweep...")
    batches = [1, 2, 4, 8, 16]
    batch_results = {}
    
    for b in batches:
        dummy = np.random.randn(b, 3, 640, 640).astype(np.float32)
        try:
            for _ in range(5):
                _ = session.run(None, {input_name: dummy})
                
            latencies = []
            for _ in range(runs):
                t0 = time.perf_counter()
                _ = session.run(None, {input_name: dummy})
                t1 = time.perf_counter()
                latencies.append(((t1 - t0) * 1000.0) / b)  # per-image latency
                
            mean_lat = float(np.mean(latencies))
            fps = 1000.0 / mean_lat if mean_lat > 0 else 0.0
            batch_results[f"batch_{b}"] = {
                "latency_per_image_ms": round(mean_lat, 2),
                "throughput_fps": round(fps, 1)
            }
            print(f"  • Batch {b:2d} : {mean_lat:6.2f} ms/image | Throughput: {fps:6.1f} FPS")
        except Exception as e:
            print(f"  • Batch {b:2d} : Dynamic batching not supported in exported ONNX graph.")
            break

    sweep_data = {
        "device": "NVIDIA GeForce RTX 3050 6GB Laptop GPU (DirectML)",
        "resolution_sweep": res_results,
        "batch_sweep": batch_results
    }
    
    out_path = "BhAi_SoC/experiments/benchmark_logs/laptop_scaling_sweep.json"
    with open(out_path, "w") as f:
        json.dump(sweep_data, f, indent=4)
        
    print("\n" + "=" * 65)
    print(f"[✓] Scaling sweep results saved to: {out_path}")
    print("=" * 65)

if __name__ == "__main__":
    run_sweep()
