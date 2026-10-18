"""
laptop_full_benchmark.py
Comprehensive Local Laptop Benchmark Suite (CPU & GPU)
Evaluates:
  1. Intel Core i5-12450H (CPU) across PyTorch FP32, ONNX Runtime, and LiteRT INT8
  2. NVIDIA GeForce RTX 3050 6GB Laptop GPU (DirectML / CUDA)
Logs:
  - Latency (Mean, Std, Median, P95, P99)
  - Throughput (FPS)
  - Process Memory (RAM RSS in MB)
  - GPU Memory (VRAM allocated in MB)
  - Hardware Telemetry via pynvml: Starting/Ending Temp (°C), Delta (°C), Power Draw (Watts)
  - Energy per inference (Joules) & Silicon Efficiency (FPS / Watt)
"""

import os
import sys
import time
import json
import psutil
import platform
import argparse
import numpy as np

# Try importing pynvml for RTX 3050 hardware telemetry
try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False

def get_ram_usage_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)

def query_gpu_telemetry():
    """Queries NVIDIA RTX 3050 for temperature, power, and VRAM."""
    if not NVML_AVAILABLE:
        return None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        temp_c = float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
        power_w = float(pynvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0  # mW -> W
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        vram_mb = float(mem_info.used) / (1024 ** 2)
        device_name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(device_name, bytes):
            device_name = device_name.decode("utf-8")
        return {
            "device_name": device_name,
            "temp_c": temp_c,
            "power_w": power_w,
            "vram_mb": vram_mb
        }
    except Exception as e:
        return None

def run_pytorch_benchmark(weights_path, device="cpu", num_warmup=10, num_runs=100, imgsz=640):
    from ultralytics import YOLO
    import torch
    
    is_cuda = "cuda" in str(device).lower()
    if is_cuda and not torch.cuda.is_available():
        print(f"  [!] CUDA not available in PyTorch, skipping PyTorch GPU.")
        return None
        
    print(f"\n[+] Benchmarking PyTorch ({weights_path}) on {device.upper()}...")
    model = YOLO(weights_path)
    dummy = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
    
    # Warmup
    for _ in range(num_warmup):
        _ = model.predict(dummy, device=device, verbose=False)
        
    gpu_start = query_gpu_telemetry()
    latencies = []
    power_samples = []
    
    for i in range(num_runs):
        t0 = time.perf_counter()
        _ = model.predict(dummy, device=device, verbose=False)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)
        
        if (i + 1) % 25 == 0:
            tele = query_gpu_telemetry()
            if tele and tele.get("power_w"):
                power_samples.append(tele["power_w"])
                
    gpu_end = query_gpu_telemetry()
    if gpu_end and gpu_end.get("power_w"):
        power_samples.append(gpu_end["power_w"])
        
    latencies = np.array(latencies)
    mean_lat = float(np.mean(latencies))
    fps = float(1000.0 / mean_lat) if mean_lat > 0 else 0.0
    
    avg_power = float(np.mean(power_samples)) if power_samples else (gpu_end["power_w"] if gpu_end else None)
    energy_j = (avg_power * (mean_lat / 1000.0)) if avg_power else None
    fps_per_watt = (fps / avg_power) if (avg_power and avg_power > 0) else None
    
    return {
        "engine": "PyTorch FP32",
        "device": device.upper(),
        "device_name": gpu_end["device_name"] if is_cuda and gpu_end else platform.processor(),
        "iterations": num_runs,
        "latency_ms": {
            "mean": round(mean_lat, 2),
            "std": round(float(np.std(latencies)), 2),
            "median": round(float(np.median(latencies)), 2),
            "p95": round(float(np.percentile(latencies, 95)), 2),
            "p99": round(float(np.percentile(latencies, 99)), 2),
        },
        "throughput_fps": round(fps, 2),
        "peak_ram_mb": round(get_ram_usage_mb(), 2),
        "peak_vram_mb": round(gpu_end["vram_mb"], 2) if is_cuda and gpu_end else 0.0,
        "thermal_delta_c": round(gpu_end["temp_c"] - gpu_start["temp_c"], 1) if (is_cuda and gpu_start and gpu_end) else None,
        "avg_power_watts": round(avg_power, 2) if avg_power else None,
        "energy_per_inference_joules": round(energy_j, 4) if energy_j else None,
        "silicon_efficiency_fps_per_watt": round(fps_per_watt, 2) if fps_per_watt else None
    }

def run_onnx_benchmark(onnx_path, provider="CPUExecutionProvider", num_warmup=10, num_runs=100, imgsz=640):
    import onnxruntime as ort
    
    available = ort.get_available_providers()
    if provider not in available:
        print(f"  [!] Provider {provider} not available in ONNX Runtime. Available: {available}")
        return None
        
    device_label = "RTX 3050 (DirectML)" if "Dml" in provider else "CPU"
    print(f"\n[+] Benchmarking ONNX Runtime ({onnx_path}) on {device_label} ({provider})...")
    
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(onnx_path, sess_options, providers=[provider])
    
    input_name = session.get_inputs()[0].name
    dummy = np.random.randn(1, 3, imgsz, imgsz).astype(np.float32)
    
    # Warmup
    for _ in range(num_warmup):
        _ = session.run(None, {input_name: dummy})
        
    gpu_start = query_gpu_telemetry()
    latencies = []
    power_samples = []
    
    for i in range(num_runs):
        t0 = time.perf_counter()
        _ = session.run(None, {input_name: dummy})
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)
        
        if (i + 1) % 25 == 0:
            tele = query_gpu_telemetry()
            if tele and tele.get("power_w"):
                power_samples.append(tele["power_w"])
                
    gpu_end = query_gpu_telemetry()
    if gpu_end and gpu_end.get("power_w"):
        power_samples.append(gpu_end["power_w"])
        
    latencies = np.array(latencies)
    mean_lat = float(np.mean(latencies))
    fps = float(1000.0 / mean_lat) if mean_lat > 0 else 0.0
    
    is_gpu = "Dml" in provider or "CUDA" in provider
    avg_power = float(np.mean(power_samples)) if power_samples else (gpu_end["power_w"] if gpu_end else None)
    energy_j = (avg_power * (mean_lat / 1000.0)) if avg_power else None
    fps_per_watt = (fps / avg_power) if (avg_power and avg_power > 0) else None
    
    return {
        "engine": f"ONNX Runtime ({'DirectML' if 'Dml' in provider else 'CPU'})",
        "device": device_label,
        "device_name": gpu_end["device_name"] if is_gpu and gpu_end else platform.processor(),
        "iterations": num_runs,
        "latency_ms": {
            "mean": round(mean_lat, 2),
            "std": round(float(np.std(latencies)), 2),
            "median": round(float(np.median(latencies)), 2),
            "p95": round(float(np.percentile(latencies, 95)), 2),
            "p99": round(float(np.percentile(latencies, 99)), 2),
        },
        "throughput_fps": round(fps, 2),
        "peak_ram_mb": round(get_ram_usage_mb(), 2),
        "peak_vram_mb": round(gpu_end["vram_mb"], 2) if is_gpu and gpu_end else 0.0,
        "thermal_delta_c": round(gpu_end["temp_c"] - gpu_start["temp_c"], 1) if (is_gpu and gpu_start and gpu_end) else None,
        "avg_power_watts": round(avg_power, 2) if avg_power else None,
        "energy_per_inference_joules": round(energy_j, 4) if energy_j else None,
        "silicon_efficiency_fps_per_watt": round(fps_per_watt, 2) if fps_per_watt else None
    }

def run_tflite_benchmark(tflite_path, num_warmup=10, num_runs=100, imgsz=640):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except Exception:
        try:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter
        except Exception:
            try:
                import tflite_runtime.interpreter as tflite
                Interpreter = tflite.Interpreter
            except Exception:
                print("  [!] TensorFlow / LiteRT runtime not installed locally for TFLite execution.")
                return None
                
    print(f"\n[+] Benchmarking TFLite ({tflite_path}) on CPU (XNNPACK)...")
    interpreter = Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    in_shape = input_details[0]['shape']
    in_dtype = input_details[0]['dtype']
    if in_dtype == np.uint8 or in_dtype == np.int8:
        dummy = np.random.randint(0, 127, in_shape, dtype=in_dtype)
    else:
        dummy = np.random.randn(*in_shape).astype(in_dtype)
        
    for _ in range(num_warmup):
        interpreter.set_tensor(input_details[0]['index'], dummy)
        interpreter.invoke()
        
    latencies = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        interpreter.set_tensor(input_details[0]['index'], dummy)
        interpreter.invoke()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)
        
    latencies = np.array(latencies)
    mean_lat = float(np.mean(latencies))
    fps = float(1000.0 / mean_lat) if mean_lat > 0 else 0.0
    
    return {
        "engine": "LiteRT INT8 (XNNPACK)",
        "device": "CPU",
        "device_name": platform.processor(),
        "iterations": num_runs,
        "latency_ms": {
            "mean": round(mean_lat, 2),
            "std": round(float(np.std(latencies)), 2),
            "median": round(float(np.median(latencies)), 2),
            "p95": round(float(np.percentile(latencies, 95)), 2),
            "p99": round(float(np.percentile(latencies, 99)), 2),
        },
        "throughput_fps": round(fps, 2),
        "peak_ram_mb": round(get_ram_usage_mb(), 2),
        "peak_vram_mb": 0.0,
        "thermal_delta_c": None,
        "avg_power_watts": None,
        "energy_per_inference_joules": None,
        "silicon_efficiency_fps_per_watt": None
    }

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Local Laptop Benchmark")
    parser.add_argument("--weights_dir", default="BhAi_SoC/runs/detect/experiments/baseline_yolo11n_dspcbsd/train_run/weights")
    parser.add_argument("--runs", type=int, default=100)
    args = parser.parse_args()
    
    pt_path = os.path.join(args.weights_dir, "best.pt")
    onnx_path = os.path.join(args.weights_dir, "best.onnx")
    int8_path = os.path.join(args.weights_dir, "best_int8.tflite")
    
    print("=" * 70)
    print(" BHARAT AI-SOC CHALLENGE: LOCAL LAPTOP BENCHMARK SUITE")
    print(f" Platform: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f" CPU: {platform.processor()} | Available Cores: {psutil.cpu_count(logical=True)}")
    if NVML_AVAILABLE:
        gpu_tele = query_gpu_telemetry()
        if gpu_tele:
            print(f" GPU: {gpu_tele['device_name']} (Temp: {gpu_tele['temp_c']}°C, Power: {gpu_tele['power_w']}W)")
    print("=" * 70)
    
    results = {}
    
    # 1. PyTorch CPU
    if os.path.exists(pt_path):
        res = run_pytorch_benchmark(pt_path, device="cpu", num_runs=args.runs)
        if res:
            results["pytorch_cpu"] = res
            
    # 2. ONNX Runtime CPU
    if os.path.exists(onnx_path):
        res = run_onnx_benchmark(onnx_path, provider="CPUExecutionProvider", num_runs=args.runs)
        if res:
            results["onnx_cpu"] = res
            
    # 3. ONNX Runtime DirectML (RTX 3050 GPU)
    if os.path.exists(onnx_path):
        res = run_onnx_benchmark(onnx_path, provider="DmlExecutionProvider", num_runs=args.runs)
        if res:
            results["onnx_directml_gpu"] = res
            
    # 4. PyTorch CUDA (if CUDA enabled)
    if os.path.exists(pt_path):
        res = run_pytorch_benchmark(pt_path, device="cuda:0", num_runs=args.runs)
        if res:
            results["pytorch_cuda_gpu"] = res
            
    # 5. LiteRT INT8 CPU
    if os.path.exists(int8_path):
        res = run_tflite_benchmark(int8_path, num_runs=args.runs)
        if res:
            results["litert_int8_cpu"] = res
            
    # Output file paths
    out_dir = "BhAi_SoC/experiments/benchmark_logs"
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "laptop_comprehensive_benchmark.json")
    md_path = os.path.join(out_dir, "laptop_benchmark_report.md")
    
    # Save JSON
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
        
    # Generate Markdown Table
    lines = [
        "# Local Laptop Cross-Engine Benchmark Report\n",
        f"**Timestamp**: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        f"**Host Platform**: {platform.system()} ({platform.processor()})"
    ]
    if NVML_AVAILABLE and query_gpu_telemetry():
        lines.append(f"**Discrete GPU**: {query_gpu_telemetry()['device_name']}\n")
    else:
        lines.append("")
        
    lines.append("| Runtime Engine | Target Hardware | Mean Latency (ms) | P95 Tail (ms) | Throughput (FPS) | Peak RAM (MB) | Peak VRAM (MB) | GPU Power (W) | Efficiency (FPS/W) |")
    lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    
    for k, v in results.items():
        mean_l = f"{v['latency_ms']['mean']:.2f}"
        p95_l = f"{v['latency_ms']['p95']:.2f}"
        fps_val = f"{v['throughput_fps']:.2f}"
        ram_val = f"{v['peak_ram_mb']:.1f}"
        vram_val = f"{v['peak_vram_mb']:.1f}" if v['peak_vram_mb'] > 0 else "-"
        pwr_val = f"{v['avg_power_watts']:.2f} W" if v['avg_power_watts'] else "-"
        eff_val = f"{v['silicon_efficiency_fps_per_watt']:.2f}" if v['silicon_efficiency_fps_per_watt'] else "-"
        lines.append(f"| {v['engine']} | {v['device']} | **{mean_l}** | {p95_l} | **{fps_val}** | {ram_val} | {vram_val} | {pwr_val} | {eff_val} |")
        
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
        
    print("\n" + "=" * 70)
    print(" BENCHMARK COMPLETED SUCCESSFULLY!")
    print(f" JSON results saved to: {json_path}")
    print(f" Markdown report saved to: {md_path}")
    print("=" * 70)
    
    if NVML_AVAILABLE:
        pynvml.nvmlShutdown()

if __name__ == "__main__":
    main()
