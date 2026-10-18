#!/usr/bin/env python3
"""
qualcomm_cloud_benchmark.py
===========================
End-to-End Qualcomm AI Hub Cloud Profiling & Benchmarking Suite.
Measures exhaustive inference metrics across target mobile devices:
  - Latency (Median, Mean, Min, Max, P90, P95, P99, StdDev)
  - Throughput (FPS)
  - Memory Footprint (Peak RAM, Allocation Range)
  - Compute Unit Allocation (NPU vs GPU vs CPU layer percentage)
  - Layer-by-Layer Operator Profiling & Execution Cycles
  - First-Load and Warm-Load Times
Saves all telemetry as structured JSON and Markdown comparison reports.
"""

import os
import sys
import time
import json
import numpy as np
from datetime import datetime
from collections import Counter

# Fix Windows console UTF-8 encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

import qai_hub as hub

WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(WORKSPACE_DIR, "experiments", "exported_models")
OUTPUT_DIR = os.path.join(WORKSPACE_DIR, "experiments", "benchmark_logs", "qualcomm_ai_hub")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Exact Qualcomm AI Hub Benchmark Matrix mapped to submitted jobs
BENCHMARK_MATRIX = [
    {
        "model_key": "best_int8",
        "model_path": os.path.join(MODELS_DIR, "best_int8.tflite"),
        "model_format": "LiteRT INT8",
        "precision": "INT8 (PTQ)",
        "device_name": "Samsung Galaxy S24",
        "chipset": "Snapdragon 8 Gen 3 (SM8650)",
        "job_id": "jpv281qmg"
    },
    {
        "model_key": "best_fp32",
        "model_path": os.path.join(MODELS_DIR, "best.tflite"),
        "model_format": "LiteRT FP32",
        "precision": "FP32",
        "device_name": "Samsung Galaxy S24",
        "chipset": "Snapdragon 8 Gen 3 (SM8650)",
        "job_id": "jp1o9d2l5"
    },
    {
        "model_key": "best_int8",
        "model_path": os.path.join(MODELS_DIR, "best_int8.tflite"),
        "model_format": "LiteRT INT8",
        "precision": "INT8 (PTQ)",
        "device_name": "Samsung Galaxy S23",
        "chipset": "Snapdragon 8 Gen 2 (SM8550)",
        "job_id": "jgd6krnlp"
    },
    {
        "model_key": "best_fp32",
        "model_path": os.path.join(MODELS_DIR, "best.tflite"),
        "model_format": "LiteRT FP32",
        "precision": "FP32",
        "device_name": "Samsung Galaxy S23",
        "chipset": "Snapdragon 8 Gen 2 (SM8550)",
        "job_id": "j5wykd26g"
    },
    {
        "model_key": "best_int8",
        "model_path": os.path.join(MODELS_DIR, "best_int8.tflite"),
        "model_format": "LiteRT INT8",
        "precision": "INT8 (PTQ)",
        "device_name": "Samsung Galaxy S22 5G",
        "chipset": "Snapdragon 8 Gen 1 (SM8450)",
        "job_id": "jg9or3jlg"
    },
    {
        "model_key": "best_int8",
        "model_path": os.path.join(MODELS_DIR, "best_int8.tflite"),
        "model_format": "LiteRT INT8",
        "precision": "INT8 (PTQ)",
        "device_name": "Samsung Galaxy A73 5G",
        "chipset": "Snapdragon 778G (SM7325)",
        "job_id": "j5m97wqwg"
    }
]

def load_jobs():
    jobs = []
    for cfg in BENCHMARK_MATRIX:
        jid = cfg["job_id"]
        print(f"Loading job {jid} for {cfg['model_format']} on {cfg['device_name']}...")
        job = hub.get_job(jid)
        jobs.append({"cfg": cfg, "job": job})
    return jobs

def wait_for_all_jobs(jobs):
    print("\n--- Checking Qualcomm AI Hub Cloud Job Statuses ---")
    completed = set()
    total = len(jobs)
    
    while len(completed) < total:
        for idx, item in enumerate(jobs):
            if idx in completed:
                continue
            job = item["job"]
            status = job.get_status()
            code = status.code
            cfg = item["cfg"]
            name = f"{cfg['model_format']} @ {cfg['device_name']}"
            
            if code == "SUCCESS":
                print(f"[COMPLETED] {name} (Job: {job.job_id})")
                completed.add(idx)
            elif code in ["FAILED", "CANCELLED"]:
                print(f"[ERROR] {name} (Job: {job.job_id}) -> {code}: {status.message}")
                completed.add(idx)
            else:
                print(f"[RUNNING] {name} (Job: {job.job_id}) -> {code}")
        
        if len(completed) < total:
            time.sleep(10)

def analyze_and_extract_metrics(jobs):
    print("\n--- Extracting Detailed Telemetry & Downloading Artifacts ---")
    results = []
    
    for item in jobs:
        cfg = item["cfg"]
        job = item["job"]
        status = job.get_status()
        
        if status.code != "SUCCESS":
            print(f"Skipping incomplete/failed job {job.job_id} for {cfg['device_name']} ({status.code})")
            continue
        
        job_subfolder = os.path.join(OUTPUT_DIR, f"{cfg['model_key']}_{cfg['device_name'].replace(' ', '_')}")
        os.makedirs(job_subfolder, exist_ok=True)
        
        # Download raw profile JSON
        profile_json_path = os.path.join(job_subfolder, "profile.json")
        try:
            profile = job.download_profile(filename=profile_json_path)
            if isinstance(profile, str):
                with open(profile_json_path, "r", encoding="utf-8") as f:
                    profile = json.load(f)
        except Exception as e:
            print(f"Warning downloading profile JSON: {e}")
            profile = job.download_profile()
            with open(profile_json_path, "w", encoding="utf-8") as f:
                json.dump(profile, f, indent=2)

        # Download raw device execution logs
        try:
            job.download_job_logs(output_dir=job_subfolder)
        except Exception as e:
            print(f"Warning downloading job logs: {e}")

        # Compute exhaustive statistical metrics
        exec_summary = profile.get("execution_summary", {})
        exec_detail = profile.get("execution_detail", [])
        
        all_times_us = exec_summary.get("all_inference_times", [])
        if all_times_us:
            all_times_ms = np.array(all_times_us) / 1000.0
            p50 = float(np.median(all_times_ms))
            mean = float(np.mean(all_times_ms))
            p90 = float(np.percentile(all_times_ms, 90))
            p95 = float(np.percentile(all_times_ms, 95))
            p99 = float(np.percentile(all_times_ms, 99))
            min_lat = float(np.min(all_times_ms))
            max_lat = float(np.max(all_times_ms))
            std_lat = float(np.std(all_times_ms))
            throughput_fps = float(1000.0 / mean) if mean > 0 else 0.0
        else:
            p50 = mean = p90 = p95 = p99 = min_lat = max_lat = std_lat = throughput_fps = 0.0
        
        # Memory metrics
        peak_memory_mb = float(exec_summary.get("estimated_inference_peak_memory", 0)) / (1024 * 1024)
        first_load_ms = float(exec_summary.get("first_load_time", 0)) / 1000.0
        warm_load_ms = float(exec_summary.get("warm_load_time", 0)) / 1000.0
        
        # Hardware delegation breakdown
        unit_counts = Counter(x.get("compute_unit", "UNKNOWN") for x in exec_detail)
        total_layers = len(exec_detail)
        npu_ops = unit_counts.get("NPU", 0)
        gpu_ops = unit_counts.get("GPU", 0)
        cpu_ops = unit_counts.get("CPU", 0)
        npu_delegation_pct = float(npu_ops / total_layers * 100.0) if total_layers > 0 else 0.0
        
        # Operator breakdown
        op_types = Counter(x.get("type", "UNKNOWN") for x in exec_detail)
        
        # Top 5 most intensive layers by cycles / time
        top_layers = sorted(exec_detail, key=lambda x: x.get("execution_time", 0), reverse=True)[:5]
        
        entry = {
            "device": cfg["device_name"],
            "chipset": cfg["chipset"],
            "model_format": cfg["model_format"],
            "precision": cfg["precision"],
            "job_id": job.job_id,
            "job_url": job.url,
            "latency_ms": {
                "median_p50": round(p50, 3),
                "mean": round(mean, 3),
                "min": round(min_lat, 3),
                "max": round(max_lat, 3),
                "p90": round(p90, 3),
                "p95": round(p95, 3),
                "p99": round(p99, 3),
                "std_dev": round(std_lat, 3)
            },
            "throughput_fps": round(throughput_fps, 1),
            "iterations_benchmarked": len(all_times_us),
            "peak_memory_mb": round(peak_memory_mb, 2),
            "first_load_ms": round(first_load_ms, 2),
            "warm_load_ms": round(warm_load_ms, 2),
            "delegation": {
                "total_layers": total_layers,
                "npu_ops": npu_ops,
                "gpu_ops": gpu_ops,
                "cpu_ops": cpu_ops,
                "npu_delegation_pct": round(npu_delegation_pct, 1)
            },
            "top_operator_types": dict(op_types.most_common(8)),
            "top_latency_layers": [
                {
                    "name": l.get("name"),
                    "type": l.get("type"),
                    "compute_unit": l.get("compute_unit"),
                    "execution_time_us": l.get("execution_time"),
                    "cycles": l.get("execution_cycles")
                }
                for l in top_layers
            ]
        }
        results.append(entry)
        print(f"  --> {cfg['device_name']} [{cfg['precision']}]: {p50:.2f} ms | {throughput_fps:.1f} FPS | NPU: {npu_delegation_pct:.1f}%")
    
    # Save structured results JSON
    master_json_path = os.path.join(WORKSPACE_DIR, "experiments", "benchmark_logs", "qualcomm_ai_hub_results.json")
    with open(master_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "account": "24f3100088@es.study.iitm.ac.in",
            "benchmark_results": results
        }, f, indent=2)
    print(f"\n[OK] Saved master JSON results to: {master_json_path}")
    
    # Generate Markdown Report
    generate_markdown_report(results)
    return results

def generate_markdown_report(results):
    report_path = os.path.join(WORKSPACE_DIR, "experiments", "benchmark_logs", "qualcomm_ai_hub_report.md")
    
    md = []
    md.append("# Qualcomm AI Hub On-Device Cloud Profiling Report")
    md.append(f"**Generated**: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}  ")
    md.append("**Platform**: Qualcomm AI Hub Cloud Device Farm (Real Physical Silicon)  ")
    md.append("**Application**: Bharat AI-SoC Challenge (PS5: Edge Defect Detection & Quality Control)  \n")
    
    md.append("## 1. Cross-Device & Precision Comparison Table\n")
    md.append("| Target Device | Silicon / SoC | Precision | Median Latency | Mean Latency | Throughput | Peak RAM | NPU Delegation | Job URL |")
    md.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |")
    
    for r in results:
        lat = r["latency_ms"]
        delg = r["delegation"]
        md.append(
            f"| **{r['device']}** | {r['chipset']} | **{r['precision']}** | "
            f"**{lat['median_p50']:.2f} ms** | {lat['mean']:.2f} ms | "
            f"**{r['throughput_fps']:.1f} FPS** | {r['peak_memory_mb']:.1f} MB | "
            f"**{delg['npu_delegation_pct']:.1f}% ({delg['npu_ops']}/{delg['total_layers']} ops)** | "
            f"[Dashboard]({r['job_url']}) |"
        )
    
    md.append("\n---\n")
    md.append("## 2. Statistical Latency Distribution (P50, P90, P95, P99, Jitter)\n")
    md.append("| Device | Precision | Min (ms) | P50 / Median | P90 (ms) | P95 (ms) | P99 (ms) | Max (ms) | StdDev (ms) |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    
    for r in results:
        l = r["latency_ms"]
        md.append(
            f"| {r['device']} | {r['precision']} | {l['min']:.2f} | **{l['median_p50']:.2f}** | "
            f"{l['p90']:.2f} | {l['p95']:.2f} | {l['p99']:.2f} | {l['max']:.2f} | ±{l['std_dev']:.2f} |"
        )
    
    md.append("\n---\n")
    md.append("## 3. Hardware Compute Unit Allocation & Layer-by-Layer Profiling\n")
    
    for r in results:
        d = r["delegation"]
        md.append(f"### {r['device']} — {r['precision']} ({r['chipset']})")
        md.append(f"- **Total Layers**: {d['total_layers']}")
        md.append(f"- **Hexagon NPU Ops**: `{d['npu_ops']}` ({d['npu_delegation_pct']}%)")
        md.append(f"- **GPU Fallback Ops**: `{d['gpu_ops']}`")
        md.append(f"- **CPU Fallback Ops**: `{d['cpu_ops']}`")
        md.append(f"- **First Load Time**: `{r['first_load_ms']:.1f} ms` | **Warm Load Time**: `{r['warm_load_ms']:.1f} ms`")
        md.append(f"- **Operator Breakdown**: " + ", ".join([f"`{k}`: {v}" for k, v in r["top_operator_types"].items()]))
        
        md.append("\n**Top 5 Latency-Critical Layers:**\n")
        md.append("| Layer Name | Op Type | Compute Unit | Execution Time ($\mu s$) | Cycles |")
        md.append("| :--- | :---: | :---: | :---: | :---: |")
        for lyr in r["top_latency_layers"]:
            md.append(f"| `{lyr['name']}` | `{lyr['type']}` | **{lyr['compute_unit']}** | {lyr['execution_time_us']} $\mu s$ | {lyr['cycles']} |")
        md.append("\n")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"[OK] Generated comprehensive Markdown report: {report_path}")

def main():
    print("=" * 70)
    print("Qualcomm AI Hub Cloud Benchmark Suite — Bharat AI-SoC PS5")
    print("=" * 70)
    jobs = load_jobs()
    wait_for_all_jobs(jobs)
    analyze_and_extract_metrics(jobs)
    print("\n[SUCCESS] All Qualcomm AI Hub benchmarks completed and logged.")

if __name__ == "__main__":
    main()
