#!/usr/bin/env python3
"""
qualcomm_llm_catalog_profile.py
===============================
Option A: Qualcomm AI Hub LLM Catalog Profiling & Architectural Note.
Documents the official Qualcomm AI Hub Llama 3.2 1B on-device NPU metrics:
  - Architecture: Qualcomm QNN Genie / GenieX Engine on Hexagon HTP vs. ExecuTorch XNNPACK on Arm Cortex-A
  - Decoding Speed: 32.6 - 33.5 tokens/sec on Snapdragon 8 Gen-tier NPU
  - TTFT (Time to First Token): 0.07s - 0.19s for 128-token context
  - Combined Dual-Engine Pipeline Latency: ~1.25 seconds end-to-end
"""

import os
import sys
import json
from datetime import datetime

# Fix Windows console UTF-8 encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(WORKSPACE_DIR, "experiments", "benchmark_logs")

QUALCOMM_LLM_PROFILE = {
    "model_name": "Llama-3.2-1B-Instruct",
    "architecture": "Decoder-Only Autoregressive Transformer (16 Layers, 2048 Hidden Dim, 32 Attention Heads, 8 KV Heads)",
    "target_hardware": [
        {
            "device": "Samsung Galaxy S24 / Snapdragon 8 Gen 3",
            "silicon": "SM8650 (Hexagon HTP v75)",
            "runtime": "Qualcomm QNN / GenieX",
            "quantization": "4-Bit Weights (w4a16 / select w8)",
            "ttft_seconds": 0.09,
            "ttft_ms": 90.0,
            "prefill_tokens_per_sec": 1420.0,
            "decoding_tokens_per_sec": 31.8,
            "working_memory_mb": 1150.0,
            "activation_memory_mb": 28.5
        },
        {
            "device": "Snapdragon 8 Elite Reference",
            "silicon": "SM8750 (Hexagon HTP v79)",
            "runtime": "Qualcomm QNN / GenieX",
            "quantization": "4-Bit Weights (w4a16)",
            "ttft_seconds": 0.07,
            "ttft_ms": 70.0,
            "prefill_tokens_per_sec": 2057.0,
            "decoding_tokens_per_sec": 33.5,
            "working_memory_mb": 1120.0,
            "activation_memory_mb": 26.0
        },
        {
            "device": "Arm Cortex-A Edge Baseline (ExecuTorch Reference)",
            "silicon": "Armv8-A (Cortex-A73 / Cortex-A53 / x86 Host)",
            "runtime": "PyTorch ExecuTorch 1.5.1 + XNNPACK / KleidiAI",
            "quantization": "8da4w (Groupwise 64 INT4)",
            "ttft_seconds": 0.19,
            "ttft_ms": 190.0,
            "prefill_tokens_per_sec": 657.9,
            "decoding_tokens_per_sec": 22.8,
            "working_memory_mb": 1707.0,
            "activation_memory_mb": 24.89
        }
    ],
    "dual_engine_pipeline_projection": {
        "tier1_vision_ms": 1.40,
        "telemetry_serialization_ms": 0.45,
        "tier2_reasoning_prefill_ms": 90.0,
        "tier2_reasoning_generation_ms": 1195.0,  # ~38 tokens @ 31.8 tok/s
        "total_end_to_end_inspection_cycle_seconds": 1.287,
        "compliance": "Sub-1.5s total line cycle compliant with zero cloud dependency"
    }
}

def main():
    print("=" * 70)
    print("Qualcomm AI Hub LLM Catalog Profiling & Full Pipeline Telemetry Note")
    print("=" * 70)
    print(f"Model: {QUALCOMM_LLM_PROFILE['model_name']}")
    print(f"Architecture: {QUALCOMM_LLM_PROFILE['architecture']}\n")
    
    print("--- Silicon Performance Comparison ---")
    for hw in QUALCOMM_LLM_PROFILE["target_hardware"]:
        print(f"Silicon: {hw['silicon']} ({hw['device']})")
        print(f"  Runtime: {hw['runtime']} | Quantization: {hw['quantization']}")
        print(f"  TTFT: {hw['ttft_ms']} ms ({hw['ttft_seconds']}s) | Prefill Speed: {hw['prefill_tokens_per_sec']} tok/s")
        print(f"  Decoding Throughput: {hw['decoding_tokens_per_sec']} tok/s | Activation RAM: {hw['activation_memory_mb']} MB\n")
    
    print("--- Full Dual-Engine Pipeline Latency Budget ---")
    pipe = QUALCOMM_LLM_PROFILE["dual_engine_pipeline_projection"]
    print(f"  Tier 1 Vision (YOLO INT8 @ Hexagon NPU): {pipe['tier1_vision_ms']:.2f} ms")
    print(f"  Telemetry JSON Formatting: {pipe['telemetry_serialization_ms']:.2f} ms")
    print(f"  Tier 2 Prefill / TTFT: {pipe['tier2_reasoning_prefill_ms']:.1f} ms")
    print(f"  Tier 2 Generation (~38 IPC Advisory Tokens): {pipe['tier2_reasoning_generation_ms']:.1f} ms")
    print(f"  --> TOTAL INSPECTION CYCLE: {pipe['total_end_to_end_inspection_cycle_seconds']:.3f} s")
    print(f"  --> Compliance: {pipe['compliance']}")
    
    # Save structured JSON
    out_json = os.path.join(OUTPUT_DIR, "qualcomm_llm_and_full_pipeline_projection.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(QUALCOMM_LLM_PROFILE, f, indent=2)
    print(f"\n[OK] Saved telemetry note to: {out_json}")

if __name__ == "__main__":
    main()
