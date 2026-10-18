# Qualcomm AI Hub On-Device Cloud Profiling Report
**Generated**: 2026-10-18 13:26:53 UTC  
**Platform**: Qualcomm AI Hub Cloud Device Farm (Real Physical Silicon)  
**Application**: Bharat AI-SoC Challenge (PS5: Edge Defect Detection & Quality Control)  

## 1. Cross-Device & Precision Comparison Table

| Target Device | Silicon / SoC | Precision | Median Latency | Mean Latency | Throughput | Peak RAM | NPU Delegation | Job URL |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Samsung Galaxy S24** | Snapdragon 8 Gen 3 (SM8650) | **INT8 (PTQ)** | **1.40 ms** | 1.44 ms | **693.1 FPS** | 151.6 MB | **100.0% (387/387 ops)** | [Dashboard](https://workbench.aihub.qualcomm.com/jobs/jpv281qmg/) |
| **Samsung Galaxy S24** | Snapdragon 8 Gen 3 (SM8650) | **FP32** | **3.45 ms** | 3.48 ms | **287.3 FPS** | 209.5 MB | **100.0% (355/355 ops)** | [Dashboard](https://workbench.aihub.qualcomm.com/jobs/jp1o9d2l5/) |
| **Samsung Galaxy S23** | Snapdragon 8 Gen 2 (SM8550) | **INT8 (PTQ)** | **2.07 ms** | 2.09 ms | **478.2 FPS** | 108.1 MB | **100.0% (387/387 ops)** | [Dashboard](https://workbench.aihub.qualcomm.com/jobs/jgd6krnlp/) |
| **Samsung Galaxy S23** | Snapdragon 8 Gen 2 (SM8550) | **FP32** | **5.48 ms** | 5.52 ms | **181.2 FPS** | 132.6 MB | **100.0% (355/355 ops)** | [Dashboard](https://workbench.aihub.qualcomm.com/jobs/j5wykd26g/) |
| **Samsung Galaxy S22 5G** | Snapdragon 8 Gen 1 (SM8450) | **INT8 (PTQ)** | **2.52 ms** | 2.56 ms | **391.1 FPS** | 151.1 MB | **100.0% (387/387 ops)** | [Dashboard](https://workbench.aihub.qualcomm.com/jobs/jg9or3jlg/) |
| **Samsung Galaxy A73 5G** | Snapdragon 778G (SM7325) | **INT8 (PTQ)** | **5.67 ms** | 5.73 ms | **174.4 FPS** | 104.0 MB | **100.0% (387/387 ops)** | [Dashboard](https://workbench.aihub.qualcomm.com/jobs/j5m97wqwg/) |

---

## 2. Statistical Latency Distribution (P50, P90, P95, P99, Jitter)

| Device | Precision | Min (ms) | P50 / Median | P90 (ms) | P95 (ms) | P99 (ms) | Max (ms) | StdDev (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Samsung Galaxy S24 | INT8 (PTQ) | 1.39 | **1.40** | 1.43 | 1.46 | 2.53 | 3.75 | ±0.26 |
| Samsung Galaxy S24 | FP32 | 3.38 | **3.45** | 3.51 | 3.53 | 3.65 | 6.56 | ±0.31 |
| Samsung Galaxy S23 | INT8 (PTQ) | 2.03 | **2.07** | 2.12 | 2.15 | 2.41 | 3.42 | ±0.14 |
| Samsung Galaxy S23 | FP32 | 5.42 | **5.48** | 5.61 | 5.64 | 5.88 | 6.81 | ±0.15 |
| Samsung Galaxy S22 5G | INT8 (PTQ) | 2.39 | **2.52** | 2.67 | 2.73 | 3.02 | 3.63 | ±0.15 |
| Samsung Galaxy A73 5G | INT8 (PTQ) | 5.46 | **5.67** | 5.92 | 6.19 | 6.39 | 6.68 | ±0.20 |

---

## 3. Hardware Compute Unit Allocation & Layer-by-Layer Profiling

### Samsung Galaxy S24 — INT8 (PTQ) (Snapdragon 8 Gen 3 (SM8650))
- **Total Layers**: 387
- **Hexagon NPU Ops**: `387` (100.0%)
- **GPU Fallback Ops**: `0`
- **CPU Fallback Ops**: `0`
- **First Load Time**: `865.5 ms` | **Warm Load Time**: `164.3 ms`
- **Operator Breakdown**: `CONV_2D`: 81, `MUL`: 81, `LOGISTIC`: 78, `QUANTIZE`: 31, `SLICE`: 23, `CONCATENATION`: 21, `ADD`: 16, `RESHAPE`: 16

**Top 5 Latency-Critical Layers:**

| Layer Name | Op Type | Compute Unit | Execution Time ($\mu s$) | Cycles |
| :--- | :---: | :---: | :---: | :---: |
| `serving_default_args_0_quantized` | `QUANTIZE` | **NPU** | 476 $\mu s$ | 877571 |
| `2` | `TRANSPOSE` | **NPU** | 361 $\mu s$ | 665554 |
| `4` | `CONV_2D` | **NPU** | 330 $\mu s$ | 608384 |
| `1` | `MUL` | **NPU** | 237 $\mu s$ | 436081 |
| `3` | `SOFTMAX` | **NPU** | 232 $\mu s$ | 427096 |


### Samsung Galaxy S24 — FP32 (Snapdragon 8 Gen 3 (SM8650))
- **Total Layers**: 355
- **Hexagon NPU Ops**: `355` (100.0%)
- **GPU Fallback Ops**: `0`
- **CPU Fallback Ops**: `0`
- **First Load Time**: `1728.0 ms` | **Warm Load Time**: `173.1 ms`
- **Operator Breakdown**: `CONV_2D`: 81, `MUL`: 81, `LOGISTIC`: 78, `SLICE`: 23, `CONCATENATION`: 21, `ADD`: 16, `RESHAPE`: 16, `TRANSPOSE`: 14

**Top 5 Latency-Critical Layers:**

| Layer Name | Op Type | Compute Unit | Execution Time ($\mu s$) | Cycles |
| :--- | :---: | :---: | :---: | :---: |
| `1` | `MUL` | **NPU** | 1327 $\mu s$ | 2446540 |
| `3` | `SOFTMAX` | **NPU** | 1017 $\mu s$ | 1873990 |
| `2` | `TRANSPOSE` | **NPU** | 920 $\mu s$ | 1695759 |
| `1` | `MUL` | **NPU** | 337 $\mu s$ | 621584 |
| `22` | `ADD` | **NPU** | 309 $\mu s$ | 569225 |


### Samsung Galaxy S23 — INT8 (PTQ) (Snapdragon 8 Gen 2 (SM8550))
- **Total Layers**: 387
- **Hexagon NPU Ops**: `387` (100.0%)
- **GPU Fallback Ops**: `0`
- **CPU Fallback Ops**: `0`
- **First Load Time**: `923.5 ms` | **Warm Load Time**: `211.4 ms`
- **Operator Breakdown**: `CONV_2D`: 81, `MUL`: 81, `LOGISTIC`: 78, `QUANTIZE`: 31, `SLICE`: 23, `CONCATENATION`: 21, `ADD`: 16, `RESHAPE`: 16

**Top 5 Latency-Critical Layers:**

| Layer Name | Op Type | Compute Unit | Execution Time ($\mu s$) | Cycles |
| :--- | :---: | :---: | :---: | :---: |
| `serving_default_args_0_quantized` | `QUANTIZE` | **NPU** | 579 $\mu s$ | 866908 |
| `2` | `TRANSPOSE` | **NPU** | 526 $\mu s$ | 787911 |
| `4` | `CONV_2D` | **NPU** | 480 $\mu s$ | 719155 |
| `1` | `MUL` | **NPU** | 323 $\mu s$ | 483361 |
| `3` | `SOFTMAX` | **NPU** | 291 $\mu s$ | 436011 |


### Samsung Galaxy S23 — FP32 (Snapdragon 8 Gen 2 (SM8550))
- **Total Layers**: 355
- **Hexagon NPU Ops**: `355` (100.0%)
- **GPU Fallback Ops**: `0`
- **CPU Fallback Ops**: `0`
- **First Load Time**: `1610.1 ms` | **Warm Load Time**: `225.2 ms`
- **Operator Breakdown**: `CONV_2D`: 81, `MUL`: 81, `LOGISTIC`: 78, `SLICE`: 23, `CONCATENATION`: 21, `ADD`: 16, `RESHAPE`: 16, `TRANSPOSE`: 14

**Top 5 Latency-Critical Layers:**

| Layer Name | Op Type | Compute Unit | Execution Time ($\mu s$) | Cycles |
| :--- | :---: | :---: | :---: | :---: |
| `1` | `MUL` | **NPU** | 2315 $\mu s$ | 3467265 |
| `3` | `SOFTMAX` | **NPU** | 1412 $\mu s$ | 2114125 |
| `2` | `TRANSPOSE` | **NPU** | 1260 $\mu s$ | 1887077 |
| `1` | `MUL` | **NPU** | 599 $\mu s$ | 897350 |
| `1` | `RESIZE_NEAREST_NEIGHBOR` | **NPU** | 380 $\mu s$ | 569413 |


### Samsung Galaxy S22 5G — INT8 (PTQ) (Snapdragon 8 Gen 1 (SM8450))
- **Total Layers**: 387
- **Hexagon NPU Ops**: `387` (100.0%)
- **GPU Fallback Ops**: `0`
- **CPU Fallback Ops**: `0`
- **First Load Time**: `831.8 ms` | **Warm Load Time**: `191.3 ms`
- **Operator Breakdown**: `CONV_2D`: 81, `MUL`: 81, `LOGISTIC`: 78, `QUANTIZE`: 31, `SLICE`: 23, `CONCATENATION`: 21, `ADD`: 16, `RESHAPE`: 16

**Top 5 Latency-Critical Layers:**

| Layer Name | Op Type | Compute Unit | Execution Time ($\mu s$) | Cycles |
| :--- | :---: | :---: | :---: | :---: |
| `serving_default_args_0_quantized` | `QUANTIZE` | **NPU** | 632 $\mu s$ | 947050 |
| `2` | `TRANSPOSE` | **NPU** | 527 $\mu s$ | 788803 |
| `4` | `CONV_2D` | **NPU** | 485 $\mu s$ | 726954 |
| `1` | `MUL` | **NPU** | 331 $\mu s$ | 494990 |
| `3` | `SOFTMAX` | **NPU** | 310 $\mu s$ | 464286 |


### Samsung Galaxy A73 5G — INT8 (PTQ) (Snapdragon 778G (SM7325))
- **Total Layers**: 387
- **Hexagon NPU Ops**: `387` (100.0%)
- **GPU Fallback Ops**: `0`
- **CPU Fallback Ops**: `0`
- **First Load Time**: `3004.2 ms` | **Warm Load Time**: `227.8 ms`
- **Operator Breakdown**: `CONV_2D`: 81, `MUL`: 81, `LOGISTIC`: 78, `QUANTIZE`: 31, `SLICE`: 23, `CONCATENATION`: 21, `ADD`: 16, `RESHAPE`: 16

**Top 5 Latency-Critical Layers:**

| Layer Name | Op Type | Compute Unit | Execution Time ($\mu s$) | Cycles |
| :--- | :---: | :---: | :---: | :---: |
| `serving_default_args_0_quantized` | `QUANTIZE` | **NPU** | 735 $\mu s$ | 1058669 |
| `2` | `TRANSPOSE` | **NPU** | 500 $\mu s$ | 720408 |
| `4` | `CONV_2D` | **NPU** | 429 $\mu s$ | 617610 |
| `1` | `MUL` | **NPU** | 401 $\mu s$ | 577911 |
| `3` | `SOFTMAX` | **NPU** | 318 $\mu s$ | 458638 |

---

## 4. Option A: Qualcomm AI Hub LLM Catalog Profiling & End-to-End Dual-Engine Projection

### 4.1 On-Device Llama 3.2 1B Architecture: Qualcomm Hexagon NPU (QNN) vs. Arm CPU (ExecuTorch)

For the Tier 2 reasoning engine, we compare the official **Qualcomm AI Hub QNN / GenieX Hexagon NPU runtime** with our **PyTorch ExecuTorch Arm NEON CPU reference runtime**:

| Evaluation Metric | Qualcomm Snapdragon 8 Gen 3 (SM8650) | Qualcomm Snapdragon 8 Elite (SM8750) | Arm Cortex-A Edge (ExecuTorch Reference) |
| :--- | :---: | :---: | :---: |
| **Silicon / Processor** | Hexagon HTP v75 (NPU) | Hexagon HTP v79 (NPU) | Armv8-A (Cortex-A73 / A53 / Host) |
| **Execution Runtime** | Qualcomm QNN / GenieX | Qualcomm QNN / GenieX | ExecuTorch 1.5.1 + XNNPACK / KleidiAI |
| **Quantization Format** | 4-Bit Weights (`w4a16` / select `w8`) | 4-Bit Weights (`w4a16`) | `8da4w` (Groupwise 64 INT4) |
| **Time to First Token (TTFT)** | **90.0 ms (0.09 s)** | **70.0 ms (0.07 s)** | **190.0 ms (0.19 s)** |
| **Prefill Throughput** | **1,420.0 tok/s** | **2,057.0 tok/s** | **657.9 tok/s** |
| **Decoding Throughput** | **31.8 tok/s** | **33.5 tok/s** | **20.4 – 22.8 tok/s (guarded)** |
| **Activation RAM Footprint** | ~28.5 MB | ~26.0 MB | **24.89 MB (26,107,712 bytes)** |
| **Model Footprint in RAM** | ~1.15 GB | ~1.12 GB | **1.70 GB (Zero-Java POSIX `mmap`)** |

### 4.2 End-to-End Dual-Engine Pipeline Latency Budget on Qualcomm Silicon

When the visual defect detection (Tier 1) and IPC-A-610 reasoning (Tier 2) are chained on Snapdragon 8 Gen 3 silicon:

```
[Optical Frame Ingestion] 
       │ 
       ▼
[Tier 1: INT8 YOLO11n @ Hexagon NPU] ─────────► 1.40 ms (693 FPS)
       │
       ▼ Event Trigger (Confidence >= 0.80)
[JSON Telemetry Serialization] ──────────────► 0.45 ms
       │
       ▼
[Tier 2: Llama 3.2 1B TTFT / Prefill] ────────► 90.0 ms
       │
       ▼
[Tier 2: IPC Advisory Generation (~38 tokens)] ► 1,195.0 ms (@ 31.8 tok/s)
       │
       ▼
[TOTAL END-TO-END AUDIT CYCLE] ──────────────► 1.287 seconds
```

* **Production Line Verdict**: The complete dual-engine inspection completes in **under 1.3 seconds per flagged board** entirely on edge silicon with **zero cloud communication**, meeting the strict takt time requirements for electronics surface-mount assembly lines.


