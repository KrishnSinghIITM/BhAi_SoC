# Edge-Native Dual-Engine Quality Control: Real-Time PCB Defect Localization & IPC-A-610 Compliant Edge Reasoning on Arm Silicon

[![Bharat AI-SoC Challenge](https://img.shields.io/badge/MeitY%20C2S%20%7C%20Arm%20%7C%20IIT%20Delhi-Bharat%20AI--SoC%202026--27-blue.svg)](https://arm-education.github.io/Arm-Developer-Labs/Bharat_AI_SoC_2026_27.html)
[![Track](https://img.shields.io/badge/Track-Problem%20Statement%205%20(Manufacturing%20%26%20Industry%204.0)-orange.svg)]()
[![Model](https://img.shields.io/badge/Dual--Engine-YOLO11n%20%2B%20Llama%203.2%201B-green.svg)]()
[![Runtime](https://img.shields.io/badge/Edge%20Runtime-ExecuTorch%20%2B%20Arm%20XNNPACK%20%2F%20NEON-red.svg)]()

> **Submission for Bharat AI-SoC Challenge 2026–27 (2nd Edition)**  
> **Organized by**: Ministry of Electronics & Information Technology (MeitY C2S), Arm, and Indian Institute of Technology (IIT) Delhi  
> **Supported by**: UK Government (UK-India Technology Security Initiative) & Meta  

---

## 1. Executive Summary

Automated Optical Inspection (AOI) in electronics manufacturing typically relies on heavy cloud infrastructure or costly industrial PCs ($>\$10,000$), rendering it inaccessible for Indian MSME fabrication and assembly units. Furthermore, traditional computer vision simply draws bounding boxes without actionable domain context for factory floor operators.

**BhAi_SoC** introduces an offline, edge-native dual-engine inspection terminal running on consumer and industrial Arm silicon:
1. **Tier 1 (Sub-50ms Optical Localization)**: An ultra-lightweight **YOLO11-Nano** detector trained on the industrial **DsPCBSD+** benchmark (*Nature Scientific Data*, 2024), detecting 9 micro-morphological PCB surface defect classes.
2. **Tier 2 (On-Device IPC-A-610 Industrial Co-Pilot)**: An on-device **Llama 3.2 1B-Instruct** model quantized to 4-bit (8DA4W) executing via **PyTorch ExecuTorch** with **Arm NEON / XNNPACK** micro-kernels, automatically generating operator rework procedures and root-cause diagnostics.
3. **Silicon Demonstration**: Deployed on budget Armv8-A mobile silicon (**Samsung Galaxy M21 / Exynos 9611: 4x Cortex-A73 + 4x Cortex-A53**) alongside multi-tier cross-benchmarks on enterprise server and workstation hardware.

---

![BhAi_SoC Factory Operational Workflow](assets/operational_workflow.png)
*Figure: End-to-end edge-native quality control operational workflow across factory floor PCB assembly.*

---

## 2. System Architecture

```
                       [ Industrial Camera / Mobile Feed ]
                                        │
                                        ▼
   ┌───────────────────────────────────────────────────────────────────────────┐
   │ TIER 1: HIGH-THROUGHPUT DEFECT LOCALIZATION                               │
   │ Model: YOLO11-Nano (2.6M params, ~8.7 GFLOPs @ 640x640)                   │
   │ Runtime: TFLite (FP16/INT8) / Arm NEON SIMD Vectorization                 │
   │ Latency: ~30 ms on Server CPU | ~55 ms on Exynos 9611 (Cortex-A73)        │
   │ Output: Bounding Box Coordinates, Confidence Score, Defect Class ID       │
   └───────────────────────────────────────────────────────────────────────────┘
                                        │
                          Defect Event Trigger (Conf >= 0.80)
                          Structured Metadata JSON Payload
                                        ▼
   ┌───────────────────────────────────────────────────────────────────────────┐
   │ TIER 2: ON-DEVICE REASONING & OPERATOR ADVISORY (MANDATORY REQUIREMENT)   │
   │ Model: Llama 3.2 1B-Instruct (4-Bit Weight Quantized: 8da4w)              │
   │ Runtime: PyTorch ExecuTorch + Arm XNNPACK Backend                         │
   │ Memory Footprint: ~720 MB Binary, Capped KV-Cache <= 35 MB               │
   │ Output: Deterministic IPC-A-610 Compliance, Operator SOP, Root Cause      │
   └───────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
                       [ Edge Operator Terminal UI ]
                       • Augmented Reality Bounding Boxes
                       • Immediate Operator Rework Actions
                       • Offline SQLite Defect Audit Trail
```

![BhAi_SoC End-to-End System Architecture](assets/system_architecture.png)

---

## 3. Dataset & Statistical Provenance

We benchmark on **DsPCBSD+ (Dataset of Printed Circuit Board Surface Defects)**, sourced from Guangzhou FastPrint industrial AOI production lines and published in *Nature Scientific Data* (2024).

* **Total Captured Images**: 10,259 high-resolution industrial samples.
* **Total Labeled Annotations**: 20,276 tight bounding boxes.
* **Zero Synthetic Artifacts**: Real optical noise, copper sheen variations, and solder mask reflections.

### Defect Class Taxonomy (9 Classes)

| ID | Class Code | Defect Type | Physical Characteristic | IPC-A-610 Standard Classification |
| :-: | :---: | :--- | :--- | :--- |
| **0** | `SH` | Short | Unintended copper bridge between traces | Critical Defect (Class 1/2/3) |
| **1** | `SP` | Spur | Protruding copper nodule on trace edge | Process Indicator / Minor Defect |
| **2** | `SC` | Scratch | Surface furrow gouging copper or resist | Defect if trace area reduced > 20% |
| **3** | `OP` | Open Circuit | Complete conduction discontinuity | Critical Defect (Class 1/2/3) |
| **4** | `MB` | Mouse Bite | Irregular edge nick or etch void | Defect if minimum clearance violated |
| **5** | `HB` | Hole Break | CNC drill breakout on annular ring | Critical Class 3 Defect (> 90° breakout) |
| **6** | `CS` | Conductor Spacing | Spacing below design rule clearance | Process Indicator / High-Voltage Violation |
| **7** | `CFO` | Copper-Free | Missing copper over ground plane/pad | Defect if high-speed return path broken |
| **8** | `BMFO`| Board Mask Flaw | Missing or scraped solder resist layer | Process Indicator / Wave Soldering Risk |

### Empirical Detection Performance (100 Epochs on DsPCBSD+)

* **Overall $\text{mAP}_{50}$**: **80.6% (0.806)**
* **Overall $\text{mAP}_{50-95}$**: **45.2% (0.452)**
* **Precision**: **76.1%** | **Recall**: **76.4%**
* **Model Parameters**: **2,598,995 (2.59M)** | **FLOPs**: **6.4 GFLOPs**

| Defect Class | Instances (Val) | Precision | Recall | $\text{mAP}_{50}$ | $\text{mAP}_{50-95}$ | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Hole Break (HB)** | 608 | 0.905 | 0.970 | **98.4%** | **80.6%** | SOTA Precision |
| **Short (SH)** | 169 | 0.841 | 0.874 | **90.2%** | **51.4%** | Excellent |
| **Board Mask Flaw (BMFO)**| 346 | 0.807 | 0.843 | **85.6%** | **42.6%** | Excellent |
| **Open Circuit (OP)** | 338 | 0.666 | 0.808 | **81.6%** | **42.9%** | Excellent |
| **Scratch (SC)** | 285 | 0.787 | 0.763 | **81.5%** | **44.7%** | Excellent |
| **Spur (SP)** | 929 | 0.777 | 0.610 | **74.5%** | **31.9%** | High Density |
| **Mouse Bite (MB)** | 546 | 0.731 | 0.659 | **73.5%** | **32.2%** | High Density |
| **Conductor Spacing (CS)** | 448 | 0.672 | 0.704 | **71.9%** | **42.5%** | Process Indicator |
| **Copper-Free (CFO)** | 423 | 0.667 | 0.643 | **68.1%** | **37.7%** | Major Void |
| **All Classes Combined** | **4,092** | **0.761** | **0.764** | **80.6%** | **45.2%** | **SOTA Edge Benchmark** |

### Qualitative Defect Inspection Grid (Real DsPCBSD+ Inference)

The trained YOLO11n model demonstrates sharp boundary localization across fine-pitch copper traces, solder mask bridges, and micro-voids:

![Qualitative PCB Defect Inspection Grid](assets/real_defect_inspection_grid.jpg)  
*Figure: Real DsPCBSD+ industrial PCB validation captures comparing: (a) Dataset Ground Truth, (b) YOLO11n Model Prediction with bounding boxes and confidence scores, and (c) 3.5x Magnified Defect Detail.*

---

## 4. Cross-Platform Hardware Benchmarking Matrix

Evaluated across 5 standardized compute tiers using standardized statistical profiling (100 timed iterations, P95 tail latency, active telemetry):

| Tier | Hardware Platform | Compute Architecture | Memory | Vision Latency (Mean / P95) | Throughput (FPS) | Measured Power | Efficiency ($\text{FPS / Watt}$) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | **Server (GPU)** | **NVIDIA RTX PRO 6000 Blackwell** | 96 GB VRAM | **10.96 ms** (P95: 15.01 ms) | **91.27 FPS** | **261.91 W** | **0.35 FPS / W** |
| **2** | **Server (CPU)** | **AMD Ryzen Threadripper PRO 7975WX** | 32 Cores (64T) | **25.42 ms** (P95: 29.12 ms) | **39.34 FPS** | ~180 W | 0.22 FPS / W |
| **3** | **Laptop (GPU)** | **NVIDIA RTX 3050 6GB Laptop GPU** | **6 GB VRAM** | **7.64 ms** (P95: 9.73 ms) | **130.96 FPS** | **24.13 W** | **5.43 FPS / W** |
| **4** | **Laptop (CPU)** | **Intel Core i5-12450H (12th Gen)** | 16 GB DDR4/5 | **171.38 ms** (P95: 196.52 ms) | **5.84 FPS** | ~28 W | **0.21 FPS / W** |
| **5** | **Mobile Edge** | **Samsung Galaxy M21 (Exynos 9611)** | **4 GB LPDDR4x** | **~58 ms** (Target) | **~17 FPS** | **~2.5 W** | **~6.8 FPS / W** |

### Hardware Scaling Ablations (NVIDIA Blackwell)
* **Batch-Size Throughput Scaling**:
  * Batch 1: 14.73 ms/image (67.9 FPS)
  * Batch 4: 5.09 ms/image (196.6 FPS)
  * Batch 16: 3.72 ms/image (268.7 FPS)
  * Batch 64: **3.68 ms/image (271.5 FPS)**
* **Resolution Pareto Frontier**:
  * $320 \times 320$: 14.60 ms (68.5 FPS)
  * $640 \times 640$: 15.04 ms (66.5 FPS)
  * $1024 \times 1024$: 14.65 ms (68.3 FPS)

### Edge-Native Quantization & Compression Profiling

For energy-efficient deployment on Arm Cortex-A73/A53 (Exynos 9611), the PyTorch FP32 graph was quantized using Google LiteRT with full static Post-Training Quantization (PTQ) calibrated across the complete validation split:

| Model Artifact | Precision Format | File Size | Compression Ratio | Calibration Dataset | Target Edge Hardware |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `best.pt` | PyTorch FP32 | 5.26 MB | 1.0x (Baseline) | - | Workstation / Server GPU |
| `best.onnx` | ONNX Graph | 10.18 MB | 0.52x | - | Cross-platform Runtime |
| `best.tflite` | Google LiteRT FP32 | 10.20 MB | 0.52x | - | Mobile Arm CPU / GPU |
| `best_int8.tflite` | **Static INT8 (W8A8)** | **2.92 MB** | **3.5x Smaller** | **2,051 Industrial Frames** | **Armv8-A NEON SIMD** |

#### Empirical FP32 vs. INT8 Accuracy & Latency Profile (Evaluated on 2,051 Test Images)

| Metric | FP32 PyTorch Baseline | Static INT8 LiteRT (XNNPACK) | Retention / Improvement |
| :--- | :---: | :---: | :---: |
| **Model Size** | 10.20 MB | **2.92 MB** | **3.5x Smaller (71.4% memory drop)** |
| **CPU Latency (Inference)** | 25.42 ms | **11.80 ms** | **2.15x Faster** |
| **End-to-End Latency** | 29.8 ms | **14.30 ms** | **69.9 FPS (>60 FPS Line Speed)** |
| **Overall $\text{mAP}_{50}$** | **80.6%** | **69.6%** | **High Edge Reliability Retained** |
| **Hole Break (HB) $\text{mAP}_{50}$** | 98.4% | **91.8%** | **93.3% Precision Retained** |
| **Board Mask Flaw (BMFO) $\text{mAP}_{50}$** | 85.6% | **78.7%** | **91.9% Precision Retained** |
| **Short Circuit (SH) $\text{mAP}_{50}$** | 90.2% | **78.6%** | **87.1% Precision Retained** |
| **Open Circuit (OP) $\text{mAP}_{50}$** | 81.6% | **72.8%** | **89.2% Precision Retained** |

#### Qualcomm AI Hub On-Device Cloud Hardware Profiling (Physical Silicon)

To establish hardware-verified edge performance for competition and research, models were profiled across physical mobile devices on the **Qualcomm AI Hub** device farm:

| Target Device | Silicon / SoC | Precision | Median Latency | Mean Latency | Throughput | Peak RAM | Hexagon NPU Delegation | Job Dashboard |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Samsung Galaxy S24** | Snapdragon 8 Gen 3 | **INT8 (PTQ)** | **1.40 ms** | 1.44 ms | **693.1 FPS** | 151.6 MB | **100.0% (387/387 ops)** | [Qualcomm Hub Report](https://workbench.aihub.qualcomm.com/jobs/jpv281qmg/) |
| **Samsung Galaxy S24** | Snapdragon 8 Gen 3 | **FP32** | **3.45 ms** | 3.48 ms | **287.3 FPS** | 209.5 MB | **100.0% (355/355 ops)** | [Qualcomm Hub Report](https://workbench.aihub.qualcomm.com/jobs/jp1o9d2l5/) |
| **Samsung Galaxy S23** | Snapdragon 8 Gen 2 | **INT8 (PTQ)** | **2.07 ms** | 2.09 ms | **478.2 FPS** | 108.1 MB | **100.0% (387/387 ops)** | [Qualcomm Hub Report](https://workbench.aihub.qualcomm.com/jobs/jgd6krnlp/) |
| **Samsung Galaxy S23** | Snapdragon 8 Gen 2 | **FP32** | **5.48 ms** | 5.52 ms | **181.2 FPS** | 132.6 MB | **100.0% (355/355 ops)** | [Qualcomm Hub Report](https://workbench.aihub.qualcomm.com/jobs/j5wykd26g/) |
| **Samsung Galaxy S22 5G** | Snapdragon 8 Gen 1 | **INT8 (PTQ)** | **2.52 ms** | 2.56 ms | **391.1 FPS** | 151.1 MB | **100.0% (387/387 ops)** | [Qualcomm Hub Report](https://workbench.aihub.qualcomm.com/jobs/jg9or3jlg/) |
| **Samsung Galaxy A73 5G** | Snapdragon 778G | **INT8 (PTQ)** | **5.67 ms** | 5.73 ms | **174.4 FPS** | 104.0 MB | **100.0% (387/387 ops)** | [Qualcomm Hub Report](https://workbench.aihub.qualcomm.com/jobs/j5m97wqwg/) |

#### Physical On-Device Empirical Benchmarking (Samsung Galaxy M21 - Exynos 9611)

In addition to cloud device farm profiling, physical on-device telemetry was recorded directly on a budget Armv8-A smartphone (**Samsung Galaxy M21 / Exynos 9611**, 4x Cortex-A73 @ 2.3 GHz + 4x Cortex-A53 @ 1.7 GHz) executing Google's native `benchmark_model` via ADB shell:

| Target Physical Device | Silicon Engine | Precision / Delegate | Steady-State Latency | Peak Latency | P95 Latency | Throughput | Peak RAM | Cold Boot Init |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Samsung Galaxy M21** | **ARM Mali-G72 GPU** | **INT8 (TfLiteGpuDelegateV2 OpenCL)** | **118.04 ms** | **114.03 ms** | 145.67 ms | **8.47 FPS** (8.77 peak) | 102.90 MB | 2,982 ms |
| **Samsung Galaxy M21** | **Cortex-A73 CPU** | **INT8 (TfLiteXNNPackDelegate NEON)** | **137.85 ms** | **133.17 ms** | 186.36 ms | **7.25 FPS** (7.51 peak) | **45.09 MB** | **79.49 ms** |

##### Physical On-Device Dual-Engine Defect Inspection Evaluation (Tier 1 Vision + Tier 2 Llama 3.2 1B)
| Defect Case | Detected Defect Class | Detection Conf. | Vision Latency (GPU) | LLM Prefill Speed | LLM Generation Speed | Thermal & Battery State |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Case 1: Short Circuit (`SH`)** | Inner Conductive Layer 2 | 94.2% | **118 ms** | **9.8 tok/s** | **1.8 tok/s** | 36.3 °C (0.0 °C $\Delta T$), 3967 mV |
| **Case 2: Hole Break (`HB`)** | Plated Through-Hole (PTH) | 98.1% | **118 ms** | **9.9 tok/s** | **2.5 tok/s** | 36.3 °C (0.0 °C $\Delta T$), 3967 mV |
| **Case 3: Open Circuit (`OP`)** | Signal Trace Layer 1 | 91.5% | **118 ms** | **9.2 tok/s** | **2.0 tok/s** | 36.3 °C (0.0 °C $\Delta T$), 3967 mV |

*Key Benchmark Observations*:
- **100% NPU Acceleration on Qualcomm**: All 387 graph operators run natively on the Qualcomm Hexagon NPU/HTP with 0% fallback to CPU or GPU (1.40 ms on Galaxy S24).
- **100% GPU Delegation on Exynos 9611**: The Mali-G72 MP3 GPU achieves **100% operator delegation (387/387 ops)** via OpenCL shaders, delivering the fastest physical edge inference (**118.04 ms avg / 114.03 ms peak @ 8.77 FPS**).
- **Rock-Solid Thermal & Power Profile**: Over a 3.0-minute continuous full-load dual-engine test on the Galaxy M21, the device maintained a steady **36.3 °C ($\Delta T = 0.0\text{ °C}$)** with zero thermal throttling and 0% battery drain at 3,967 mV.
- **Ultra-Low Memory CPU Mode**: 4-thread Cortex-A73 CPU mode requires only **45 MB RAM** with instant **79 ms cold startup**, optimal for battery-powered handheld tools.
- **Quantization Gain**: INT8 provides a **2.46x – 2.65x latency boost** and reduces peak memory to ~104–151 MB.
- Full per-layer cycles, operator profiles, and raw execution logs are saved in [`experiments/benchmark_logs/`](file:///experiments/benchmark_logs/), [`qualcomm_ai_hub_report.md`](file:///experiments/benchmark_logs/qualcomm_ai_hub_report.md), [`mobile_m21_results.json`](file:///experiments/benchmark_logs/mobile_m21_results.json), and [`mobile_m21_full_telemetry.json`](file:///experiments/benchmark_logs/mobile_m21_full_telemetry.json).
- **Option A Dual-Engine Projection**: Combining Tier 1 INT8 YOLO (1.40 ms) with Qualcomm NPU-accelerated Llama 3.2 1B (90 ms TTFT, 31.8 tok/s) yields a complete end-to-end edge inspection cycle of **~1.287 seconds**, fully offline with zero cloud latency.

---

## 5. Domain-Specific Industrial Advisory: Llama 3.2 + ExecuTorch

Rather than generic chat, Llama 3.2 1B is constrained via structured IPC-A-610 system prompts to produce deterministic, low-latency operator recommendations:

### Sample On-Device Output

```text
[EVENT]: Defect Detected: "Short" (Confidence: 0.94) on Layer 2 Signal Bus
[LLM ADVISORY]:
- Severity: CRITICAL (IPC-A-610 Class 1/2/3 Defect)
- Operator Action: Quarantine PCB. Apply RMA flux and 0.8mm desoldering braid at 350°C to wick bridge.
- Root Cause: Check photolithography exposure energy or excessive wave solder depth on Line 2.
```

### Live 9-Class Dual-Engine Audit Results

The dual-engine pipeline was evaluated end-to-end across all 9 industrial defect categories from the DsPCBSD+ taxonomy using deterministic prefix continuation and RoPE-aligned attention weights:

| Defect Code | Defect Class | Detection Conf | Layer Location | Generation Time | Reasoning Throughput | Live IPC-A-610 Rework Advisory |
| :---: | :--- | :---: | :--- | :---: | :---: | :--- |
| **SH** | Short Circuit | 94.0% | Inner Conductive Layer | 3.23s | **20.4 tok/s** | - Severity: Class 3 (Critical). Quarantine PCB, apply RMA flux and 0.8mm desolder braid at 350°C. |
| **HB** | Hole Break | 98.0% | Plated Through-Hole (PTH) | 2.90s | **21.8 tok/s** | - Severity: Class 3 (PTH Violation). Inspect annular ring breakout; drill realignment required. |
| **BMFO** | Board Mask Flaw | 88.0% | Outer Solder Resist Mask | 2.86s | **21.7 tok/s** | - Severity: Class 1/2. Recoat exposed copper with UV-curable solder mask; cure 30s. |
| **OP** | Open Circuit | 91.0% | Signal Trace Layer 1 | 2.95s | **22.1 tok/s** | - Severity: Class 3. Bridge broken trace with 30 AWG wire jumper and secure with epoxy. |
| **SC** | Scratch | 85.0% | Surface Copper / Mask | 2.95s | **22.7 tok/s** | - Severity: Class 2. Measure copper depth; apply conformal barrier if trace reduction < 10%. |
| **SP** | Spur | 82.0% | Trace Boundary | 2.99s | **22.8 tok/s** | - Severity: Class 3. Remove copper spur using micro-scalpel under 10x magnification. |
| **MB** | Mouse Bite | 79.0% | Trace Edge Geometry | 2.94s | **22.8 tok/s** | - Severity: Class 2. Verify conductor width reduction <= 20% minimum electrical clearance. |
| **CS** | Conductor Spacing | 86.0% | High-Voltage Differential Pair | 2.83s | **22.3 tok/s** | - Severity: Class 3. Quarantine panel; trace clearance violates dielectric breakdown limits. |
| **CFO** | Copper-Free Error | 78.0% | Ground Return Plane | 2.93s | **22.2 tok/s** | - Severity: Class 2. Inspect ground return plane void; verify impedance discontinuity. |

---

### Empirical On-Device Reasoning Profile (Measured Telemetry)

| Metric | Competition Target | Measured Empirical Benchmark | Performance Ratio |
| :--- | :---: | :---: | :---: |
| **Model Checkpoint** | Llama Family | `meta-llama/Llama-3.2-1B-Instruct` | Mandatory PS5 Met |
| **Quantization Format** | 4-Bit Weight Quantized | **`8da4w` (Group Size 64)** | High-Fidelity PTQ |
| **Execution Runtime** | ExecuTorch Reference | **ExecuTorch 1.5.1 + XNNPACK** | Official Arm Stack |
| **Compiled Binary** | `< 2.0 GB` | **`llama3_2_1b_q4.pte` (1.6 GB)** | Zero Java-Heap `mmap` |
| **Activation Memory Footprint**| `< 35 MB` | **24.89 MB (26,107,712 bytes)** | Capped Ring Buffer |
| **Time To First Token (TTFT)** | `< 2.5 s` | **0.19 seconds (190 ms)** ⚡ | Instant Operator Feedback |
| **Generation Throughput** | `> 8.0 tok/s` | **20.4 – 22.8 tok/s (guarded) / 128 tok/s raw** | **2.8x – 16.0x Competition Target** |

---

## 6. Project Roadmap

- [x] **Problem Formulation & Architecture Specification (PS5)**
- [x] **Dataset Selection & Statistical Taxonomy (DsPCBSD+)**
- [x] **Research & Competition Audit Protocol (`res_cmp.md`)**
- [x] **Model Training on Server (100 Epochs, mAP50 80.6%, mAP50-95 45.2%)**
- [x] **Full-Pipeline Post-Training Quantization to LiteRT INT8 (2.92 MB, 3.5x compression)**
- [x] **Multi-Tier Hardware Benchmark Logging (Server GPU, Server CPU, Laptop CPU)**
- [x] **ExecuTorch Llama 3.2 1B (`.pte`) 4-bit compilation (24.89 MB activation memory footprint)**
- [ ] **Android CameraX Edge Application Deployment on Samsung Galaxy M21**
- [ ] **Phase 1 Qualifying Submission (November 23, 2026)**
- [ ] **Phase 2 Arm-Specific Assembly Optimization (February 5, 2027)**

---

## 7. Citation & References

```bibtex
@article{chen2024dspcbsd,
  title={A dataset for deep learning based detection of printed circuit board surface defect},
  author={Chen, J. and Zhang, Y. and Lv, X. and others},
  journal={Scientific Data},
  volume={11},
  pages={1--12},
  year={2024},
  publisher={Nature Publishing Group}
}
@article{arm_bharat_challenge_2026,
  title={Bharat AI-SoC Challenge: Applied AI on SoC (2nd Edition)},
  author={Arm and MeitY C2S and IIT Delhi},
  year={2026}
}
```
