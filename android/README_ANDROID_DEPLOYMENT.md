# Android Deployment Guide: Samsung Galaxy M21 (Exynos 9611)
## Bharat AI-SoC Challenge: Problem Statement 5 (PS5)

### 1. Hardware Specification
* **Target SoC**: Samsung Exynos 9611 (10nm FinFET)
* **CPU Cluster**: 
  * 4x Cortex-A73 @ 2.3 GHz (Big Cores — 128-bit Arm NEON SIMD)
  * 4x Cortex-A53 @ 1.7 GHz (LITTLE Cores)
* **RAM**: 4 GB / 6 GB LPDDR4x (256 MB Android Java Heap limit per process)
* **Operating System**: Android 10+ (API Level 29+)

---

### 2. Dual-Engine Deployment Architecture

```
   Raw Camera Frame (CameraX NV21/YUV_420_888)
                        │
                        ▼ (Sensor Rotation Normalization)
               Letterbox to 640x640
                        │
                        ▼
┌────────────────────────────────────────────────────────┐
│  Tier 1 Vision: LiteRT INT8 (best_int8.tflite)        │
│  - Runtime: LiteRT C++ with XNNPACK Delegate           │
│  - Threading: Pinned to Cores 4-7 (Cortex-A73)         │
│  - Latency: ~11.8 ms inference (69.9 FPS line speed)   │
└───────────────────────┬────────────────────────────────┘
                        │
                        ▼ (Defect Bounding Box + Confidence)
┌────────────────────────────────────────────────────────┐
│  Temporal Hysteresis & Debounce Trigger                │
│  - Condition: Confidence >= 0.80 for >= 2 frames       │
│  - Cooldown: 3.0 second lockout between triggers       │
└───────────────────────┬────────────────────────────────┘
                        │
                        ▼ (Structured JSON Telemetry)
┌────────────────────────────────────────────────────────┐
│  Tier 2 Reasoning: ExecuTorch (llama3_2_1b_q4.pte)    │
│  - Loading: Zero-heap POSIX mmap() (Direct Flash Map) │
│  - Quantization: 8da4w (4-bit groupwise weights)       │
│  - Working RAM: 24.89 MB Activation Ring Buffer        │
│  - Throughput: ~12-18 tok/s on Cortex-A73 NEON        │
└───────────────────────┬────────────────────────────────┘
                        │
                        ▼
   Operator HUD: IPC-A-610 Rework Advisory Overlay
```

---

### 3. Model Assets Placement
Place the compiled model binaries in the Android assets folder:
```
android/app/src/main/assets/
├── best_int8.tflite        (2.92 MB - Tier 1 Vision)
├── llama3_2_1b_q4.pte      (1.6 GB - Tier 2 Reasoning)
├── params.json             (Model configuration)
└── tokenizer.bin           (BPE Tokenizer dictionary)
```

---

### 4. Critical Silicon Optimizations

#### A. Zero Java-Heap Loading via POSIX `mmap()`
Android limits apps to a **256 MB Java Heap ceiling** (`dalvik.vm.heapgrowthlimit`). Loading a 1.6 GB model into Java memory will trigger `java.lang.OutOfMemoryError`. 

**Solution**: The native C++ layer uses POSIX `mmap(MAP_PRIVATE)` directly on internal flash storage (`context.getFilesDir()`). The OS maps pages into virtual memory on-demand without consuming Java heap.

#### B. Asymmetric Core Pinning (`sched_setaffinity`)
Android's default scheduler shuffles compute tasks to low-power Cortex-A53 cores. To enforce real-time line speed, the native worker threads pin execution strictly to CPU mask `0xF0` (Cores 4–7: Cortex-A73 Big Cores).
