# Local Laptop Cross-Engine Benchmark Report

**Timestamp**: 2026-10-18 08:24:24 UTC
**Host Platform**: Windows (Intel64 Family 6 Model 154 Stepping 3, GenuineIntel)
**Discrete GPU**: NVIDIA GeForce RTX 3050 6GB Laptop GPU

| Runtime Engine | Target Hardware | Mean Latency (ms) | P95 Tail (ms) | Throughput (FPS) | Peak RAM (MB) | Peak VRAM (MB) | GPU Power (W) | Efficiency (FPS/W) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| PyTorch FP32 | CPU | **188.89** | 286.90 | **5.29** | 387.4 | - | 6.87 W | 0.77 |
| ONNX Runtime (CPU) | CPU | **587.74** | 784.16 | **1.70** | 515.9 | - | 6.76 W | 0.25 |
| ONNX Runtime (DirectML) | RTX 3050 (DirectML) | **7.64** | 9.73 | **130.96** | 654.6 | 722.7 | 24.13 W | 5.43 |
| LiteRT INT8 (XNNPACK) | CPU | **207.14** | 272.32 | **4.83** | 666.9 | - | - | - |
