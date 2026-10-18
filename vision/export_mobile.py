"""
export_mobile.py
Edge Compilation & Quantization Pipeline
Exports trained PyTorch weights to mobile-ready formats:
1. ONNX (Cross-platform runtime)
2. TFLite / LiteRT Standard (Baseline Android runner)
3. TFLite / LiteRT INT8 (Arm NEON / SDOT quantized)
"""

import os
import json
import argparse
from pathlib import Path
from ultralytics import YOLO

def export_for_mobile(weights_path, data_path="data.yaml", imgsz=640, output_dir="experiments/exported_models"):
    os.makedirs(output_dir, exist_ok=True)
    print(f"[+] Loading model weights: {weights_path}")
    model = YOLO(weights_path)
    
    export_manifest = {
        "source_weights": weights_path,
        "input_resolution": [imgsz, imgsz],
        "formats": {}
    }
    
    # 1. Export to ONNX (if not already exported)
    onnx_path = weights_path.replace(".pt", ".onnx")
    if not os.path.exists(onnx_path):
        print("[+] Exporting to ONNX format...")
        onnx_file = model.export(format="onnx", imgsz=imgsz, opset=12)
    else:
        onnx_file = onnx_path
        
    onnx_size = os.path.getsize(onnx_file) / (1024**2)
    export_manifest["formats"]["onnx"] = {
        "path": onnx_file,
        "size_mb": round(onnx_size, 2)
    }
    print(f"[✓] ONNX Ready: {onnx_file} ({onnx_size:.2f} MB)")
    
    # 2. Export to TFLite (Standard FP32 / LiteRT baseline)
    print("\n[+] Exporting to TFLite (Standard Android LiteRT)...")
    try:
        tflite_std = model.export(format="tflite", imgsz=imgsz)
        std_size = os.path.getsize(tflite_std) / (1024**2)
        export_manifest["formats"]["tflite_standard"] = {
            "path": tflite_std,
            "size_mb": round(std_size, 2)
        }
        print(f"[✓] TFLite Standard Exported: {tflite_std} ({std_size:.2f} MB)")
    except Exception as e:
        print(f"[!] TFLite standard export note: {e}")

    # 3. Export to TFLite (INT8 PTQ for Arm NEON)
    print("\n[+] Exporting to TFLite INT8 (Post-Training Quantization for Arm NEON)...")
    try:
        tflite_int8 = model.export(format="tflite", imgsz=imgsz, int8=True, data=data_path)
        int8_size = os.path.getsize(tflite_int8) / (1024**2)
        export_manifest["formats"]["tflite_int8"] = {
            "path": tflite_int8,
            "size_mb": round(int8_size, 2)
        }
        print(f"[✓] TFLite INT8 Exported: {tflite_int8} ({int8_size:.2f} MB)")
    except Exception as e:
        print(f"[!] INT8 export note: {e}")
        
    # 4. Summary & Compression Ratio
    pt_size = os.path.getsize(weights_path) / (1024**2)
    export_manifest["formats"]["pytorch_pt"] = {
        "path": weights_path,
        "size_mb": round(pt_size, 2)
    }
    
    if "tflite_int8" in export_manifest["formats"]:
        int8_sz = export_manifest["formats"]["tflite_int8"]["size_mb"]
        export_manifest["compression_ratio_pt_to_int8"] = f"{round(pt_size / int8_sz, 2)}x"
        export_manifest["storage_reduction_percent"] = f"{round((1 - int8_sz / pt_size) * 100, 1)}%"
        
    summary_path = os.path.join(output_dir, "export_manifest.json")
    with open(summary_path, "w") as f:
        json.dump(export_manifest, f, indent=4)
        
    print(f"\n==========================================")
    print(f" EXPORT SUMMARY")
    print(f" PyTorch FP32 : {pt_size:.2f} MB")
    print(f" ONNX Model   : {onnx_size:.2f} MB")
    if "tflite_standard" in export_manifest["formats"]:
        print(f" TFLite Model : {export_manifest['formats']['tflite_standard']['size_mb']:.2f} MB")
    if "tflite_int8" in export_manifest["formats"]:
        print(f" TFLite INT8  : {export_manifest['formats']['tflite_int8']['size_mb']:.2f} MB ({export_manifest['compression_ratio_pt_to_int8']} compression)")
    print(f" Manifest     : {summary_path}")
    print(f"==========================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export YOLO11 for mobile Android deployment.")
    parser.add_argument("--weights", required=True, help="Path to best.pt")
    parser.add_argument("--data", default="data.yaml", help="Path to data.yaml")
    parser.add_argument("--imgsz", type=int, default=640, help="Input resolution")
    parser.add_argument("--out-dir", default="experiments/exported_models", help="Output directory")
    args = parser.parse_args()
    
    export_for_mobile(args.weights, args.data, args.imgsz, args.out_dir)
