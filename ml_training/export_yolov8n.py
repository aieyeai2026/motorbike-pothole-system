"""
Exports a trained YOLOv8-Nano pothole detection checkpoint (.pt)
to ONNX (for Rust `ort` runtime) and INT8 TFLite (for mobile NPU/NNAPI).

Usage:
    pip install ultralytics
    python ml_training/export_yolov8n.py --weights best.pt --imgsz 640
"""
import argparse
from pathlib import Path


def export_model(weights_path: str, imgsz: int = 640) -> None:
    from ultralytics import YOLO  # Imported lazily so CLI help works without torch installed

    weights = Path(weights_path)
    if not weights.exists():
        raise FileNotFoundError(f"Checkpoint not found: {weights}")

    model = YOLO(str(weights))
    onnx_path = model.export(format="onnx", imgsz=imgsz, opset=12, simplify=True)
    tflite_path = model.export(format="tflite", imgsz=imgsz, int8=True)
    print(f"Exported ONNX:   {onnx_path}")
    print(f"Exported TFLite: {tflite_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export YOLOv8n pothole model for Edge Rust/Mobile")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="Path to trained .pt weights")
    parser.add_argument("--imgsz", type=int, default=640, help="Square input size (default: 640)")
    args = parser.parse_args()
    export_model(args.weights, args.imgsz)
