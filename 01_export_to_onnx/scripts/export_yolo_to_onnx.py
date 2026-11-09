"""Example ONNX export for an Ultralytics YOLO .pt model."""
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Path to trained .pt weights")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--dynamic", action="store_true")
    args = parser.parse_args()

    weights = Path(args.weights)
    if not weights.is_file():
        raise SystemExit(f"Weights file not found: {weights}")

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("Install ultralytics first: pip install ultralytics")

    model = YOLO(str(weights))
    result = model.export(format="onnx", imgsz=args.imgsz,
                          opset=args.opset, dynamic=args.dynamic)
    print("Export result:", result)
    print("Verify the exported ONNX model and record its actual size and tensor shapes.")

if __name__ == "__main__":
    main()
