"""Load an ONNX model and print runtime/provider information."""
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    model = Path(args.model)
    if not model.is_file():
        raise SystemExit(f"Model not found: {model}")
    try:
        import onnxruntime as ort
    except ImportError:
        raise SystemExit("Install onnxruntime for CPU, or the supported accelerator package.")
    print("ONNX Runtime:", ort.__version__)
    print("Available providers:", ort.get_available_providers())
    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    print("Active providers:", session.get_providers())
    print("Inputs:")
    for item in session.get_inputs():
        print(item.name, item.shape, item.type)
    print("Outputs:")
    for item in session.get_outputs():
        print(item.name, item.shape, item.type)
    print("Session initialized. Add model-specific preprocessing/postprocessing for live camera inference.")

if __name__ == "__main__":
    main()
