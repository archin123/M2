"""Basic ONNX Runtime inference benchmark; adapt input shape/dtype to your model."""
import argparse, time, statistics
import numpy as np
import onnxruntime as ort

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--runs", type=int, default=100)
    p.add_argument("--height", type=int, default=320)
    p.add_argument("--width", type=int, default=320)
    p.add_argument("--channels", type=int, default=3)
    p.add_argument("--layout", choices=["NCHW", "NHWC"], default="NCHW")
    args = p.parse_args()

    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    inp = session.get_inputs()[0]
    shape = ((1, args.channels, args.height, args.width) if args.layout == "NCHW"
             else (1, args.height, args.width, args.channels))
    dtype = np.float32 if "float" in inp.type else np.uint8
    sample = np.zeros(shape, dtype=dtype)
    feed = {inp.name: sample}

    for _ in range(args.warmup):
        session.run(None, feed)

    times = []
    for _ in range(args.runs):
        start = time.perf_counter()
        session.run(None, feed)
        times.append((time.perf_counter() - start) * 1000)

    ordered = sorted(times)
    p95 = ordered[min(len(ordered)-1, int(0.95 * len(ordered)))]
    median = statistics.median(times)
    print(f"Provider: {session.get_providers()}")
    print(f"Runs: {len(times)} | Batch: 1 | Warm-up: {args.warmup}")
    print(f"Median latency: {median:.3f} ms")
    print(f"P95 latency: {p95:.3f} ms")
    print(f"Inference-only FPS estimate: {1000/median:.2f}")
    print("NOTE: Zero-input synthetic benchmark; replace with representative inputs and document preprocessing/postprocessing separately.")

if __name__ == "__main__":
    main()
