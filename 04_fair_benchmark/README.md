# 04 — Fair benchmark

Protocol required by the assessment:
- Batch size 1.
- At least 20 warm-up iterations.
- At least 100 timed inferences.
- Same model input size and test input across runtimes.
- Report median and p95 latency, FPS and model size.
- Record device, runtime/provider, CPU load, accelerator utilization where available, RAM and temperature.

Use `benchmark_onnx.py` as a basic timing harness after confirming the input tensor shape and dtype. It measures session inference only; include preprocessing and postprocessing separately for end-to-end latency.
