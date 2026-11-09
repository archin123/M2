# M2 — Model Optimization

**Milestone:** M2 — Optimize  
**Package type:** Demo structure, scripts and reporting templates. Not proof of completed optimization.

This repository is organized around the six M2 assessment criteria:

1. Export to ONNX
2. Convert and run live
3. Accuracy check
4. Fair benchmark
5. CPU vs accelerator + software stack
6. Own target and trade-off

## Important status note
All result tables start as placeholders. Do not claim export, conversion, accelerator execution, accuracy retention, or benchmark results until you have run them on the target device and saved the logs.

## Suggested workflow
1. Identify the exact trained model and class names from M1.
2. Export ONNX and validate the exported model.
3. Select a runtime supported by the target hardware.
4. Run the same test set through the original and optimized models.
5. Benchmark with batch size 1, 20 warm-ups and 100+ timed runs.
6. Compare results against a target written before optimization.

See each numbered folder for its own README and starter assets.
