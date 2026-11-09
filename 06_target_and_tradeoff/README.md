# 06 — Own target and trade-off

Write the target **before** running optimization. Keep the original timestamped target record in version control; do not rewrite it after seeing results.

Example target categories (set your own numeric values after establishing a baseline):
- Minimum acceptable mAP@0.5.
- Maximum median and p95 inference latency.
- Minimum sustained FPS.
- Maximum model size.
- Maximum temperature or power envelope.

After the run, compare target vs actual and explain the trade-off in plain language.
