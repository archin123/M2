# 05 — CPU vs accelerator + software stack

Only compare accelerators that are actually present and supported by the device. If the Raspberry Pi run uses CPUExecutionProvider, label it CPU. Do not invent GPU/NPU utilization.

Document the complete path:
Camera → frame preprocessing → runtime/session → CPU or accelerator provider → model inference → postprocessing/NMS → decision/output → dashboard/logging.
