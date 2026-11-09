# 02 — Convert and run live

Choose a conversion target supported by the actual device:
- ONNX Runtime CPU as a portable baseline.
- TensorRT only on supported NVIDIA hardware.
- TFLite INT8 only when the model and target runtime support the conversion path and calibration requirements.

Do not describe an ONNX Runtime CPU run as accelerator inference. Record provider/device explicitly.
