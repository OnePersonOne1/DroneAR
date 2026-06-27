# Official Ultralytics GPU image: bundles CUDA + PyTorch + ultralytics.
# We do NOT build CUDA/torch ourselves — reproducibility comes from pinning this base.
FROM ultralytics/ultralytics:latest

WORKDIR /workspace

# Export / quantization toolchain (ONNX path for Magic Leap 2).
RUN pip install --no-cache-dir onnxruntime onnxslim onnxconverter-common

# Project code (datasets + weights + runs are mounted at runtime, not baked in).
COPY scripts/ ./scripts/
COPY configs/ ./configs/
