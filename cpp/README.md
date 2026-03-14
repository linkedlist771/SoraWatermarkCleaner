# SoraWatermarkCleaner – C++ Edition

High-performance C++ implementation of the SoraWatermarkCleaner pipeline.
Replaces the Python core with native C++17 for significantly faster video
processing.

## Architecture

```
cpp/
├── CMakeLists.txt
├── include/sorawm/        # Public headers
│   ├── types.hpp          # BBox, DetectionResult, CleanerType
│   ├── video_loader.hpp   # OpenCV-based video I/O
│   ├── imputation.hpp     # Change-point detection & bbox interpolation
│   ├── watermark_detector.hpp  # YOLO via ONNX Runtime
│   ├── watermark_cleaner.hpp   # LAMA via ONNX Runtime
│   └── pipeline.hpp       # Main processing orchestrator
├── src/                   # Implementation files
│   ├── video_loader.cpp
│   ├── imputation.cpp
│   ├── watermark_detector.cpp
│   ├── watermark_cleaner.cpp
│   ├── pipeline.cpp
│   └── main.cpp           # CLI entry point
└── tests/
    ├── test_imputation.cpp
    └── test_video_utils.cpp
```

## Dependencies

| Library       | Purpose              | Required |
|---------------|----------------------|----------|
| OpenCV ≥ 4.5  | Video I/O, image ops | Yes      |
| ONNX Runtime  | Model inference      | For detection/cleaning |
| C++17 compiler| Build                | Yes      |
| CMake ≥ 3.20  | Build system         | Yes      |

## Building

```bash
# Install dependencies (Ubuntu/Debian)
sudo apt-get install -y libopencv-dev cmake g++

# Optional: Install ONNX Runtime for inference support
# Download from https://github.com/microsoft/onnxruntime/releases

# Build
cd cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Run tests
ctest --output-on-failure
```

## Usage

```bash
# Single video
./sorawm_cli -i input.mp4 -o output.mp4 \
    --detector models/yolo.onnx \
    --cleaner models/lama.onnx

# Batch processing
./sorawm_cli -i /path/to/videos -o /path/to/output --batch

# Quiet mode
./sorawm_cli -i input.mp4 -o output.mp4 --quiet
```

## Model Preparation

Export the Python models to ONNX format:

```python
# Export YOLO model
from ultralytics import YOLO
model = YOLO("resources/best.pt")
model.export(format="onnx", imgsz=640)

# Export LAMA model (use IOPaint's export utilities)
# See: https://github.com/Sanster/IOPaint
```

## Comparison with Python Version

| Aspect          | Python           | C++                |
|-----------------|------------------|--------------------|
| Video I/O       | FFmpeg subprocess| OpenCV VideoCapture|
| Detection       | Ultralytics YOLO | ONNX Runtime       |
| Inpainting      | PyTorch LAMA     | ONNX Runtime       |
| Imputation      | ruptures + sklearn| Pure C++ (zero-dep)|
| Memory          | Python GC        | RAII               |
| Startup time    | ~5–10s (imports) | < 1s               |
| Deployment      | Python + venv    | Single binary      |

## Key Design Decisions

1. **ONNX Runtime** for inference – cross-platform, supports CUDA/TensorRT/
   DirectML with zero code changes.
2. **OpenCV** for video I/O – robust, well-tested, hardware-accelerated.
3. **Pure C++ imputation** – the change-point detection uses a simple
   sliding-window variance method that produces equivalent results to the
   Python `ruptures` library for this use-case.
4. **PIMPL pattern** in Pipeline – keeps the public API header-only clean.
5. **Two static libraries** – `sorawm_core` (no ONNX dep) and `sorawm_infer`
   (needs ONNX Runtime), so users who only need utilities can link just the
   core.
