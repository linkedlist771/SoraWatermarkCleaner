#pragma once

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <onnxruntime_cxx_api.h>

#include "types.hpp"

namespace sorawm {

/// YOLO-based watermark detector using ONNX Runtime for inference.
class WatermarkDetector {
public:
    /// Construct with path to a YOLO ONNX model.
    explicit WatermarkDetector(const std::string& model_path);
    ~WatermarkDetector();

    WatermarkDetector(const WatermarkDetector&) = delete;
    WatermarkDetector& operator=(const WatermarkDetector&) = delete;

    /// Detect watermark in a single BGR image.
    DetectionResult detect(const cv::Mat& image) const;

    /// Batch detection on multiple BGR images.
    std::vector<DetectionResult> detect_batch(
        const std::vector<cv::Mat>& images) const;

private:
    DetectionResult postprocess(
        const float* output_data,
        int num_detections,
        float scale_x,
        float scale_y,
        float conf_threshold = 0.25f) const;

    cv::Mat preprocess(const cv::Mat& image, float& scale_x, float& scale_y) const;

    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    int input_w_ = 640;
    int input_h_ = 640;
};

} // namespace sorawm
