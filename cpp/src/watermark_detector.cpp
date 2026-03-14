#include "sorawm/watermark_detector.hpp"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace sorawm {

WatermarkDetector::WatermarkDetector(const std::string& model_path) {
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "WatermarkDetector");
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(4);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // Try to enable CUDA if available.
    try {
        OrtCUDAProviderOptions cuda_opts{};
        opts.AppendExecutionProvider_CUDA(cuda_opts);
    } catch (...) {
        // CUDA not available – fall back to CPU.
    }

    session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), opts);

    // Query expected input dimensions [1, 3, H, W].
    auto input_shape = session_->GetInputTypeInfo(0)
                           .GetTensorTypeAndShapeInfo()
                           .GetShape();
    if (input_shape.size() == 4 && input_shape[2] > 0 && input_shape[3] > 0) {
        input_h_ = static_cast<int>(input_shape[2]);
        input_w_ = static_cast<int>(input_shape[3]);
    }
}

WatermarkDetector::~WatermarkDetector() = default;

cv::Mat WatermarkDetector::preprocess(
    const cv::Mat& image, float& scale_x, float& scale_y) const {

    cv::Mat rgb;
    cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);
    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(input_w_, input_h_));
    resized.convertTo(resized, CV_32F, 1.0 / 255.0);

    scale_x = static_cast<float>(image.cols) / input_w_;
    scale_y = static_cast<float>(image.rows) / input_h_;
    return resized;
}

DetectionResult WatermarkDetector::postprocess(
    const float* data,
    int num_detections,
    float sx, float sy,
    float conf_threshold) const {

    DetectionResult best;
    best.detected = false;

    // YOLO ONNX output layout per detection (stride = 6):
    //   [0] x_center  [1] y_center  [2] width  [3] height
    //   [4] confidence  [5] class_id (unused – single-class model)
    constexpr int kStride = 6;
    for (int i = 0; i < num_detections; ++i) {
        float cx   = data[i * kStride + 0];
        float cy   = data[i * kStride + 1];
        float w    = data[i * kStride + 2];
        float h    = data[i * kStride + 3];
        float conf = data[i * kStride + 4];

        if (conf < conf_threshold) continue;

        if (conf > best.confidence) {
            best.detected = true;
            best.confidence = conf;
            best.bbox.x1 = static_cast<int>((cx - w / 2) * sx);
            best.bbox.y1 = static_cast<int>((cy - h / 2) * sy);
            best.bbox.x2 = static_cast<int>((cx + w / 2) * sx);
            best.bbox.y2 = static_cast<int>((cy + h / 2) * sy);
        }
    }
    return best;
}

DetectionResult WatermarkDetector::detect(const cv::Mat& image) const {
    float sx, sy;
    cv::Mat preprocessed = preprocess(image, sx, sy);

    // HWC -> CHW, then flatten.
    const int c = 3, h = input_h_, w = input_w_;
    std::vector<float> input_tensor(c * h * w);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            for (int ch = 0; ch < c; ++ch)
                input_tensor[ch * h * w + y * w + x] =
                    preprocessed.at<cv::Vec3f>(y, x)[ch];

    std::array<int64_t, 4> input_shape = {1, c, h, w};
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value input_val = Ort::Value::CreateTensor<float>(
        mem, input_tensor.data(), input_tensor.size(),
        input_shape.data(), input_shape.size());

    Ort::AllocatorWithDefaultOptions alloc;
    auto in_name  = session_->GetInputNameAllocated(0, alloc);
    auto out_name = session_->GetOutputNameAllocated(0, alloc);
    const char* input_names[]  = {in_name.get()};
    const char* output_names[] = {out_name.get()};

    auto outputs = session_->Run(
        Ort::RunOptions{nullptr},
        input_names, &input_val, 1,
        output_names, 1);

    auto& out_tensor = outputs[0];
    auto out_shape = out_tensor.GetTensorTypeAndShapeInfo().GetShape();
    int num_detections = (out_shape.size() >= 2) ? static_cast<int>(out_shape[1]) : 0;
    const float* out_data = out_tensor.GetTensorData<float>();

    return postprocess(out_data, num_detections, sx, sy);
}

std::vector<DetectionResult> WatermarkDetector::detect_batch(
    const std::vector<cv::Mat>& images) const {

    std::vector<DetectionResult> results;
    results.reserve(images.size());
    for (auto& img : images) {
        results.push_back(detect(img));
    }
    return results;
}

} // namespace sorawm
