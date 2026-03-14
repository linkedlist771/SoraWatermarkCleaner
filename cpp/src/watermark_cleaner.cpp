#include "sorawm/watermark_cleaner.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace sorawm {

LamaCleaner::LamaCleaner(const std::string& model_path) {
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "LamaCleaner");
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(4);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    try {
        OrtCUDAProviderOptions cuda_opts{};
        opts.AppendExecutionProvider_CUDA(cuda_opts);
    } catch (...) {}

    session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), opts);
}

LamaCleaner::~LamaCleaner() = default;

cv::Mat LamaCleaner::clean(const cv::Mat& image, const cv::Mat& mask) const {
    const int h = image.rows;
    const int w = image.cols;

    // --- Prepare image tensor (1, 3, H, W) normalised to [0, 1] ---------
    cv::Mat rgb;
    cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

    std::vector<float> img_tensor(3 * h * w);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            for (int c = 0; c < 3; ++c)
                img_tensor[c * h * w + y * w + x] = rgb.at<cv::Vec3f>(y, x)[c];

    // --- Prepare mask tensor (1, 1, H, W) normalised to [0, 1] ----------
    cv::Mat mf;
    mask.convertTo(mf, CV_32F, 1.0 / 255.0);
    std::vector<float> mask_tensor(h * w);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            mask_tensor[y * w + x] = mf.at<float>(y, x);

    // --- Run inference ---------------------------------------------------
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);

    std::array<int64_t, 4> img_shape  = {1, 3, h, w};
    std::array<int64_t, 4> mask_shape = {1, 1, h, w};

    Ort::Value img_val = Ort::Value::CreateTensor<float>(
        mem, img_tensor.data(), img_tensor.size(),
        img_shape.data(), img_shape.size());

    Ort::Value mask_val = Ort::Value::CreateTensor<float>(
        mem, mask_tensor.data(), mask_tensor.size(),
        mask_shape.data(), mask_shape.size());

    Ort::AllocatorWithDefaultOptions alloc;
    auto in0 = session_->GetInputNameAllocated(0, alloc);
    auto in1 = session_->GetInputNameAllocated(1, alloc);
    auto out0 = session_->GetOutputNameAllocated(0, alloc);

    const char* input_names[]  = {in0.get(), in1.get()};
    const char* output_names[] = {out0.get()};

    std::array<Ort::Value, 2> inputs = {std::move(img_val), std::move(mask_val)};

    auto outputs = session_->Run(
        Ort::RunOptions{nullptr},
        input_names, inputs.data(), 2,
        output_names, 1);

    // --- Convert output tensor back to cv::Mat ---------------------------
    const float* out_data = outputs[0].GetTensorData<float>();

    cv::Mat result(h, w, CV_8UC3);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            for (int c = 0; c < 3; ++c) {
                float v = out_data[c * h * w + y * w + x] * 255.0f;
                v = std::clamp(v, 0.0f, 255.0f);
                // Output is RGB; OpenCV expects BGR.
                result.at<cv::Vec3b>(y, x)[2 - c] = static_cast<uint8_t>(v);
            }

    return result;
}

} // namespace sorawm
