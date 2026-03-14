#pragma once

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <onnxruntime_cxx_api.h>

#include "types.hpp"

namespace sorawm {

/// LAMA-based inpainting cleaner using ONNX Runtime.
class LamaCleaner {
public:
    /// Construct with path to a LAMA ONNX model.
    explicit LamaCleaner(const std::string& model_path);
    ~LamaCleaner();

    LamaCleaner(const LamaCleaner&) = delete;
    LamaCleaner& operator=(const LamaCleaner&) = delete;

    /// Inpaint the masked region.
    /// @param image  BGR input image (H x W x 3, uint8).
    /// @param mask   Single-channel mask (H x W, uint8, 255 = inpaint region).
    /// @return       Cleaned BGR image (H x W x 3, uint8).
    cv::Mat clean(const cv::Mat& image, const cv::Mat& mask) const;

private:
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
};

} // namespace sorawm
