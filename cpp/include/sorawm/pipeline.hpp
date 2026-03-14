#pragma once

#include <functional>
#include <string>
#include <vector>

#include "types.hpp"

namespace sorawm {

/// Configuration for the watermark removal pipeline.
struct PipelineConfig {
    std::string detector_model;   ///< Path to YOLO ONNX model.
    std::string cleaner_model;    ///< Path to LAMA ONNX model.
    int detect_batch_size = 4;    ///< Batch size for watermark detection.
    bool quiet = false;           ///< Suppress progress output.
};

/// Main processing pipeline: detect → impute → clean → encode.
class Pipeline {
public:
    explicit Pipeline(const PipelineConfig& cfg);
    ~Pipeline();

    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;

    /// Process a single video.
    void run(const std::string& input_path,
             const std::string& output_path,
             ProgressCallback progress = nullptr) const;

    /// Process all videos in a directory.
    void run_batch(const std::string& input_dir,
                   const std::string& output_dir,
                   ProgressCallback progress = nullptr) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace sorawm
