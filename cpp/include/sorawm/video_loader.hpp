#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

namespace sorawm {

/// Loads video frames using OpenCV and provides metadata.
class VideoLoader {
public:
    explicit VideoLoader(const std::string& path);

    int width() const { return width_; }
    int height() const { return height_; }
    double fps() const { return fps_; }
    int total_frames() const { return total_frames_; }
    int64_t bitrate() const { return bitrate_; }

    /// Read all frames into memory.
    std::vector<cv::Mat> read_all() const;

    /// Read a slice of frames [start, end).
    std::vector<cv::Mat> get_slice(int start, int end) const;

    /// Iterate one frame at a time using a callback.
    /// Callback returns false to stop iteration.
    void for_each_frame(const std::function<bool(int idx, const cv::Mat&)>& fn) const;

private:
    std::string path_;
    int width_ = 0;
    int height_ = 0;
    double fps_ = 0.0;
    int total_frames_ = 0;
    int64_t bitrate_ = 0;
};

/// Merge chunk_frames into result_frames with alpha-blended overlap.
void merge_frames_with_overlap(
    std::vector<cv::Mat>& result_frames,
    const std::vector<cv::Mat>& chunk_frames,
    int start_idx,
    int overlap_size,
    bool is_first_chunk);

} // namespace sorawm
