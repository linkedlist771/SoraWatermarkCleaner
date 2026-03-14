#include "sorawm/video_loader.hpp"

#include <functional>
#include <iostream>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

namespace sorawm {

VideoLoader::VideoLoader(const std::string& path) : path_(path) {
    cv::VideoCapture cap(path);
    if (!cap.isOpened()) {
        throw std::runtime_error("Cannot open video: " + path);
    }
    width_ = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    height_ = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    fps_ = cap.get(cv::CAP_PROP_FPS);
    total_frames_ = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    bitrate_ = static_cast<int64_t>(cap.get(cv::CAP_PROP_BITRATE));
}

std::vector<cv::Mat> VideoLoader::read_all() const {
    return get_slice(0, total_frames_);
}

std::vector<cv::Mat> VideoLoader::get_slice(int start, int end) const {
    cv::VideoCapture cap(path_);
    if (!cap.isOpened()) {
        throw std::runtime_error("Cannot open video: " + path_);
    }

    std::vector<cv::Mat> frames;
    frames.reserve(end - start);

    if (start > 0) {
        cap.set(cv::CAP_PROP_POS_FRAMES, start);
    }

    for (int i = start; i < end; ++i) {
        cv::Mat frame;
        if (!cap.read(frame)) break;
        frames.push_back(frame.clone());
    }
    return frames;
}

void VideoLoader::for_each_frame(
    const std::function<bool(int idx, const cv::Mat&)>& fn) const {
    cv::VideoCapture cap(path_);
    if (!cap.isOpened()) {
        throw std::runtime_error("Cannot open video: " + path_);
    }

    cv::Mat frame;
    for (int idx = 0; cap.read(frame); ++idx) {
        if (!fn(idx, frame)) break;
    }
}

// ---------------------------------------------------------------------------
// merge_frames_with_overlap
// ---------------------------------------------------------------------------
void merge_frames_with_overlap(
    std::vector<cv::Mat>& result_frames,
    const std::vector<cv::Mat>& chunk_frames,
    int start_idx,
    int overlap_size,
    bool is_first_chunk) {

    const int chunk_size = static_cast<int>(chunk_frames.size());
    const int required = start_idx + chunk_size;

    // First chunk: just copy everything in.
    if (is_first_chunk || result_frames.empty()) {
        result_frames.resize(required);
        for (int i = 0; i < chunk_size; ++i) {
            result_frames[start_idx + i] = chunk_frames[i].clone();
        }
        return;
    }

    // Grow if needed.
    if (static_cast<int>(result_frames.size()) < required) {
        result_frames.resize(required);
    }

    // Blend the overlap region with linear interpolation.
    // alpha ramps from 0.0 (100 % old) to 1.0 (100 % new) across the
    // overlap window, matching the original Python behaviour.
    const int overlap_end = std::min(overlap_size, chunk_size);
    for (int i = 0; i < overlap_end; ++i) {
        const int ri = start_idx + i;
        if (!result_frames[ri].empty() && !chunk_frames[i].empty()) {
            const double alpha = (overlap_end > 1)
                ? static_cast<double>(i) / (overlap_end - 1)
                : 1.0;
            cv::addWeighted(result_frames[ri], 1.0 - alpha,
                            chunk_frames[i], alpha, 0.0,
                            result_frames[ri]);
        } else if (!chunk_frames[i].empty()) {
            result_frames[ri] = chunk_frames[i].clone();
        }
    }

    // Copy the rest.
    for (int i = overlap_end; i < chunk_size; ++i) {
        result_frames[start_idx + i] = chunk_frames[i].clone();
    }
}

} // namespace sorawm
