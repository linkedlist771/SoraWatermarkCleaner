#include "sorawm/pipeline.hpp"

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <optional>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "sorawm/imputation.hpp"
#include "sorawm/video_loader.hpp"
#include "sorawm/watermark_cleaner.hpp"
#include "sorawm/watermark_detector.hpp"

namespace fs = std::filesystem;

namespace sorawm {

static const std::vector<std::string> kVideoExts =
    {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"};

// --------------------------------------------------------------------------
// Pipeline::Impl
// --------------------------------------------------------------------------
struct Pipeline::Impl {
    PipelineConfig cfg;
    WatermarkDetector detector;
    LamaCleaner cleaner;

    Impl(const PipelineConfig& c)
        : cfg(c), detector(c.detector_model), cleaner(c.cleaner_model) {}
};

Pipeline::Pipeline(const PipelineConfig& cfg)
    : impl_(std::make_unique<Impl>(cfg)) {}

Pipeline::~Pipeline() = default;

// --------------------------------------------------------------------------
// run – single video
// --------------------------------------------------------------------------
void Pipeline::run(const std::string& input_path,
                   const std::string& output_path,
                   ProgressCallback progress) const {

    VideoLoader loader(input_path);
    const int W = loader.width();
    const int H = loader.height();
    const int total = loader.total_frames();
    const double fps = loader.fps();

    if (!impl_->cfg.quiet) {
        std::cout << "Processing: " << input_path
                  << " (" << W << "x" << H << ", "
                  << total << " frames, " << fps << " fps)\n";
    }

    // ---- 1. Detection phase (10–50 %) -----------------------------------
    std::vector<std::optional<BBox>> frame_bboxes(total, std::nullopt);
    std::vector<std::optional<std::pair<int, int>>> centers(total, std::nullopt);
    std::vector<int> missed;

    std::vector<cv::Mat> batch;
    std::vector<int> batch_ids;

    auto flush_batch = [&]() {
        if (batch.empty()) return;
        auto results = impl_->detector.detect_batch(batch);
        for (size_t k = 0; k < results.size(); ++k) {
            int fi = batch_ids[k];
            if (results[k].detected) {
                frame_bboxes[fi] = results[k].bbox;
                auto& b = results[k].bbox;
                centers[fi] = {b.center_x(), b.center_y()};
            } else {
                missed.push_back(fi);
            }
            if (progress && fi % 10 == 0) {
                progress(10 + static_cast<int>(40.0 * fi / total));
            }
        }
        batch.clear();
        batch_ids.clear();
    };

    loader.for_each_frame([&](int idx, const cv::Mat& frame) {
        batch.push_back(frame.clone());
        batch_ids.push_back(idx);
        if (static_cast<int>(batch.size()) >= impl_->cfg.detect_batch_size) {
            flush_batch();
        }
        return true;
    });
    flush_batch();

    // ---- 2. Imputation of missed detections -----------------------------
    if (!missed.empty()) {
        auto bkps_raw = find_2d_data_bkps(centers);
        std::vector<int> bkps_full;
        bkps_full.push_back(0);
        for (int b : bkps_raw) bkps_full.push_back(b);
        bkps_full.push_back(total);

        // Convert frame_bboxes to the format expected by get_interval_average_bbox.
        auto avg_bboxes = get_interval_average_bbox(frame_bboxes, bkps_full);
        auto intervals  = find_idxs_interval(missed, bkps_full);

        for (size_t k = 0; k < missed.size(); ++k) {
            int mi = missed[k];
            int ii = intervals[k];
            if (ii < static_cast<int>(avg_bboxes.size()) && avg_bboxes[ii].has_value()) {
                frame_bboxes[mi] = avg_bboxes[ii];
            } else {
                // Fallback: use nearest neighbour.
                int before = std::max(mi - 1, 0);
                int after  = std::min(mi + 1, total - 1);
                if (frame_bboxes[before].has_value())
                    frame_bboxes[mi] = frame_bboxes[before];
                else if (frame_bboxes[after].has_value())
                    frame_bboxes[mi] = frame_bboxes[after];
            }
        }
    }

    if (progress) progress(50);

    // ---- 3. Cleaning phase (50–95 %) ------------------------------------
    // Create output directory.
    fs::path out(output_path);
    fs::create_directories(out.parent_path());

    // Open video writer (H.264).
    cv::VideoWriter writer(
        output_path,
        cv::VideoWriter::fourcc('a', 'v', 'c', '1'),
        fps,
        cv::Size(W, H));

    if (!writer.isOpened()) {
        // Fallback to MJPG if H.264 is not available.
        writer.open(output_path,
                    cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                    fps, cv::Size(W, H));
    }
    if (!writer.isOpened()) {
        throw std::runtime_error("Cannot open video writer for: " + output_path);
    }

    int written = 0;
    loader.for_each_frame([&](int idx, const cv::Mat& frame) {
        cv::Mat out_frame;
        if (frame_bboxes[idx].has_value()) {
            auto& bb = *frame_bboxes[idx];
            cv::Mat mask = cv::Mat::zeros(H, W, CV_8UC1);
            cv::rectangle(mask,
                          cv::Point(bb.x1, bb.y1),
                          cv::Point(bb.x2, bb.y2),
                          cv::Scalar(255), cv::FILLED);
            out_frame = impl_->cleaner.clean(frame, mask);
        } else {
            out_frame = frame;
        }
        writer.write(out_frame);
        ++written;

        if (progress && written % 10 == 0) {
            progress(50 + static_cast<int>(45.0 * written / total));
        }
        return true;
    });

    writer.release();

    if (progress) progress(99);

    if (!impl_->cfg.quiet) {
        std::cout << "Saved: " << output_path
                  << " (" << written << " frames)\n";
    }
}

// --------------------------------------------------------------------------
// run_batch
// --------------------------------------------------------------------------
void Pipeline::run_batch(const std::string& input_dir,
                         const std::string& output_dir,
                         ProgressCallback progress) const {

    fs::create_directories(output_dir);

    std::vector<fs::path> videos;
    for (auto& entry : fs::recursive_directory_iterator(input_dir)) {
        if (!entry.is_regular_file()) continue;
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (std::find(kVideoExts.begin(), kVideoExts.end(), ext) != kVideoExts.end()) {
            videos.push_back(entry.path());
        }
    }
    std::sort(videos.begin(), videos.end());

    if (!impl_->cfg.quiet) {
        std::cout << "Found " << videos.size() << " video(s) to process.\n";
    }

    for (size_t i = 0; i < videos.size(); ++i) {
        fs::path out = fs::path(output_dir) / videos[i].filename();
        auto per_video = [&](int pct) {
            if (progress) {
                int overall = static_cast<int>(
                    (static_cast<double>(i) / videos.size()) * 100
                    + static_cast<double>(pct) / videos.size());
                progress(std::min(overall, 100));
            }
        };
        run(videos[i].string(), out.string(), per_video);
    }
}

} // namespace sorawm
