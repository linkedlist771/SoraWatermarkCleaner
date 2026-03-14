#include "sorawm/imputation.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>

namespace sorawm {

// ---------------------------------------------------------------------------
// find_2d_data_bkps – lightweight change-point detection
// ---------------------------------------------------------------------------
// The Python version uses the ruptures KernelCPD algorithm.  Here we use a
// simple sliding-window variance approach which gives comparable results for
// the watermark-position use-case without external dependencies.

std::vector<int> find_2d_data_bkps(
    const std::vector<std::optional<std::pair<int, int>>>& centers,
    int min_segment_length,
    double threshold_factor) {

    const int n = static_cast<int>(centers.size());
    if (n < 2 * min_segment_length) return {};

    // Interpolate missing values (forward-fill then backward-fill).
    std::vector<std::pair<double, double>> filled(n);
    {
        std::pair<double, double> last{0.0, 0.0};
        bool have_last = false;
        for (int i = 0; i < n; ++i) {
            if (centers[i].has_value()) {
                filled[i] = {static_cast<double>(centers[i]->first),
                             static_cast<double>(centers[i]->second)};
                last = filled[i];
                have_last = true;
            } else if (have_last) {
                filled[i] = last;
            }
        }
        // Backward-fill remaining.
        have_last = false;
        for (int i = n - 1; i >= 0; --i) {
            if (centers[i].has_value()) {
                last = {static_cast<double>(centers[i]->first),
                        static_cast<double>(centers[i]->second)};
                have_last = true;
            } else if (have_last) {
                filled[i] = last;
            }
        }
    }

    // Compute global mean and variance of the trajectory.
    double mean_x = 0, mean_y = 0;
    for (auto& [x, y] : filled) { mean_x += x; mean_y += y; }
    mean_x /= n;  mean_y /= n;

    double var = 0;
    for (auto& [x, y] : filled) {
        double dx = x - mean_x;
        double dy = y - mean_y;
        var += dx * dx + dy * dy;
    }
    var /= n;
    const double threshold = threshold_factor * std::sqrt(var + 1e-8);

    // Sliding-window: detect points where a jump exceeds the threshold.
    std::vector<int> bkps;
    const int half = min_segment_length / 2;
    for (int i = half; i < n - half; ++i) {
        // Mean of left window.
        double lx = 0, ly = 0;
        for (int j = i - half; j < i; ++j) { lx += filled[j].first; ly += filled[j].second; }
        lx /= half;  ly /= half;
        // Mean of right window.
        double rx = 0, ry = 0;
        for (int j = i; j < i + half; ++j) { rx += filled[j].first; ry += filled[j].second; }
        rx /= half;  ry /= half;

        double dist = std::sqrt((lx - rx) * (lx - rx) + (ly - ry) * (ly - ry));
        if (dist > threshold) {
            bkps.push_back(i);
            // Skip ahead by half a window to avoid redundant breakpoints
            // within the same jump.  The loop increment adds 1, so the
            // next iteration starts at i + half.
            i += half - 1;
        }
    }
    return bkps;
}

// ---------------------------------------------------------------------------
// get_interval_average_bbox
// ---------------------------------------------------------------------------
std::vector<std::optional<BBox>> get_interval_average_bbox(
    const std::vector<std::optional<BBox>>& bboxes,
    const std::vector<int>& bkps) {

    std::vector<std::optional<BBox>> result;
    for (size_t k = 0; k + 1 < bkps.size(); ++k) {
        int left = bkps[k];
        int right = bkps[k + 1];
        double sx1 = 0, sy1 = 0, sx2 = 0, sy2 = 0;
        int cnt = 0;
        for (int i = left; i < right && i < static_cast<int>(bboxes.size()); ++i) {
            if (bboxes[i].has_value()) {
                sx1 += bboxes[i]->x1;
                sy1 += bboxes[i]->y1;
                sx2 += bboxes[i]->x2;
                sy2 += bboxes[i]->y2;
                ++cnt;
            }
        }
        if (cnt > 0) {
            BBox avg;
            avg.x1 = static_cast<int>(sx1 / cnt);
            avg.y1 = static_cast<int>(sy1 / cnt);
            avg.x2 = static_cast<int>(sx2 / cnt);
            avg.y2 = static_cast<int>(sy2 / cnt);
            result.push_back(avg);
        } else {
            result.push_back(std::nullopt);
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// find_idxs_interval – binary search mapping indices to intervals
// ---------------------------------------------------------------------------
std::vector<int> find_idxs_interval(
    const std::vector<int>& idxs,
    const std::vector<int>& bkps) {

    auto find_one = [&](int idx) -> int {
        int lo = 0;
        int hi = static_cast<int>(bkps.size()) - 2;
        while (lo <= hi) {
            int mid = (lo + hi) / 2;
            if (bkps[mid] <= idx && idx < bkps[mid + 1]) return mid;
            if (idx < bkps[mid]) hi = mid - 1;
            else lo = mid + 1;
        }
        return std::min(std::max(lo, 0), static_cast<int>(bkps.size()) - 2);
    };

    std::vector<int> result;
    result.reserve(idxs.size());
    for (int idx : idxs) {
        result.push_back(find_one(idx));
    }
    return result;
}

// ---------------------------------------------------------------------------
// refine_bkps_by_chunk_size
// ---------------------------------------------------------------------------
std::vector<int> refine_bkps_by_chunk_size(
    const std::vector<int>& bkps,
    int chunk_size) {

    if (bkps.empty() || chunk_size <= 0) return bkps;

    std::set<int> pts;
    pts.insert(bkps.front());
    for (size_t k = 0; k + 1 < bkps.size(); ++k) {
        for (int v = bkps[k]; v < bkps[k + 1]; v += chunk_size) {
            pts.insert(v);
        }
        pts.insert(bkps[k + 1]);
    }
    return {pts.begin(), pts.end()};
}

} // namespace sorawm
