#pragma once

#include <optional>
#include <vector>

#include "types.hpp"

namespace sorawm {

/// Detect breakpoints in 2D bbox-center trajectory using a simple
/// variance-based sliding-window method (replaces Python ruptures dependency).
std::vector<int> find_2d_data_bkps(
    const std::vector<std::optional<std::pair<int, int>>>& centers,
    int min_segment_length = 10,
    double threshold_factor = 1.5);

/// Compute the average bbox within each interval defined by consecutive
/// breakpoints.  bkps must include 0 and total_frames as boundaries.
std::vector<std::optional<BBox>> get_interval_average_bbox(
    const std::vector<std::optional<BBox>>& bboxes,
    const std::vector<int>& bkps);

/// Map each index to its interval [bkps[i], bkps[i+1]).
std::vector<int> find_idxs_interval(
    const std::vector<int>& idxs,
    const std::vector<int>& bkps);

/// Sub-divide large intervals so every segment <= chunk_size.
std::vector<int> refine_bkps_by_chunk_size(
    const std::vector<int>& bkps,
    int chunk_size);

} // namespace sorawm
