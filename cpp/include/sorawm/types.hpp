#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace sorawm {

/// Bounding box in (x1, y1, x2, y2) format.
struct BBox {
    int x1 = 0, y1 = 0, x2 = 0, y2 = 0;

    bool empty() const { return x1 == 0 && y1 == 0 && x2 == 0 && y2 == 0; }
    int center_x() const { return (x1 + x2) / 2; }
    int center_y() const { return (y1 + y2) / 2; }
};

/// Result of watermark detection on a single frame.
struct DetectionResult {
    bool detected = false;
    BBox bbox;
    float confidence = 0.0f;
};

/// Cleaner algorithm selection.
enum class CleanerType { LAMA, E2FGVI_HQ };

/// Convert string to CleanerType.
inline CleanerType cleaner_type_from_string(const std::string& s) {
    if (s == "lama") return CleanerType::LAMA;
    if (s == "e2fgvi_hq") return CleanerType::E2FGVI_HQ;
    throw std::invalid_argument("Unknown cleaner type: " + s);
}

/// Progress callback: receives a percentage 0–100.
using ProgressCallback = void (*)(int percent);

} // namespace sorawm
