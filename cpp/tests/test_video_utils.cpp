/// Unit tests for sorawm video utility functions (merge_frames_with_overlap).

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include <opencv2/core.hpp>

#include "sorawm/video_loader.hpp"

using namespace sorawm;

// ---- Helpers ------------------------------------------------------------
#define ASSERT_EQ(a, b) \
    do { if ((a) != (b)) { \
        std::cerr << __FILE__ << ":" << __LINE__ \
                  << "  ASSERT_EQ failed: " << (a) << " != " << (b) << "\n"; \
        return false; \
    }} while(0)

#define ASSERT_TRUE(x) \
    do { if (!(x)) { \
        std::cerr << __FILE__ << ":" << __LINE__ << "  ASSERT_TRUE failed\n"; \
        return false; \
    }} while(0)

// ---- Tests --------------------------------------------------------------

static bool test_merge_first_chunk() {
    cv::Mat f1(10, 10, CV_8UC3, cv::Scalar(100, 100, 100));
    cv::Mat f2(10, 10, CV_8UC3, cv::Scalar(200, 200, 200));

    std::vector<cv::Mat> result;
    merge_frames_with_overlap(result, {f1, f2}, 0, 0, true);

    ASSERT_EQ(static_cast<int>(result.size()), 2);
    ASSERT_EQ(result[0].at<cv::Vec3b>(0, 0)[0], 100);
    ASSERT_EQ(result[1].at<cv::Vec3b>(0, 0)[0], 200);
    return true;
}

static bool test_merge_no_overlap() {
    cv::Mat f1(10, 10, CV_8UC3, cv::Scalar(100, 100, 100));
    cv::Mat f2(10, 10, CV_8UC3, cv::Scalar(200, 200, 200));
    cv::Mat f3(10, 10, CV_8UC3, cv::Scalar(50, 50, 50));

    std::vector<cv::Mat> result;
    merge_frames_with_overlap(result, {f1, f2}, 0, 0, true);
    merge_frames_with_overlap(result, {f3}, 2, 0, false);

    ASSERT_EQ(static_cast<int>(result.size()), 3);
    ASSERT_EQ(result[0].at<cv::Vec3b>(0, 0)[0], 100);
    ASSERT_EQ(result[1].at<cv::Vec3b>(0, 0)[0], 200);
    ASSERT_EQ(result[2].at<cv::Vec3b>(0, 0)[0], 50);
    return true;
}

static bool test_merge_with_overlap() {
    // Chunk 1: frames [0,1].  Chunk 2 starts at index 1 with overlap 1.
    cv::Mat f1(10, 10, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat f2(10, 10, CV_8UC3, cv::Scalar(100, 100, 100));
    cv::Mat f3(10, 10, CV_8UC3, cv::Scalar(200, 200, 200));
    cv::Mat f4(10, 10, CV_8UC3, cv::Scalar(255, 255, 255));

    std::vector<cv::Mat> result;
    merge_frames_with_overlap(result, {f1, f2}, 0, 0, true);
    merge_frames_with_overlap(result, {f3, f4}, 1, 1, false);

    ASSERT_EQ(static_cast<int>(result.size()), 3);
    // Index 0: unchanged.
    ASSERT_EQ(result[0].at<cv::Vec3b>(0, 0)[0], 0);
    // Index 2: straight copy.
    ASSERT_EQ(result[2].at<cv::Vec3b>(0, 0)[0], 255);
    // Index 1: overlap_end=1, only frame (i=0), alpha=1.0 → 100% new = 200.
    ASSERT_EQ(result[1].at<cv::Vec3b>(0, 0)[0], 200);
    return true;
}

static bool test_merge_extends_result() {
    cv::Mat f1(5, 5, CV_8UC3, cv::Scalar(10, 10, 10));

    std::vector<cv::Mat> result;
    merge_frames_with_overlap(result, {f1}, 0, 0, true);
    ASSERT_EQ(static_cast<int>(result.size()), 1);

    // Add at index 5 → result must grow.
    cv::Mat f2(5, 5, CV_8UC3, cv::Scalar(20, 20, 20));
    merge_frames_with_overlap(result, {f2}, 5, 0, false);
    ASSERT_EQ(static_cast<int>(result.size()), 6);
    ASSERT_EQ(result[5].at<cv::Vec3b>(0, 0)[0], 20);
    return true;
}

// ---- Runner -------------------------------------------------------------

int main() {
    struct Test { const char* name; bool(*fn)(); };
    Test tests[] = {
        {"merge_first_chunk",     test_merge_first_chunk},
        {"merge_no_overlap",      test_merge_no_overlap},
        {"merge_with_overlap",    test_merge_with_overlap},
        {"merge_extends_result",  test_merge_extends_result},
    };

    int passed = 0, failed = 0;
    for (auto& t : tests) {
        std::cout << "  " << t.name << " ... ";
        if (t.fn()) {
            std::cout << "PASSED\n";
            ++passed;
        } else {
            std::cout << "FAILED\n";
            ++failed;
        }
    }

    std::cout << "\n" << passed << " passed, " << failed << " failed.\n";
    return failed > 0 ? 1 : 0;
}
