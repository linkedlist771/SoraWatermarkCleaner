/// Unit tests for sorawm::imputation utilities.

#include <cassert>
#include <cmath>
#include <iostream>
#include <optional>
#include <vector>

#include "sorawm/imputation.hpp"
#include "sorawm/types.hpp"

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

static bool test_find_idxs_interval_basic() {
    // Intervals: [0,5), [5,10), [10,15)
    std::vector<int> bkps = {0, 5, 10, 15};
    auto res = find_idxs_interval({0, 4, 5, 9, 10, 14}, bkps);
    ASSERT_EQ(res.size(), 6u);
    ASSERT_EQ(res[0], 0);  // 0 in [0,5)
    ASSERT_EQ(res[1], 0);  // 4 in [0,5)
    ASSERT_EQ(res[2], 1);  // 5 in [5,10)
    ASSERT_EQ(res[3], 1);  // 9 in [5,10)
    ASSERT_EQ(res[4], 2);  // 10 in [10,15)
    ASSERT_EQ(res[5], 2);  // 14 in [10,15)
    return true;
}

static bool test_find_idxs_interval_single() {
    std::vector<int> bkps = {0, 100};
    auto res = find_idxs_interval({50}, bkps);
    ASSERT_EQ(res.size(), 1u);
    ASSERT_EQ(res[0], 0);
    return true;
}

static bool test_get_interval_average_bbox() {
    // Two intervals: [0,3) and [3,5).
    std::vector<int> bkps = {0, 3, 5};
    std::vector<std::optional<BBox>> boxes = {
        BBox{10, 20, 30, 40},
        BBox{12, 22, 32, 42},
        std::nullopt,
        BBox{100, 200, 300, 400},
        BBox{102, 202, 302, 402},
    };
    auto avg = get_interval_average_bbox(boxes, bkps);
    ASSERT_EQ(avg.size(), 2u);
    // First interval: average of (10,20,30,40) and (12,22,32,42) = (11,21,31,41)
    ASSERT_TRUE(avg[0].has_value());
    ASSERT_EQ(avg[0]->x1, 11);
    ASSERT_EQ(avg[0]->y1, 21);
    ASSERT_EQ(avg[0]->x2, 31);
    ASSERT_EQ(avg[0]->y2, 41);
    // Second interval: average of (100,200,300,400) and (102,202,302,402)
    ASSERT_TRUE(avg[1].has_value());
    ASSERT_EQ(avg[1]->x1, 101);
    ASSERT_EQ(avg[1]->y1, 201);
    return true;
}

static bool test_get_interval_average_bbox_all_none() {
    std::vector<int> bkps = {0, 3};
    std::vector<std::optional<BBox>> boxes = {
        std::nullopt, std::nullopt, std::nullopt
    };
    auto avg = get_interval_average_bbox(boxes, bkps);
    ASSERT_EQ(avg.size(), 1u);
    ASSERT_TRUE(!avg[0].has_value());
    return true;
}

static bool test_refine_bkps_basic() {
    std::vector<int> bkps = {0, 20};
    auto refined = refine_bkps_by_chunk_size(bkps, 5);
    // Should produce: 0, 5, 10, 15, 20
    ASSERT_EQ(refined.size(), 5u);
    ASSERT_EQ(refined[0], 0);
    ASSERT_EQ(refined[4], 20);
    // Check that all intervals are <= 5
    for (size_t i = 0; i + 1 < refined.size(); ++i) {
        ASSERT_TRUE(refined[i + 1] - refined[i] <= 5);
    }
    return true;
}

static bool test_refine_bkps_already_small() {
    std::vector<int> bkps = {0, 3, 5};
    auto refined = refine_bkps_by_chunk_size(bkps, 10);
    // No sub-division needed; should still contain 0, 3, 5.
    ASSERT_TRUE(refined.size() >= 3u);
    ASSERT_EQ(refined.front(), 0);
    ASSERT_EQ(refined.back(), 5);
    return true;
}

static bool test_find_2d_data_bkps_empty() {
    std::vector<std::optional<std::pair<int, int>>> centers;
    auto bkps = find_2d_data_bkps(centers);
    ASSERT_TRUE(bkps.empty());
    return true;
}

static bool test_find_2d_data_bkps_constant() {
    // All centers are the same – no breakpoints expected.
    std::vector<std::optional<std::pair<int, int>>> centers(50, std::make_pair(100, 200));
    auto bkps = find_2d_data_bkps(centers);
    ASSERT_TRUE(bkps.empty());
    return true;
}

static bool test_find_2d_data_bkps_with_jump() {
    // First half at (100,100), second half at (500,500) – should detect a breakpoint.
    std::vector<std::optional<std::pair<int, int>>> centers;
    for (int i = 0; i < 30; ++i) centers.push_back(std::make_pair(100, 100));
    for (int i = 0; i < 30; ++i) centers.push_back(std::make_pair(500, 500));
    auto bkps = find_2d_data_bkps(centers);
    ASSERT_TRUE(!bkps.empty());
    // The breakpoint should be near index 30 (±7 frames tolerance for the
    // heuristic sliding-window algorithm).
    ASSERT_TRUE(bkps[0] >= 23 && bkps[0] <= 37);
    return true;
}

// ---- Runner -------------------------------------------------------------

int main() {
    struct Test { const char* name; bool(*fn)(); };
    Test tests[] = {
        {"find_idxs_interval_basic",       test_find_idxs_interval_basic},
        {"find_idxs_interval_single",      test_find_idxs_interval_single},
        {"get_interval_average_bbox",      test_get_interval_average_bbox},
        {"get_interval_average_bbox_all_none", test_get_interval_average_bbox_all_none},
        {"refine_bkps_basic",              test_refine_bkps_basic},
        {"refine_bkps_already_small",      test_refine_bkps_already_small},
        {"find_2d_data_bkps_empty",        test_find_2d_data_bkps_empty},
        {"find_2d_data_bkps_constant",     test_find_2d_data_bkps_constant},
        {"find_2d_data_bkps_with_jump",    test_find_2d_data_bkps_with_jump},
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
