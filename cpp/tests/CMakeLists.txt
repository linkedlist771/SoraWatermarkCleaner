# Tests depend only on sorawm_core (no ONNX Runtime needed).

# Simple test runner (no external test framework required).
add_executable(test_imputation test_imputation.cpp)
target_link_libraries(test_imputation PRIVATE sorawm_core)
add_test(NAME test_imputation COMMAND test_imputation)

add_executable(test_video_utils test_video_utils.cpp)
target_link_libraries(test_video_utils PRIVATE sorawm_core)
add_test(NAME test_video_utils COMMAND test_video_utils)
