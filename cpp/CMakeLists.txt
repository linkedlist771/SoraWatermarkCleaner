cmake_minimum_required(VERSION 3.20)
project(SoraWatermarkCleaner VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---- Dependencies --------------------------------------------------------
find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio)

# ONNX Runtime – expected to be installed or provided via CMAKE_PREFIX_PATH.
find_package(onnxruntime QUIET)
if(NOT onnxruntime_FOUND)
    # Fallback: try pkg-config or manual paths.
    find_library(ONNXRUNTIME_LIB onnxruntime)
    find_path(ONNXRUNTIME_INCLUDE onnxruntime_cxx_api.h
              PATH_SUFFIXES onnxruntime)
    if(ONNXRUNTIME_LIB AND ONNXRUNTIME_INCLUDE)
        add_library(onnxruntime::onnxruntime SHARED IMPORTED)
        set_target_properties(onnxruntime::onnxruntime PROPERTIES
            IMPORTED_LOCATION "${ONNXRUNTIME_LIB}"
            INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE}")
    else()
        message(WARNING
            "ONNX Runtime not found. Build will succeed for imputation/video "
            "components, but detector and cleaner targets require ONNX Runtime. "
            "Install it or set CMAKE_PREFIX_PATH.")
    endif()
endif()

# ---- Library: sorawm_core (no ONNX dependency) --------------------------
add_library(sorawm_core STATIC
    src/video_loader.cpp
    src/imputation.cpp
)
target_include_directories(sorawm_core PUBLIC include)
target_link_libraries(sorawm_core PUBLIC ${OpenCV_LIBS})

# ---- Library: sorawm_infer (needs ONNX Runtime) -------------------------
if(TARGET onnxruntime::onnxruntime OR ONNXRUNTIME_LIB)
    set(ONNX_TARGET onnxruntime::onnxruntime)
    if(NOT TARGET onnxruntime::onnxruntime)
        set(ONNX_TARGET ${ONNXRUNTIME_LIB})
    endif()

    add_library(sorawm_infer STATIC
        src/watermark_detector.cpp
        src/watermark_cleaner.cpp
        src/pipeline.cpp
    )
    target_include_directories(sorawm_infer PUBLIC include)
    if(ONNXRUNTIME_INCLUDE)
        target_include_directories(sorawm_infer PUBLIC ${ONNXRUNTIME_INCLUDE})
    endif()
    target_link_libraries(sorawm_infer PUBLIC sorawm_core ${ONNX_TARGET})

    # ---- CLI executable --------------------------------------------------
    add_executable(sorawm_cli src/main.cpp)
    target_link_libraries(sorawm_cli PRIVATE sorawm_infer)
    install(TARGETS sorawm_cli DESTINATION bin)
endif()

# ---- Tests ---------------------------------------------------------------
option(SORAWM_BUILD_TESTS "Build unit tests" ON)
if(SORAWM_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
