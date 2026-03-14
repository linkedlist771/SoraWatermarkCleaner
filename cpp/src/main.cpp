/// SoraWatermarkCleaner – C++ CLI
///
/// Usage:
///   sorawm_cli -i <input> -o <output>
///              [--detector <yolo.onnx>] [--cleaner <lama.onnx>]
///              [--batch] [--quiet]

#include <cstring>
#include <iostream>
#include <string>

#include "sorawm/pipeline.hpp"

static void print_usage(const char* prog) {
    std::cout
        << "SoraWatermarkCleaner – C++ Edition\n\n"
        << "Usage:\n"
        << "  " << prog << " -i <input> -o <output> [options]\n\n"
        << "Options:\n"
        << "  -i, --input      Input video file or directory (required)\n"
        << "  -o, --output     Output video file or directory (required)\n"
        << "  --detector       Path to YOLO ONNX model  (default: models/yolo.onnx)\n"
        << "  --cleaner        Path to LAMA ONNX model  (default: models/lama.onnx)\n"
        << "  --batch          Process entire input directory\n"
        << "  --batch-size N   Detection batch size      (default: 4)\n"
        << "  --quiet          Suppress progress output\n"
        << "  -h, --help       Show this help\n";
}

int main(int argc, char** argv) {
    std::string input, output;
    std::string detector = "models/yolo.onnx";
    std::string cleaner  = "models/lama.onnx";
    bool batch = false;
    bool quiet = false;
    int batch_size = 4;

    for (int i = 1; i < argc; ++i) {
        auto arg = [&](const char* s) { return std::strcmp(argv[i], s) == 0; };
        auto next = [&]() -> const char* {
            return (i + 1 < argc) ? argv[++i] : nullptr;
        };

        if (arg("-i") || arg("--input"))       { auto v = next(); if (v) input = v; }
        else if (arg("-o") || arg("--output")) { auto v = next(); if (v) output = v; }
        else if (arg("--detector"))            { auto v = next(); if (v) detector = v; }
        else if (arg("--cleaner"))             { auto v = next(); if (v) cleaner = v; }
        else if (arg("--batch"))               { batch = true; }
        else if (arg("--batch-size"))          { auto v = next(); if (v) batch_size = std::atoi(v); }
        else if (arg("--quiet"))               { quiet = true; }
        else if (arg("-h") || arg("--help"))   { print_usage(argv[0]); return 0; }
        else {
            std::cerr << "Unknown option: " << argv[i] << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    if (input.empty() || output.empty()) {
        std::cerr << "Error: --input and --output are required.\n\n";
        print_usage(argv[0]);
        return 1;
    }

    sorawm::PipelineConfig cfg;
    cfg.detector_model    = detector;
    cfg.cleaner_model     = cleaner;
    cfg.detect_batch_size = batch_size;
    cfg.quiet             = quiet;

    sorawm::ProgressCallback progress_fn = nullptr;
    if (!quiet) {
        progress_fn = [](int pct) {
            std::cout << "\rProgress: " << pct << " %" << std::flush;
        };
    }

    try {
        sorawm::Pipeline pipeline(cfg);

        if (batch) {
            pipeline.run_batch(input, output, progress_fn);
        } else {
            pipeline.run(input, output, progress_fn);
        }

        if (!quiet) std::cout << "\nDone.\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "\nError: " << ex.what() << "\n";
        return 1;
    }
}
