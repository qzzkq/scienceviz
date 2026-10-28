#include "npy.hpp"
#include "render.hpp"
#include "synthetic.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace {

void usage() {
    std::cerr
        << "scienceviz --size 64 --out frame.png\n"
        << "scienceviz --size 64 --dump cube.f32\n"
        << "scienceviz --bin cube.f32 --out frame.png\n"
        << "scienceviz --npy cube.npy --out frame.png\n";
}

bool next_value(int& index, int argc, char** argv, std::string& value) {
    if (index + 1 >= argc) {
        std::cerr << argv[index] << " требует значение\n";
        return false;
    }
    value = argv[++index];
    return true;
}

}  // namespace

int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif

    int size = 64;
    bool size_set = false;
    std::string out_path;
    std::string dump_path;
    std::string bin_path;
    std::string npy_path;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        std::string value;
        if (arg == "--size") {
            if (!next_value(i, argc, argv, value)) {
                return 1;
            }
            try {
                size = std::stoi(value);
            } catch (const std::exception&) {
                std::cerr << "плохой --size\n";
                return 1;
            }
            size_set = true;
        } else if (arg == "--out") {
            if (!next_value(i, argc, argv, out_path)) {
                return 1;
            }
        } else if (arg == "--dump") {
            if (!next_value(i, argc, argv, dump_path)) {
                return 1;
            }
        } else if (arg == "--bin") {
            if (!next_value(i, argc, argv, bin_path)) {
                return 1;
            }
        } else if (arg == "--npy") {
            if (!next_value(i, argc, argv, npy_path)) {
                return 1;
            }
        } else {
            usage();
            return 1;
        }
    }

    if (out_path.empty() && dump_path.empty()) {
        usage();
        return 1;
    }
    if (!bin_path.empty() && !npy_path.empty()) {
        std::cerr << "нужен один источник: --bin или --npy\n";
        return 1;
    }
    if (size_set && (!bin_path.empty() || !npy_path.empty())) {
        std::cerr << "размер берётся из файла, --size здесь не нужен\n";
        return 1;
    }
    if (size < 1 || size > Volume::kMaxSize) {
        std::cerr << "размер куба от 1 до 512\n";
        return 1;
    }

    Volume volume;
    std::string error;
    if (!npy_path.empty()) {
        if (!load_npy(npy_path, volume, error)) {
            std::cerr << error << "\n";
            return 1;
        }
    } else if (!bin_path.empty()) {
        if (!Volume::load_f32(bin_path, volume, error)) {
            std::cerr << error << "\n";
            return 1;
        }
    } else {
        volume = Volume(size);
        fill_synthetic(volume);
    }

    if (!dump_path.empty() && !Volume::save_f32(volume, dump_path, error)) {
        std::cerr << error << "\n";
        return 1;
    }
    if (!out_path.empty()) {
        constexpr int kWidth = 1280;
        constexpr int kHeight = 720;
        const Image image = render_image(volume, kWidth, kHeight);
        if (!write_png(image, out_path, error)) {
            std::cerr << error << "\n";
            return 1;
        }
        std::cout << out_path << " " << kWidth << "x" << kHeight << ", куб " << volume.size() << "\n";
    }
    return 0;
}
