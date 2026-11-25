#include "grid_volume.hpp"
#include "npy.hpp"
#include "render.hpp"
#include "synthetic.hpp"

#include <cmath>
#include <iostream>
#include <sstream>
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
        << "scienceviz --npy cube.npy --out frame.png\n"
        << "scienceviz --grid-npy cube.npy --out frame.png\n"
        << "камера: --eye x,y,z --target x,y,z (куб от 0 до 1 по каждой оси)\n";
}

bool next_value(int& index, int argc, char** argv, std::string& value) {
    if (index + 1 >= argc) {
        std::cerr << argv[index] << " требует значение\n";
        return false;
    }
    value = argv[++index];
    return true;
}

bool parse_point(const std::string& value, Vec3& point) {
    std::istringstream input(value);
    char first = 0;
    char second = 0;
    if (!(input >> point.x >> first >> point.y >> second >> point.z) || first != ',' || second != ',') {
        return false;
    }
    input >> std::ws;
    return input.eof() && std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z);
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
    std::string grid_npy_path;
    Camera camera;

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
        } else if (arg == "--grid-npy") {
            if (!next_value(i, argc, argv, grid_npy_path)) {
                return 1;
            }
        } else if (arg == "--eye" || arg == "--target") {
            if (!next_value(i, argc, argv, value)) {
                return 1;
            }
            Vec3& point = arg == "--eye" ? camera.eye : camera.target;
            if (!parse_point(value, point)) {
                std::cerr << arg << " требует три числа x,y,z\n";
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
    if (static_cast<int>(!bin_path.empty()) + static_cast<int>(!npy_path.empty())
        + static_cast<int>(!grid_npy_path.empty()) > 1) {
        std::cerr << "нужен один источник: --bin, --npy или --grid-npy\n";
        return 1;
    }
    if (size_set && (!bin_path.empty() || !npy_path.empty() || !grid_npy_path.empty())) {
        std::cerr << "размер берётся из файла, --size здесь не нужен\n";
        return 1;
    }
    if (size < 1 || size > Volume::kMaxSize) {
        std::cerr << "размер куба от 1 до 512\n";
        return 1;
    }

    Volume volume;
    std::string error;
    if (!npy_path.empty() || !grid_npy_path.empty()) {
        if (!load_npy(grid_npy_path.empty() ? npy_path : grid_npy_path, volume, error)) {
            std::cerr << error << "\n";
            return 1;
        }
        if (!grid_npy_path.empty()) {
            const int n = volume.size();
            svcore::Grid3D grid(n, n, n, 1);
            density_to_grid(volume, grid);
            std::cout << "средняя плотность сетки: "
                      << grid.sum_own(svcore::Field::Rho) / grid.own_count() << "\n";
            volume = density_to_volume(grid);
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
        try {
            const Image image = render_image(volume, kWidth, kHeight, camera);
            if (!write_png(image, out_path, error)) {
                std::cerr << error << "\n";
                return 1;
            }
        } catch (const std::invalid_argument& exc) {
            std::cerr << exc.what() << "\n";
            return 1;
        }
        std::cout << out_path << " " << kWidth << "x" << kHeight << ", куб " << volume.size() << "\n";
    }
    return 0;
}
