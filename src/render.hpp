#pragma once

#include "volume.hpp"

#include <string>
#include <vector>

struct Vec3 {
    float x = 0.f;
    float y = 0.f;
    float z = 0.f;
};

struct Color {
    float r = 0.f;
    float g = 0.f;
    float b = 0.f;
};

struct Image {
    int width = 0;
    int height = 0;
    std::vector<Color> pixels;
};

struct Camera {
    Vec3 eye{1.15f, 1.28f, -0.22f};
    Vec3 target{0.50f, 0.50f, 0.50f};
};

// direction — единичный вектор. Луч, не попавший в куб, даёт чёрный цвет.
[[nodiscard]] Color march(const Volume& volume, Vec3 origin, Vec3 direction);

[[nodiscard]] Image render_image(const Volume& volume, int width, int height, Camera camera = {});

[[nodiscard]] bool write_png(const Image& image, const std::string& path, std::string& error);
