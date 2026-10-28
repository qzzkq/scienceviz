#include "render.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace {

constexpr Vec3 kEye{1.15f, 1.28f, -0.22f};
constexpr Vec3 kTarget{0.50f, 0.50f, 0.50f};
constexpr float kFovDeg = 34.f;
constexpr float kInset = 0.18f;
constexpr float kDensityFloor = 0.9f;

struct BoxHit {
    bool hit = false;
    float t_near = 0.f;
    float t_far = 0.f;
    Vec3 normal{};
};

BoxHit hit_box(Vec3 origin, Vec3 direction, Vec3 box_min, Vec3 box_max) {
    BoxHit result;
    result.t_near = -1.0e30f;
    result.t_far = 1.0e30f;
    const float origin_a[3] = {origin.x, origin.y, origin.z};
    const float direction_a[3] = {direction.x, direction.y, direction.z};
    const float min_a[3] = {box_min.x, box_min.y, box_min.z};
    const float max_a[3] = {box_max.x, box_max.y, box_max.z};
    for (int axis = 0; axis < 3; ++axis) {
        if (std::fabs(direction_a[axis]) < 1.0e-8f) {
            if (origin_a[axis] < min_a[axis] || origin_a[axis] > max_a[axis]) {
                return {};
            }
            continue;
        }
        const float t_at_min = (min_a[axis] - origin_a[axis]) / direction_a[axis];
        const float t_at_max = (max_a[axis] - origin_a[axis]) / direction_a[axis];
        const float t_enter = t_at_min < t_at_max ? t_at_min : t_at_max;
        const float t_exit = t_at_min < t_at_max ? t_at_max : t_at_min;
        const float entry_sign = t_at_min < t_at_max ? -1.f : 1.f;
        if (t_enter > result.t_near) {
            result.t_near = t_enter;
            result.normal = {};
            if (axis == 0) {
                result.normal.x = entry_sign;
            } else if (axis == 1) {
                result.normal.y = entry_sign;
            } else {
                result.normal.z = entry_sign;
            }
        }
        if (t_exit < result.t_far) {
            result.t_far = t_exit;
        }
        if (result.t_near > result.t_far) {
            return {};
        }
    }
    result.hit = true;
    return result;
}

Vec3 sub(Vec3 a, Vec3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

float dot(Vec3 a, Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vec3 cross(Vec3 a, Vec3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

Vec3 scale(Vec3 a, float s) {
    return {a.x * s, a.y * s, a.z * s};
}

Vec3 add(Vec3 a, Vec3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

Vec3 normalize(Vec3 a) {
    const float length = std::sqrt(dot(a, a));
    return scale(a, 1.f / length);
}

Color mix(Color a, Color b, float t) {
    return {
        a.r * (1.f - t) + b.r * t,
        a.g * (1.f - t) + b.g * t,
        a.b * (1.f - t) + b.b * t,
    };
}

Color ramp(float sigma) {
    float u = sigma / 2.8f;
    if (u < 0.f) {
        u = 0.f;
    }
    if (u > 1.f) {
        u = 1.f;
    }
    const Color low{0.02f, 0.05f, 0.12f};
    const Color mid{0.15f, 0.78f, 0.88f};
    const Color high{1.f, 0.97f, 0.92f};
    if (u < 0.5f) {
        return mix(low, mid, u / 0.5f);
    }
    return mix(mid, high, (u - 0.5f) / 0.5f);
}

bool slab(Vec3 origin, Vec3 direction, float& tmin, float& tmax) {
    tmin = 0.f;
    tmax = 1.0e30f;
    const float origin_a[3] = {origin.x, origin.y, origin.z};
    const float direction_a[3] = {direction.x, direction.y, direction.z};
    for (int axis = 0; axis < 3; ++axis) {
        if (std::fabs(direction_a[axis]) < 1.0e-8f) {
            if (origin_a[axis] < 0.f || origin_a[axis] > 1.f) {
                return false;
            }
            continue;
        }
        const float inv = 1.f / direction_a[axis];
        float t1 = (0.f - origin_a[axis]) * inv;
        float t2 = (1.f - origin_a[axis]) * inv;
        if (t1 > t2) {
            const float swap = t1;
            t1 = t2;
            t2 = swap;
        }
        if (t1 > tmin) {
            tmin = t1;
        }
        if (t2 < tmax) {
            tmax = t2;
        }
        if (tmin > tmax) {
            return false;
        }
    }
    return tmax > 0.f;
}

int clamp_index(int index, int size) {
    if (index < 0) {
        return 0;
    }
    if (index >= size) {
        return size - 1;
    }
    return index;
}

int to_byte(float channel) {
    if (channel < 0.f) {
        channel = 0.f;
    }
    if (channel > 1.f) {
        channel = 1.f;
    }
    const float curved = std::pow(channel, 0.45f);
    return static_cast<int>(curved * 255.f + 0.5f);
}

}  // namespace

Color march(const Volume& volume, Vec3 origin, Vec3 direction) {
    Color out{};
    if (volume.size() < 1) {
        return out;
    }
    float tmin = 0.f;
    float tmax = 0.f;
    if (!slab(origin, direction, tmin, tmax)) {
        return out;
    }
    const int n = volume.size();
    const float cell = 1.f / static_cast<float>(n);
    const float gap = kInset * cell;
    const Vec3 light = normalize(Vec3{0.35f, 0.85f, 0.4f});
    float t = tmin > 0.f ? tmin : 0.f;
    const Vec3 start = add(origin, scale(direction, t + cell * 1.0e-3f));
    int ix = clamp_index(static_cast<int>(std::floor(start.x / cell)), n);
    int iy = clamp_index(static_cast<int>(std::floor(start.y / cell)), n);
    int iz = clamp_index(static_cast<int>(std::floor(start.z / cell)), n);
    const int step_x = direction.x < 0.f ? -1 : 1;
    const int step_y = direction.y < 0.f ? -1 : 1;
    const int step_z = direction.z < 0.f ? -1 : 1;
    const float t_delta_x = std::fabs(direction.x) < 1.0e-8f ? 1.0e30f : cell / std::fabs(direction.x);
    const float t_delta_y = std::fabs(direction.y) < 1.0e-8f ? 1.0e30f : cell / std::fabs(direction.y);
    const float t_delta_z = std::fabs(direction.z) < 1.0e-8f ? 1.0e30f : cell / std::fabs(direction.z);
    auto boundary = [&](int index, int step, float origin_a, float direction_a) {
        if (std::fabs(direction_a) < 1.0e-8f) {
            return 1.0e30f;
        }
        const float edge = (index + (step > 0 ? 1 : 0)) * cell;
        return (edge - origin_a) / direction_a;
    };
    float t_max_x = boundary(ix, step_x, origin.x, direction.x);
    float t_max_y = boundary(iy, step_y, origin.y, direction.y);
    float t_max_z = boundary(iz, step_z, origin.z, direction.z);
    float transmittance = 1.f;

    for (int steps = 0; steps < n * 3; ++steps) {
        if (ix < 0 || iy < 0 || iz < 0 || ix >= n || iy >= n || iz >= n || t > tmax) {
            break;
        }
        const float density = volume.at(ix, iy, iz);
        if (density >= kDensityFloor) {
            const Vec3 box_min{
                ix * cell + gap,
                iy * cell + gap,
                iz * cell + gap,
            };
            const Vec3 box_max{
                (ix + 1) * cell - gap,
                (iy + 1) * cell - gap,
                (iz + 1) * cell - gap,
            };
            const BoxHit hit = hit_box(origin, direction, box_min, box_max);
            if (hit.hit && hit.t_far > t && hit.t_near < tmax) {
                const float shade = 0.28f + 0.72f * std::max(0.f, dot(hit.normal, light));
                const Color glow = ramp(std::log(1.f + density));
                constexpr float alpha = 0.97f;
                out.r += transmittance * alpha * glow.r * shade;
                out.g += transmittance * alpha * glow.g * shade;
                out.b += transmittance * alpha * glow.b * shade;
                transmittance *= 1.f - alpha;
                if (transmittance < 0.02f) {
                    break;
                }
                t = hit.t_far + cell * 1.0e-3f;
            }
        }
        if (t_max_x <= t_max_y && t_max_x <= t_max_z) {
            ix += step_x;
            t = t_max_x;
            t_max_x += t_delta_x;
        } else if (t_max_y <= t_max_z) {
            iy += step_y;
            t = t_max_y;
            t_max_y += t_delta_y;
        } else {
            iz += step_z;
            t = t_max_z;
            t_max_z += t_delta_z;
        }
    }
    return out;
}

Image render_image(const Volume& volume, int width, int height) {
    Image image;
    image.width = width;
    image.height = height;
    image.pixels.resize(static_cast<std::size_t>(width) * static_cast<std::size_t>(height));
    if (width < 1 || height < 1) {
        return image;
    }

    const Vec3 forward = normalize(sub(kTarget, kEye));
    const Vec3 right = normalize(cross(forward, Vec3{0.f, 1.f, 0.f}));
    const Vec3 up = cross(right, forward);
    const float half_h = std::tan(kFovDeg * 3.14159265f / 360.f);
    const float half_w = half_h * (static_cast<float>(width) / static_cast<float>(height));

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const float nx = ((static_cast<float>(x) + 0.5f) / static_cast<float>(width)) * 2.f - 1.f;
            const float ny = 1.f - ((static_cast<float>(y) + 0.5f) / static_cast<float>(height)) * 2.f;
            const Vec3 direction = normalize(add(forward, add(scale(right, nx * half_w), scale(up, ny * half_h))));
            image.pixels[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x)] =
                march(volume, kEye, direction);
        }
    }
    return image;
}

bool write_png(const Image& image, const std::string& path, std::string& error) {
    if (image.width < 1 || image.height < 1) {
        error = "пустой кадр";
        return false;
    }
    std::vector<unsigned char> bytes(static_cast<std::size_t>(image.width) * static_cast<std::size_t>(image.height) * 3);
    for (std::size_t i = 0; i < image.pixels.size(); ++i) {
        bytes[i * 3] = static_cast<unsigned char>(to_byte(image.pixels[i].r));
        bytes[i * 3 + 1] = static_cast<unsigned char>(to_byte(image.pixels[i].g));
        bytes[i * 3 + 2] = static_cast<unsigned char>(to_byte(image.pixels[i].b));
    }
    if (stbi_write_png(path.c_str(), image.width, image.height, 3, bytes.data(), image.width * 3) == 0) {
        error = "не удалось записать " + path;
        return false;
    }
    return true;
}
