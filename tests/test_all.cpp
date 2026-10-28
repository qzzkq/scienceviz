#include "npy.hpp"
#include "render.hpp"
#include "synthetic.hpp"
#include "volume.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

int g_failed = 0;

void check(bool ok, const char* expr, int line) {
    if (!ok) {
        std::cerr << "fail " << line << ": " << expr << "\n";
        ++g_failed;
    }
}

#define CHECK(expr) check(static_cast<bool>(expr), #expr, __LINE__)

float luma(Color color) {
    return 0.2126f * color.r + 0.7152f * color.g + 0.0722f * color.b;
}

void write_bytes(const fs::path& path, const std::vector<unsigned char>& bytes) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

std::vector<unsigned char> make_npy(const std::vector<int>& shape,
                                    const std::string& descr,
                                    bool fortran,
                                    const std::vector<float>& data) {
    std::string shape_text = "(";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) {
            shape_text += ", ";
        }
        shape_text += std::to_string(shape[i]);
    }
    if (shape.size() == 1) {
        shape_text += ",";
    }
    shape_text += ")";

    std::string header = "{'descr': '" + descr + "', 'fortran_order': " + (fortran ? "True" : "False")
        + ", 'shape': " + shape_text + ", }";
    const std::size_t header_len = header.size() + 1;
    const std::size_t pad = (16 - ((10 + header_len) % 16)) % 16;
    header.append(pad, ' ');
    header.push_back('\n');

    std::vector<unsigned char> file;
    file.push_back(0x93);
    const char magic[] = "NUMPY";
    file.insert(file.end(), magic, magic + 5);
    file.push_back(1);
    file.push_back(0);
    const auto len = static_cast<std::uint16_t>(header.size());
    file.push_back(static_cast<unsigned char>(len & 0xff));
    file.push_back(static_cast<unsigned char>((len >> 8) & 0xff));
    file.insert(file.end(), header.begin(), header.end());
    const auto* raw = reinterpret_cast<const unsigned char*>(data.data());
    file.insert(file.end(), raw, raw + data.size() * sizeof(float));
    return file;
}

void test_sample() {
    Volume volume(4);
    volume.set(2, 3, 1, 3.5f);
    const float at_center = volume.sample(2.5f / 4.f, 3.5f / 4.f, 1.5f / 4.f);
    CHECK(std::fabs(at_center - 3.5f) < 1.0e-5f);

    Volume line(4);
    line.set(0, 0, 0, 0.f);
    line.set(1, 0, 0, 10.f);
    const float midway = line.sample(0.25f, 0.5f / 4.f, 0.5f / 4.f);
    CHECK(std::fabs(midway - 5.f) < 1.0e-4f);
}

void test_synthetic() {
    const float node = synthetic_density(kNodeAx, kNodeY, kNodeZ);
    const float corner = synthetic_density(0.02f, 0.02f, 0.02f);
    CHECK(node > 10.f);
    CHECK(corner < 0.01f);
    CHECK(node > corner);
}

void test_march() {
    Volume volume(32);
    fill_synthetic(volume);
    const Vec3 direction{1.f, 0.f, 0.f};
    const float through_cell = 15.5f / 32.f;
    const Color through_node = march(volume, {-0.2f, through_cell, through_cell}, direction);
    const Color through_void = march(volume, {-0.2f, 0.08f, 0.08f}, direction);
    CHECK(luma(through_node) > 0.05f);
    CHECK(luma(through_node) > luma(through_void) + 0.05f);
}

void test_frame() {
    Volume volume(32);
    fill_synthetic(volume);
    const Image image = render_image(volume, 32, 32);
    const Color center = image.pixels[16 * 32 + 16];
    const Color corner = image.pixels[0];
    CHECK(luma(center) > 0.05f);
    CHECK(luma(center) > luma(corner) + 0.02f);

    const fs::path png = fs::temp_directory_path() / "scienceviz_test.png";
    std::string error;
    CHECK(write_png(image, png.string(), error));
    std::ifstream in(png, std::ios::binary);
    unsigned char signature[8] = {};
    in.read(reinterpret_cast<char*>(signature), 8);
    const unsigned char expected[8] = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'};
    CHECK(in.gcount() == 8);
    CHECK(std::memcmp(signature, expected, 8) == 0);
    std::error_code ignored;
    fs::remove(png, ignored);
}

void test_f32_roundtrip() {
    Volume volume(4);
    volume.set(1, 2, 3, 7.25f);
    const fs::path path = fs::temp_directory_path() / "scienceviz_test.f32";
    std::string error;
    CHECK(Volume::save_f32(volume, path.string(), error));
    Volume loaded;
    CHECK(Volume::load_f32(path.string(), loaded, error));
    CHECK(loaded.size() == 4);
    CHECK(std::fabs(loaded.at(1, 2, 3) - 7.25f) < 1.0e-6f);

    std::ofstream garbage(path, std::ios::binary);
    garbage << "not-a-cube-file!!";
    garbage.close();
    CHECK(!Volume::load_f32(path.string(), loaded, error));
    CHECK(error.find("не похож") != std::string::npos);
    std::error_code ignored;
    fs::remove(path, ignored);
}

void test_npy() {
    const fs::path path = fs::temp_directory_path() / "scienceviz_test.npy";
    std::vector<float> data(8, 0.f);
    data[0] = 1.f;
    data[1] = 2.f;
    data[7] = 4.f;
    write_bytes(path, make_npy({2, 2, 2}, "<f4", false, data));

    Volume volume;
    std::string error;
    if (!load_npy(path.string(), volume, error)) {
        std::cerr << error << "\n";
    }
    CHECK(volume.size() == 2);
    if (volume.size() == 2) {
        CHECK(std::fabs(volume.at(0, 0, 0) - 1.f) < 1.0e-6f);
        CHECK(std::fabs(volume.at(1, 0, 0) - 2.f) < 1.0e-6f);
        CHECK(std::fabs(volume.at(1, 1, 1) - 4.f) < 1.0e-6f);
    }

    write_bytes(path, make_npy({4, 4}, "<f4", false, std::vector<float>(16, 1.f)));
    CHECK(!load_npy(path.string(), volume, error));
    CHECK(error.find("трёх") != std::string::npos);

    write_bytes(path, make_npy({2, 3, 4}, "<f4", false, std::vector<float>(24, 1.f)));
    CHECK(!load_npy(path.string(), volume, error));
    CHECK(error.find("N×N×N") != std::string::npos);

    write_bytes(path, make_npy({2, 2, 2}, "<f8", false, {}));
    CHECK(!load_npy(path.string(), volume, error));
    CHECK(error.find("float32") != std::string::npos);

    write_bytes(path, make_npy({2, 2, 2}, "<f4", true, data));
    CHECK(!load_npy(path.string(), volume, error));
    CHECK(error.find("fortran") != std::string::npos);

    auto short_file = make_npy({2, 2, 2}, "<f4", false, data);
    short_file.resize(short_file.size() - 4);
    write_bytes(path, short_file);
    CHECK(!load_npy(path.string(), volume, error));
    CHECK(error.find("короче") != std::string::npos);

    std::error_code ignored;
    fs::remove(path, ignored);
}

int main() {
    test_sample();
    test_synthetic();
    test_march();
    test_frame();
    test_f32_roundtrip();
    test_npy();
    if (g_failed != 0) {
        std::cerr << g_failed << " checks failed\n";
        return 1;
    }
    std::cout << "ok\n";
    return 0;
}
