#include "volume.hpp"

#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <utility>

namespace {

constexpr char kMagic[8] = {'S', 'V', 'Z', 'C', 'U', 'B', 'E', '1'};

std::uint64_t voxel_count(int n) {
    const auto side = static_cast<std::uint64_t>(n);
    return side * side * side;
}

std::size_t index_of(int n, int x, int y, int z) {
    const auto side = static_cast<std::size_t>(n);
    return static_cast<std::size_t>(x) + side * (static_cast<std::size_t>(y) + side * static_cast<std::size_t>(z));
}

void write_u32_le(std::ostream& out, std::uint32_t value) {
    const unsigned char bytes[4] = {
        static_cast<unsigned char>(value & 0xffu),
        static_cast<unsigned char>((value >> 8) & 0xffu),
        static_cast<unsigned char>((value >> 16) & 0xffu),
        static_cast<unsigned char>((value >> 24) & 0xffu),
    };
    out.write(reinterpret_cast<const char*>(bytes), 4);
}

std::uint32_t read_u32_le(const unsigned char* bytes) {
    return static_cast<std::uint32_t>(bytes[0])
        | (static_cast<std::uint32_t>(bytes[1]) << 8)
        | (static_cast<std::uint32_t>(bytes[2]) << 16)
        | (static_cast<std::uint32_t>(bytes[3]) << 24);
}

}  // namespace

Volume::Volume(int n) {
    if (n < 1 || n > kMaxSize) {
        throw std::invalid_argument("размер куба от 1 до 512");
    }
    n_ = n;
    data_.assign(static_cast<std::size_t>(voxel_count(n)), 0.f);
}

float Volume::at(int x, int y, int z) const {
    return data_[index_of(n_, x, y, z)];
}

void Volume::set(int x, int y, int z, float value) {
    data_[index_of(n_, x, y, z)] = value;
}

float Volume::sample(float x, float y, float z) const {
    if (n_ <= 0) {
        return 0.f;
    }

    const auto coord = [n = n_](float p, int& i0, int& i1, float& t) {
        const float u = p * static_cast<float>(n) - 0.5f;
        const float base = std::floor(u);
        i0 = static_cast<int>(base);
        t = u - base;
        i1 = i0 + 1;
        if (i0 < 0) {
            i0 = 0;
        }
        if (i1 < 0) {
            i1 = 0;
        }
        if (i0 >= n) {
            i0 = n - 1;
        }
        if (i1 >= n) {
            i1 = n - 1;
        }
    };

    int x0 = 0;
    int x1 = 0;
    int y0 = 0;
    int y1 = 0;
    int z0 = 0;
    int z1 = 0;
    float tx = 0.f;
    float ty = 0.f;
    float tz = 0.f;
    coord(x, x0, x1, tx);
    coord(y, y0, y1, ty);
    coord(z, z0, z1, tz);

    const float c000 = at(x0, y0, z0);
    const float c100 = at(x1, y0, z0);
    const float c010 = at(x0, y1, z0);
    const float c110 = at(x1, y1, z0);
    const float c001 = at(x0, y0, z1);
    const float c101 = at(x1, y0, z1);
    const float c011 = at(x0, y1, z1);
    const float c111 = at(x1, y1, z1);

    const float c00 = c000 * (1.f - tx) + c100 * tx;
    const float c01 = c001 * (1.f - tx) + c101 * tx;
    const float c10 = c010 * (1.f - tx) + c110 * tx;
    const float c11 = c011 * (1.f - tx) + c111 * tx;
    const float c0 = c00 * (1.f - ty) + c10 * ty;
    const float c1 = c01 * (1.f - ty) + c11 * ty;
    return c0 * (1.f - tz) + c1 * tz;
}

bool Volume::save_f32(const Volume& volume, const std::string& path, std::string& error) {
    if (volume.empty()) {
        error = "нечего записывать: куб пуст";
        return false;
    }
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        error = "не удалось записать " + path;
        return false;
    }
    out.write(kMagic, 8);
    write_u32_le(out, static_cast<std::uint32_t>(volume.size()));
    out.write(reinterpret_cast<const char*>(volume.data().data()),
              static_cast<std::streamsize>(volume.data().size() * sizeof(float)));
    if (!out) {
        error = "запись оборвалась: " + path;
        return false;
    }
    return true;
}

bool Volume::load_f32(const std::string& path, Volume& volume, std::string& error) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        error = "не удалось открыть " + path;
        return false;
    }
    in.seekg(0, std::ios::end);
    const auto end_pos = in.tellg();
    if (end_pos < 0) {
        error = "не удалось прочитать " + path;
        return false;
    }
    const auto length = static_cast<std::uint64_t>(end_pos);
    in.seekg(0);
    if (length < 12) {
        error = "файл куба слишком короткий";
        return false;
    }

    char magic[8] = {};
    in.read(magic, 8);
    if (std::memcmp(magic, kMagic, 8) != 0) {
        error = "файл не похож на куб scienceviz";
        return false;
    }
    unsigned char size_bytes[4] = {};
    in.read(reinterpret_cast<char*>(size_bytes), 4);
    const int n = static_cast<int>(read_u32_le(size_bytes));
    if (n < 1 || n > kMaxSize) {
        error = "размер куба от 1 до 512";
        return false;
    }
    const std::uint64_t expect = 12 + voxel_count(n) * sizeof(float);
    if (length != expect) {
        error = "длина файла не совпадает с N³";
        return false;
    }

    Volume loaded(n);
    in.read(reinterpret_cast<char*>(loaded.data().data()),
            static_cast<std::streamsize>(loaded.data().size() * sizeof(float)));
    if (!in) {
        error = "не удалось прочитать воксели";
        return false;
    }
    volume = std::move(loaded);
    return true;
}
