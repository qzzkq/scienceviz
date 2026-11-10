#include "core/grid.hpp"

#include <stdexcept>
#include <string>

namespace svcore {

namespace {

const char* const kFieldNames[kFieldCount] = {"rho", "mom_x", "mom_y", "mom_z", "phi"};

}  // namespace

const char* field_name(Field field) {
    const int i = static_cast<int>(field);
    assert(i >= 0 && i < kFieldCount);
    return kFieldNames[i];
}

Grid3D::Grid3D(int nx, int ny, int nz, int halo)
    : nx_(nx), ny_(ny), nz_(nz), halo_(halo) {
    if (nx < 1 || ny < 1 || nz < 1) {
        throw std::invalid_argument("Grid3D: размеры блока должны быть положительными");
    }
    if (halo < 1 || halo > kMaxHalo) {
        throw std::invalid_argument("Grid3D: толщина гало — 1 или 2, получено "
                                    + std::to_string(halo));
    }
    // Периодическое гало копируется из своих ячеек на той же оси, поэтому блок
    // не может быть тоньше гало. То же требование у обмена слоями по z.
    if (nx < halo || ny < halo || nz < halo) {
        throw std::invalid_argument("Grid3D: блок тоньше гало");
    }

    sx_ = nx_ + 2 * halo_;
    sy_ = ny_ + 2 * halo_;
    sz_ = nz_ + 2 * halo_;

    for (int i = 0; i < kFieldCount; ++i) {
        fields_[i].assign(total_count(), 0.0f);
    }
}

std::size_t Grid3D::own_count() const {
    return static_cast<std::size_t>(nx_) * static_cast<std::size_t>(ny_)
        * static_cast<std::size_t>(nz_);
}

std::size_t Grid3D::total_count() const {
    return static_cast<std::size_t>(sx_) * static_cast<std::size_t>(sy_)
        * static_cast<std::size_t>(sz_);
}

std::size_t Grid3D::plane_count() const {
    return static_cast<std::size_t>(sx_) * static_cast<std::size_t>(sy_);
}

bool Grid3D::contains_own(int x, int y, int z) const {
    return x >= 0 && x < nx_ && y >= 0 && y < ny_ && z >= 0 && z < nz_;
}

bool Grid3D::contains_with_halo(int x, int y, int z) const {
    return x >= -halo_ && x < nx_ + halo_
        && y >= -halo_ && y < ny_ + halo_
        && z >= -halo_ && z < nz_ + halo_;
}

std::vector<float>& Grid3D::field_data(Field field) {
    const int i = static_cast<int>(field);
    assert(i >= 0 && i < kFieldCount);
    return fields_[i];
}

const std::vector<float>& Grid3D::field_data(Field field) const {
    const int i = static_cast<int>(field);
    assert(i >= 0 && i < kFieldCount);
    return fields_[i];
}

float* Grid3D::data(Field field) { return field_data(field).data(); }

const float* Grid3D::data(Field field) const { return field_data(field).data(); }

void Grid3D::fill(Field field, float value) {
    std::vector<float>& cells = field_data(field);
    const std::size_t n = cells.size();
    float* out = cells.data();
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = value;
    }
}

void Grid3D::fill_all(float value) {
    for (int i = 0; i < kFieldCount; ++i) {
        fill(static_cast<Field>(i), value);
    }
}

double Grid3D::sum_own(Field field) const {
    const float* cells = data(field);
    double sum = 0.0;
    for (int z = 0; z < nz_; ++z) {
        for (int y = 0; y < ny_; ++y) {
            const std::size_t row = index(0, y, z);
            for (int x = 0; x < nx_; ++x) {
                sum += static_cast<double>(cells[row + static_cast<std::size_t>(x)]);
            }
        }
    }
    return sum;
}

void Grid3D::fill_halo_periodic(Field field) {
    if (empty()) {
        return;
    }
    float* cells = data(field);
    const int g = halo_;

    // Три прохода по всё более широкому диапазону: так рёбра и углы получаются
    // сами, без отдельного кода для них.

    // 1) По x, только по своим y и z.
    for (int z = 0; z < nz_; ++z) {
        for (int y = 0; y < ny_; ++y) {
            for (int t = 1; t <= g; ++t) {
                cells[index(-t, y, z)] = cells[index(nx_ - t, y, z)];
                cells[index(nx_ - 1 + t, y, z)] = cells[index(t - 1, y, z)];
            }
        }
    }

    // 2) По y, уже по всему диапазону x вместе с гало, по своим z.
    for (int z = 0; z < nz_; ++z) {
        for (int t = 1; t <= g; ++t) {
            for (int x = -g; x < nx_ + g; ++x) {
                cells[index(x, -t, z)] = cells[index(x, ny_ - t, z)];
                cells[index(x, ny_ - 1 + t, z)] = cells[index(x, t - 1, z)];
            }
        }
    }

    // 3) По z: целыми плоскостями, они лежат одним куском.
    const std::size_t plane = plane_count();
    for (int t = 1; t <= g; ++t) {
        const float* src_lo = cells + z_plane_offset(nz_ - t);
        float* dst_lo = cells + z_plane_offset(-t);
        const float* src_hi = cells + z_plane_offset(t - 1);
        float* dst_hi = cells + z_plane_offset(nz_ - 1 + t);
        for (std::size_t i = 0; i < plane; ++i) {
            dst_lo[i] = src_lo[i];
            dst_hi[i] = src_hi[i];
        }
    }
}

void Grid3D::fill_halo_periodic() {
    for (int i = 0; i < kFieldCount; ++i) {
        fill_halo_periodic(static_cast<Field>(i));
    }
}

}  // namespace svcore
