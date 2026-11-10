#pragma once

#include <cassert>
#include <cstddef>
#include <vector>

namespace svcore {

// Поля газа на сетке. На каждое поле — свой плоский массив float, внутри ячейки
// нет ни указателей, ни объектов: такую раскладку можно без переделки отдать на GPU.
enum class Field : int {
    Rho = 0,
    MomX = 1,
    MomY = 2,
    MomZ = 3,
    Phi = 4,
};

constexpr int kFieldCount = 5;

const char* field_name(Field field);

// Локальный блок сетки: nx*ny*nz своих ячеек плюс гало толщиной halo со всех сторон.
//
// Раскладка памяти — x самый быстрый (как в Volume у рендера):
//     flat = (x + halo) + sx * ((y + halo) + sy * (z + halo)),  sx = nx + 2*halo
// Отсюда плоскость z = const лежит в памяти одним куском размера sx*sy, а слой гало
// по z (halo таких плоскостей) — тоже один непрерывный кусок. Это нужно для обмена
// MPI_Sendrecv при декомпозиции слоями по оси z.
//
// Координаты своих ячеек: [0, nx) x [0, ny) x [0, nz). Гало адресуется теми же
// функциями отрицательными индексами и индексами >= n, диапазон [-halo, n + halo).
class Grid3D {
public:
    Grid3D() = default;
    Grid3D(int nx, int ny, int nz, int halo);

    // Свои ячейки.
    int nx() const { return nx_; }
    int ny() const { return ny_; }
    int nz() const { return nz_; }
    int halo() const { return halo_; }

    // Полные размеры с гало.
    int sx() const { return sx_; }
    int sy() const { return sy_; }
    int sz() const { return sz_; }

    bool empty() const { return nx_ == 0 || ny_ == 0 || nz_ == 0; }

    // Своих ячеек и всех ячеек вместе с гало.
    std::size_t own_count() const;
    std::size_t total_count() const;

    // Шаги плоского индекса: по x всегда 1.
    std::size_t stride_x() const { return 1; }
    std::size_t stride_y() const { return static_cast<std::size_t>(sx_); }
    std::size_t stride_z() const { return plane_count(); }

    // Ячеек в одной плоскости z = const вместе с гало по x и y.
    std::size_t plane_count() const;

    bool contains_own(int x, int y, int z) const;
    bool contains_with_halo(int x, int y, int z) const;

    // Плоский индекс. Принимает и свои ячейки, и гало.
    std::size_t index(int x, int y, int z) const {
        assert(contains_with_halo(x, y, z));
        const std::size_t px = static_cast<std::size_t>(x + halo_);
        const std::size_t py = static_cast<std::size_t>(y + halo_);
        const std::size_t pz = static_cast<std::size_t>(z + halo_);
        return px + static_cast<std::size_t>(sx_) * (py + static_cast<std::size_t>(sy_) * pz);
    }

    float* data(Field field);
    const float* data(Field field) const;

    float& at(Field field, int x, int y, int z) { return data(field)[index(x, y, z)]; }
    float at(Field field, int x, int y, int z) const { return data(field)[index(x, y, z)]; }

    void fill(Field field, float value);
    void fill_all(float value);

    // Сумма по своим ячейкам, гало не учитывается.
    double sum_own(Field field) const;

    // Заполнить гало периодически по всем трём осям. Это случай одного процесса:
    // при декомпозиции по z слой гало по z вместо этого приходит от соседа по MPI.
    void fill_halo_periodic(Field field);
    void fill_halo_periodic();

    // Куски для обмена слоями по z (аргументы для MPI_Sendrecv у Ч3).
    // Каждый кусок непрерывен и содержит z_slab_count() элементов.
    std::size_t z_slab_count() const { return plane_count() * static_cast<std::size_t>(halo_); }

    float* z_send_lo(Field field) { return data(field) + z_plane_offset(0); }
    float* z_send_hi(Field field) { return data(field) + z_plane_offset(nz_ - halo_); }
    float* z_recv_lo(Field field) { return data(field) + z_plane_offset(-halo_); }
    float* z_recv_hi(Field field) { return data(field) + z_plane_offset(nz_); }

    const float* z_send_lo(Field field) const { return data(field) + z_plane_offset(0); }
    const float* z_send_hi(Field field) const { return data(field) + z_plane_offset(nz_ - halo_); }
    const float* z_recv_lo(Field field) const { return data(field) + z_plane_offset(-halo_); }
    const float* z_recv_hi(Field field) const { return data(field) + z_plane_offset(nz_); }

    // Начало плоскости z = const вместе с её гало по x и y.
    std::size_t z_plane_offset(int z) const { return index(-halo_, -halo_, z); }

    static constexpr int kMaxHalo = 2;

private:
    std::vector<float>& field_data(Field field);
    const std::vector<float>& field_data(Field field) const;

    int nx_ = 0;
    int ny_ = 0;
    int nz_ = 0;
    int halo_ = 0;
    int sx_ = 0;
    int sy_ = 0;
    int sz_ = 0;
    std::vector<float> fields_[kFieldCount];
};

}  // namespace svcore
