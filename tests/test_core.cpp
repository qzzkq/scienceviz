#include "core/grid.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using svcore::Field;
using svcore::Grid3D;

namespace {

int g_failed = 0;

void check(bool ok, const char* expr, int line) {
    if (!ok) {
        std::cerr << "fail " << line << ": " << expr << "\n";
        ++g_failed;
    }
}

#define CHECK(expr) check(static_cast<bool>(expr), #expr, __LINE__)

// Разные значения в каждой своей ячейке, чтобы перепутанные оси были видны.
float marker(int x, int y, int z) {
    return static_cast<float>(1 + x + 100 * y + 10000 * z);
}

int wrap(int i, int n) {
    return ((i % n) + n) % n;
}

void fill_own_markers(Grid3D& grid, Field field) {
    for (int z = 0; z < grid.nz(); ++z) {
        for (int y = 0; y < grid.ny(); ++y) {
            for (int x = 0; x < grid.nx(); ++x) {
                grid.at(field, x, y, z) = marker(x, y, z);
            }
        }
    }
}

void test_indexing() {
    const Grid3D grid(4, 3, 5, 1);

    CHECK(grid.nx() == 4);
    CHECK(grid.ny() == 3);
    CHECK(grid.nz() == 5);
    CHECK(grid.halo() == 1);
    CHECK(grid.sx() == 6);
    CHECK(grid.sy() == 5);
    CHECK(grid.sz() == 7);
    CHECK(grid.own_count() == 60u);
    CHECK(grid.total_count() == 6u * 5u * 7u);
    CHECK(grid.plane_count() == 30u);

    // x самый быстрый, как в Volume у рендера.
    CHECK(grid.stride_x() == 1u);
    CHECK(grid.index(1, 0, 0) - grid.index(0, 0, 0) == grid.stride_x());
    CHECK(grid.index(0, 1, 0) - grid.index(0, 0, 0) == grid.stride_y());
    CHECK(grid.index(0, 0, 1) - grid.index(0, 0, 0) == grid.stride_z());
    CHECK(grid.stride_y() == 6u);
    CHECK(grid.stride_z() == 30u);

    // Первая своя ячейка смещена на гало по всем осям, последняя попадает в конец.
    CHECK(grid.index(0, 0, 0) == 1u + 6u + 30u);
    CHECK(grid.index(-1, -1, -1) == 0u);
    CHECK(grid.index(grid.nx(), grid.ny(), grid.nz()) == grid.total_count() - 1u);

    // Границы названы явно.
    CHECK(grid.contains_own(0, 0, 0));
    CHECK(grid.contains_own(3, 2, 4));
    CHECK(!grid.contains_own(-1, 0, 0));
    CHECK(!grid.contains_own(4, 0, 0));
    CHECK(!grid.contains_own(0, 0, 5));
    CHECK(grid.contains_with_halo(-1, -1, -1));
    CHECK(grid.contains_with_halo(4, 3, 5));
    CHECK(!grid.contains_with_halo(-2, 0, 0));
    CHECK(!grid.contains_with_halo(0, 0, 6));

    // Своя ячейка не пересекается ни с одной другой.
    Grid3D probe(4, 3, 5, 2);
    std::vector<int> hits(probe.total_count(), 0);
    for (int z = -2; z < probe.nz() + 2; ++z) {
        for (int y = -2; y < probe.ny() + 2; ++y) {
            for (int x = -2; x < probe.nx() + 2; ++x) {
                ++hits[probe.index(x, y, z)];
            }
        }
    }
    bool once = true;
    for (std::size_t i = 0; i < hits.size(); ++i) {
        once = once && hits[i] == 1;
    }
    CHECK(once);
}

void test_fields_are_separate() {
    Grid3D grid(3, 3, 3, 1);
    CHECK(svcore::kFieldCount == 5);

    grid.fill(Field::Rho, 1.0f);
    grid.fill(Field::MomX, 2.0f);
    grid.fill(Field::MomY, 3.0f);
    grid.fill(Field::MomZ, 4.0f);
    grid.fill(Field::Phi, 5.0f);

    CHECK(grid.at(Field::Rho, 1, 1, 1) == 1.0f);
    CHECK(grid.at(Field::MomX, 1, 1, 1) == 2.0f);
    CHECK(grid.at(Field::MomY, 1, 1, 1) == 3.0f);
    CHECK(grid.at(Field::MomZ, 1, 1, 1) == 4.0f);
    CHECK(grid.at(Field::Phi, 1, 1, 1) == 5.0f);

    // Указатель на поле — плоский массив float, без обёрток: годится для GPU.
    float* rho = grid.data(Field::Rho);
    rho[grid.index(0, 0, 0)] = 42.0f;
    CHECK(grid.at(Field::Rho, 0, 0, 0) == 42.0f);
    CHECK(grid.at(Field::MomX, 0, 0, 0) == 2.0f);

    CHECK(std::string(svcore::field_name(Field::Rho)) == "rho");
    CHECK(std::string(svcore::field_name(Field::MomZ)) == "mom_z");
}

// Гало по всем осям, включая рёбра и углы, равно периодическому продолжению.
void check_periodic(int nx, int ny, int nz, int halo) {
    Grid3D grid(nx, ny, nz, halo);
    fill_own_markers(grid, Field::Rho);
    grid.fill_halo_periodic(Field::Rho);

    int bad = 0;
    int checked_halo = 0;
    for (int z = -halo; z < nz + halo; ++z) {
        for (int y = -halo; y < ny + halo; ++y) {
            for (int x = -halo; x < nx + halo; ++x) {
                const float want = marker(wrap(x, nx), wrap(y, ny), wrap(z, nz));
                if (grid.at(Field::Rho, x, y, z) != want) {
                    ++bad;
                }
                if (!grid.contains_own(x, y, z)) {
                    ++checked_halo;
                }
            }
        }
    }
    CHECK(bad == 0);
    CHECK(checked_halo > 0);
}

void test_periodic_halo() {
    check_periodic(4, 3, 5, 1);
    check_periodic(4, 3, 5, 2);
    check_periodic(2, 2, 2, 1);
    check_periodic(2, 2, 2, 2);
    check_periodic(8, 8, 8, 1);
    check_periodic(8, 8, 8, 2);
    check_periodic(5, 4, 2, 2);
}

void test_periodic_halo_all_fields() {
    Grid3D grid(4, 4, 4, 2);
    for (int i = 0; i < svcore::kFieldCount; ++i) {
        fill_own_markers(grid, static_cast<Field>(i));
    }
    grid.fill_halo_periodic();

    int bad = 0;
    for (int i = 0; i < svcore::kFieldCount; ++i) {
        const Field field = static_cast<Field>(i);
        for (int z = -2; z < 6; ++z) {
            for (int y = -2; y < 6; ++y) {
                for (int x = -2; x < 6; ++x) {
                    const float want = marker(wrap(x, 4), wrap(y, 4), wrap(z, 4));
                    if (grid.at(field, x, y, z) != want) {
                        ++bad;
                    }
                }
            }
        }
    }
    CHECK(bad == 0);
}

void test_sum_own_skips_halo() {
    for (int halo = 1; halo <= 2; ++halo) {
        Grid3D grid(4, 3, 5, halo);
        grid.fill(Field::Rho, 0.0f);
        for (int z = 0; z < grid.nz(); ++z) {
            for (int y = 0; y < grid.ny(); ++y) {
                for (int x = 0; x < grid.nx(); ++x) {
                    grid.at(Field::Rho, x, y, z) = 1.0f;
                }
            }
        }
        CHECK(grid.sum_own(Field::Rho) == 60.0);

        // Мусор в гало на сумму не влияет.
        grid.fill_halo_periodic(Field::Rho);
        CHECK(grid.sum_own(Field::Rho) == 60.0);

        for (int z = -halo; z < grid.nz() + halo; ++z) {
            for (int y = -halo; y < grid.ny() + halo; ++y) {
                for (int x = -halo; x < grid.nx() + halo; ++x) {
                    if (!grid.contains_own(x, y, z)) {
                        grid.at(Field::Rho, x, y, z) = 1000.0f;
                    }
                }
            }
        }
        CHECK(grid.sum_own(Field::Rho) == 60.0);
    }
}

// Слой гало по z — один непрерывный кусок: это аргумент для MPI_Sendrecv.
void test_z_slabs_are_contiguous() {
    for (int halo = 1; halo <= 2; ++halo) {
        Grid3D grid(4, 3, 5, halo);
        const std::size_t plane = grid.plane_count();
        CHECK(grid.z_slab_count() == plane * static_cast<std::size_t>(halo));

        // Индексы внутри слоя идут подряд, без дыр.
        int gaps = 0;
        std::size_t expected = grid.z_plane_offset(-halo);
        for (int z = -halo; z < 0; ++z) {
            for (int y = -halo; y < grid.ny() + halo; ++y) {
                for (int x = -halo; x < grid.nx() + halo; ++x) {
                    if (grid.index(x, y, z) != expected) {
                        ++gaps;
                    }
                    ++expected;
                }
            }
        }
        CHECK(gaps == 0);

        const float* base = grid.data(Field::Rho);
        CHECK(grid.z_recv_lo(Field::Rho) == base + grid.index(-halo, -halo, -halo));
        CHECK(grid.z_send_lo(Field::Rho) == base + grid.index(-halo, -halo, 0));
        CHECK(grid.z_send_hi(Field::Rho) == base + grid.index(-halo, -halo, grid.nz() - halo));
        CHECK(grid.z_recv_hi(Field::Rho) == base + grid.index(-halo, -halo, grid.nz()));

        // Верхний слой гало заканчивается ровно на конце массива.
        CHECK(grid.z_recv_hi(Field::Rho) + grid.z_slab_count() == base + grid.total_count());

        // Своё и принятое не перекрываются.
        CHECK(grid.z_send_lo(Field::Rho) == grid.z_recv_lo(Field::Rho) + grid.z_slab_count());
        CHECK(grid.z_recv_hi(Field::Rho) == grid.z_send_hi(Field::Rho) + grid.z_slab_count());

        // Копирование слоя как одного куска даёт то же, что периодическое гало.
        fill_own_markers(grid, Field::Rho);
        grid.fill_halo_periodic(Field::Rho);
        int bad = 0;
        const float* send_hi = grid.z_send_hi(Field::Rho);
        const float* recv_lo = grid.z_recv_lo(Field::Rho);
        for (std::size_t i = 0; i < grid.z_slab_count(); ++i) {
            if (send_hi[i] != recv_lo[i]) {
                ++bad;
            }
        }
        CHECK(bad == 0);
    }
}

void test_rejects_bad_dims() {
    bool threw = false;
    try {
        Grid3D bad(4, 4, 4, 0);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);

    threw = false;
    try {
        Grid3D bad(4, 4, 4, 3);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);

    threw = false;
    try {
        Grid3D bad(0, 4, 4, 1);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);

    // Блок тоньше гало: периодическое гало и обмен по z так не работают.
    threw = false;
    try {
        Grid3D bad(4, 4, 1, 2);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

}  // namespace

int main() {
    test_indexing();
    test_fields_are_separate();
    test_periodic_halo();
    test_periodic_halo_all_fields();
    test_sum_own_skips_halo();
    test_z_slabs_are_contiguous();
    test_rejects_bad_dims();

    if (g_failed != 0) {
        std::cerr << g_failed << " проверок упало\n";
        return 1;
    }
    std::cout << "svcore tests ok\n";
    return 0;
}
