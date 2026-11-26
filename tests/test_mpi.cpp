#include "mpi/halo.hpp"
#include "mpi/slab.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
int failed = 0;
int rank = 0;
int ranks = 1;

void check(bool ok, const char* what) {
    if (!ok) {
        ++failed;
        std::cerr << "rank " << rank << ": " << what << '\n';
    }
}

int wrap(int i, int n) { return (i % n + n) % n; }

float marker(int x, int y, int z, int field) {
    return static_cast<float>(x + 32 * y + 1024 * z + 100000 * field);
}

void test_slabs() {
    for (int halo : {1, 2}) {
        for (int nz : {6, 7, 17, 256}) {
            int end = 0;
            for (int r = 0; r < ranks; ++r) {
                const auto slab = svmpi::split_z(nz, ranks, r, halo);
                check(slab.z0 == end && slab.nz >= halo, "slabs cover domain without gaps");
                end += slab.nz;
            }
            check(end == nz, "slabs cover all z planes");
        }
    }
    check(svmpi::split_z(256, 3, 0, 2).nz == 86, "256/3 first slab = 86");
    check(svmpi::split_z(256, 3, 2, 2).z0 == 171, "256/3 last offset = 171");
    bool rejected = false;
    try { (void)svmpi::split_z(5, 3, 0, 2); }
    catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "reject blocks thinner than halo");
}

void test_halo(int nz, int halo) {
    constexpr int nx = 5, ny = 4;
    const auto slab = svmpi::split_z(nz, ranks, rank, halo);
    svcore::Grid3D grid(nx, ny, slab.nz, halo);
    std::vector<svcore::Field> fields;
    for (int i = 0; i < svcore::kFieldCount; ++i) {
        fields.push_back(static_cast<svcore::Field>(i));
    }
    auto exchange_fields = fields;
    if (rank % 2) std::reverse(exchange_fields.begin(), exchange_fields.end());
    svmpi::HaloExchange exchange(grid, exchange_fields);
    // Повторный обмен выявляет использование старого содержимого гало.
    for (int pass = 0; pass < 2; ++pass) {
        grid.fill_all(std::numeric_limits<float>::quiet_NaN());
        for (int f = 0; f < svcore::kFieldCount; ++f) {
            for (int z = 0; z < slab.nz; ++z) {
                for (int y = 0; y < ny; ++y) {
                    for (int x = 0; x < nx; ++x) {
                        grid.at(fields[f], x, y, z) = marker(x, y, slab.z0 + z, f) + pass;
                    }
                }
            }
        }
        exchange.exchange();
        int bad = 0;
        for (int f = 0; f < svcore::kFieldCount; ++f) {
            for (int z = -halo; z < slab.nz + halo; ++z) {
                for (int y = -halo; y < ny + halo; ++y) {
                    for (int x = -halo; x < nx + halo; ++x) {
                        const float expected = marker(wrap(x, nx), wrap(y, ny),
                            wrap(slab.z0 + z, nz), f) + pass;
                        bad += grid.at(fields[f], x, y, z) != expected;
                    }
                }
            }
        }
        check(bad == 0, "own cells, halo faces, edges and corners match global markers");
    }
}

void test_subset_and_validation() {
    svcore::Grid3D grid(4, 4, 4, 1);
    grid.fill_all(-9.0f);
    for (int z = 0; z < 4; ++z)
        for (int y = 0; y < 4; ++y)
            for (int x = 0; x < 4; ++x) grid.at(svcore::Field::Phi, x, y, z) = 7.0f;
    {
        svmpi::HaloExchange exchange(grid, {svcore::Field::Phi});
        exchange.exchange();
        check(grid.at(svcore::Field::Phi, -1, -1, -1) == 7.0f, "selected field exchanged");
        check(grid.at(svcore::Field::Rho, -1, -1, -1) == -9.0f, "other fields untouched");
    }
    bool rejected = false;
    try {
        const auto bad = static_cast<svcore::Field>(99);
        svmpi::HaloExchange exchange(grid, {rank == 0 ? bad : svcore::Field::Rho});
    } catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "invalid field rejected collectively");
    if (ranks > 1) {
        rejected = false;
        try {
            svmpi::HaloExchange exchange(grid,
                {rank == 0 ? svcore::Field::Phi : svcore::Field::Rho});
        } catch (const std::invalid_argument&) { rejected = true; }
        check(rejected, "inconsistent field lists rejected collectively");
    }
}

void test_dt() {
    const double local = rank == ranks - 1 ? 0.125 : 0.5;
    check(svmpi::global_time_step(local) == 0.125, "all ranks use smallest dt");
    for (double bad : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                       std::numeric_limits<double>::quiet_NaN()}) {
        bool rejected = false;
        try { (void)svmpi::global_time_step(rank == 0 ? bad : 1.0); }
        catch (const std::invalid_argument&) { rejected = true; }
        check(rejected, "invalid dt rejected on every rank");
    }
}
}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    try {
        test_slabs();
        for (int halo : {1, 2}) {
            test_halo(6, halo);  // В том числе два слоя на ранг при ng=2.
            test_halo(17, halo); // Неравномерное разбиение на 2 и 3 ранга.
        }
        test_subset_and_validation();
        test_dt();
    } catch (const std::exception& e) {
        std::cerr << "rank " << rank << ": " << e.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int total = 0;
    MPI_Allreduce(&failed, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) std::cout << "MPI ranks=" << ranks << " failures=" << total << '\n';
    MPI_Finalize();
    return total == 0 ? 0 : 1;
}
