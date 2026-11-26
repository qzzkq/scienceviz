#pragma once

#include <algorithm>
#include <stdexcept>

namespace svmpi {

struct Slab {
    int nz;
    int z0;
};

// Первые global_nz % ranks процессов получают на одну плоскость больше.
inline Slab split_z(int global_nz, int ranks, int rank, int halo) {
    if (global_nz <= 0 || ranks <= 0 || rank < 0 || rank >= ranks || halo < 1
        || global_nz / ranks < halo) {
        throw std::invalid_argument("split_z: every rank needs at least halo z planes");
    }
    const int base = global_nz / ranks;
    const int extra = global_nz % ranks;
    return {base + (rank < extra ? 1 : 0), rank * base + std::min(rank, extra)};
}

}  // namespace svmpi
