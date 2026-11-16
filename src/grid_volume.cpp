#include "grid_volume.hpp"

#include <stdexcept>

void density_to_grid(const Volume& volume, svcore::Grid3D& grid) {
    const int n = volume.size();
    if (n == 0 || grid.nx() != n || grid.ny() != n || grid.nz() != n) {
        throw std::invalid_argument("размеры куба и сетки не совпадают");
    }
    for (int z = 0; z < n; ++z) {
        for (int y = 0; y < n; ++y) {
            for (int x = 0; x < n; ++x) {
                grid.at(svcore::Field::Rho, x, y, z) = volume.at(x, y, z);
            }
        }
    }
    grid.fill_halo_periodic(svcore::Field::Rho);
}

Volume density_to_volume(const svcore::Grid3D& grid) {
    const int n = grid.nx();
    if (n == 0 || grid.ny() != n || grid.nz() != n) {
        throw std::invalid_argument("для рендера нужна непустая кубическая сетка");
    }
    Volume volume(n);
    for (int z = 0; z < n; ++z) {
        for (int y = 0; y < n; ++y) {
            for (int x = 0; x < n; ++x) {
                volume.set(x, y, z, grid.at(svcore::Field::Rho, x, y, z));
            }
        }
    }
    return volume;
}
