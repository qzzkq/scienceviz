#include "grid_volume.hpp"

#include <iostream>
#include <stdexcept>

int main() {
    constexpr int n = 3;
    Volume original(n);
    for (int z = 0; z < n; ++z)
        for (int y = 0; y < n; ++y)
            for (int x = 0; x < n; ++x)
                original.set(x, y, z, static_cast<float>(x + 10 * y + 100 * z));

    svcore::Grid3D grid(n, n, n, 1);
    density_to_grid(original, grid);
    if (grid.at(svcore::Field::Rho, -1, 0, 0) != original.at(2, 0, 0) ||
        grid.at(svcore::Field::Rho, 0, -1, 0) != original.at(0, 2, 0) ||
        grid.at(svcore::Field::Rho, 0, 0, -1) != original.at(0, 0, 2)) {
        std::cerr << "неверное периодическое гало\n";
        return 1;
    }
    const Volume restored = density_to_volume(grid);
    if (original.data() != restored.data()) {
        std::cerr << "плотность изменилась при переносе через сетку\n";
        return 1;
    }
    try {
        svcore::Grid3D wrong(2, 2, 2, 1);
        density_to_grid(original, wrong);
        std::cerr << "несовпадающие размеры не отклонены\n";
        return 1;
    } catch (const std::invalid_argument&) {
    }
    return 0;
}
