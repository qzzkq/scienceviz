#pragma once

#include "core/grid.hpp"
#include "volume.hpp"

// Переносит только свои ячейки, без гало, сохраняя координаты x, y, z.
void density_to_grid(const Volume& volume, svcore::Grid3D& grid);
Volume density_to_volume(const svcore::Grid3D& grid);
