#pragma once

#include "volume.hpp"

inline constexpr float kNodeAx = 0.32f;
inline constexpr float kNodeBx = 0.68f;
inline constexpr float kNodeY = 0.50f;
inline constexpr float kNodeZ = 0.50f;

float synthetic_density(float x, float y, float z);
void fill_synthetic(Volume& volume);
