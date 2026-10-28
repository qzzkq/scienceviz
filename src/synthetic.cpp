#include "synthetic.hpp"

#include <cmath>

namespace {

float blob(float dx, float dy, float dz, float sigma) {
    const float r2 = dx * dx + dy * dy + dz * dz;
    return std::exp(-r2 / (2.f * sigma * sigma));
}

}  // namespace

float synthetic_density(float x, float y, float z) {
    const float node_a = blob(x - kNodeAx, y - kNodeY, z - kNodeZ, 0.055f);
    const float node_b = blob(x - kNodeBx, y - kNodeY, z - kNodeZ, 0.055f);

    const float span = kNodeBx - kNodeAx;
    float along = (x - kNodeAx) / span;
    if (along < 0.f) {
        along = 0.f;
    }
    if (along > 1.f) {
        along = 1.f;
    }
    const float closest_x = kNodeAx + span * along;
    const float tube = blob(x - closest_x, y - kNodeY, z - kNodeZ, 0.030f);
    return 12.f * node_a + 12.f * node_b + 4.5f * tube;
}

void fill_synthetic(Volume& volume) {
    const int n = volume.size();
    for (int z = 0; z < n; ++z) {
        for (int y = 0; y < n; ++y) {
            for (int x = 0; x < n; ++x) {
                const float px = (static_cast<float>(x) + 0.5f) / static_cast<float>(n);
                const float py = (static_cast<float>(y) + 0.5f) / static_cast<float>(n);
                const float pz = (static_cast<float>(z) + 0.5f) / static_cast<float>(n);
                volume.set(x, y, z, synthetic_density(px, py, pz));
            }
        }
    }
}
