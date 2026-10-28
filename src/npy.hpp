#pragma once

#include "volume.hpp"

#include <string>

// Один куб float32, порядок C, shape (N, N, N). Стек из многих кубов (4D) отвергается.
[[nodiscard]] bool load_npy(const std::string& path, Volume& volume, std::string& error);
