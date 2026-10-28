#pragma once

#include <cstdint>
#include <string>
#include <vector>

// Плотный куб плотности N³, float32. Индекс x + N * (y + N * z): x самый быстрый.
class Volume {
public:
    Volume() = default;
    explicit Volume(int n);

    int size() const { return n_; }
    bool empty() const { return n_ == 0; }

    float at(int x, int y, int z) const;
    void set(int x, int y, int z, float value);

    // Мировые координаты в [0, 1]. В центре вокселя совпадает с его значением.
    float sample(float x, float y, float z) const;

    std::vector<float>& data() { return data_; }
    const std::vector<float>& data() const { return data_; }

    // "SVZCUBE1", uint32 N (LE), затем N³ float32 (LE).
    [[nodiscard]] static bool save_f32(const Volume& volume, const std::string& path, std::string& error);
    [[nodiscard]] static bool load_f32(const std::string& path, Volume& volume, std::string& error);

    static constexpr int kMaxSize = 512;

private:
    int n_ = 0;
    std::vector<float> data_;
};
