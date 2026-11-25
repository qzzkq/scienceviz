#pragma once

#include <string>
#include <vector>

namespace svcore {

struct Config1D {
    int cells = 200;
    double length = 1.0;
    double end_time = 0.2;
    double cfl = 0.4;
    int order = 2;
    std::string gas = "adiabatic";
    double gamma = 1.4;
    double sound_speed = 1.0;
    double rho_floor = 1e-10;
    std::string boundary = "outflow";
    std::string initial = "sod";
    int snapshot_every = 0;
    int checkpoint_every = 0;
    std::string output_prefix = "sim1d";
    std::string restart;
};

Config1D load_config_1d(const std::string& path);

struct Cell1D {
    double rho = 0.0;
    double mom = 0.0;
    double energy = 0.0;
};

class Sim1D {
public:
    explicit Sim1D(Config1D config);
    void step(double dt);
    void run();
    void save_snapshot(const std::string& path) const;
    void save_checkpoint(const std::string& path) const;
    void load_checkpoint(const std::string& path);
    double timestep() const;
    double mass() const;
    double time() const { return time_; }
    int steps() const { return steps_; }
    const std::vector<Cell1D>& cells() const { return cells_; }

private:
    Config1D config_;
    std::vector<Cell1D> cells_;
    double time_ = 0.0;
    int steps_ = 0;

    double pressure(const Cell1D& cell) const;
    void sync_energy(std::vector<Cell1D>& cells) const;
    std::vector<Cell1D> rhs(const std::vector<Cell1D>& cells) const;
};

}  // namespace svcore
