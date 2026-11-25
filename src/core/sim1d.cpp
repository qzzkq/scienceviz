#include "core/sim1d.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace svcore {
namespace {

std::string trim(const std::string& value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

double number(const std::string& value, const std::string& key) {
    std::size_t end = 0;
    double parsed;
    try { parsed = std::stod(value, &end); }
    catch (const std::exception&) { throw std::invalid_argument("bad number for " + key); }
    if (end != value.size() || !std::isfinite(parsed))
        throw std::invalid_argument("bad number for " + key);
    return parsed;
}

int integer(const std::string& value, const std::string& key) {
    std::size_t end = 0;
    int parsed;
    try { parsed = std::stoi(value, &end); }
    catch (const std::exception&) { throw std::invalid_argument("bad integer for " + key); }
    if (end != value.size()) throw std::invalid_argument("bad integer for " + key);
    return parsed;
}

void validate(const Config1D& c) {
    if (c.cells < 4 || c.cells > 10000000) throw std::invalid_argument("cells must be between 4 and 10000000");
    if (!(c.length > 0 && c.end_time >= 0 && c.cfl > 0 && c.cfl <= 0.5))
        throw std::invalid_argument("length, end_time or cfl out of range");
    if (c.order != 1 && c.order != 2) throw std::invalid_argument("order must be 1 or 2");
    if (c.gas != "dust" && c.gas != "isothermal" && c.gas != "adiabatic")
        throw std::invalid_argument("unknown gas");
    if (c.gamma <= 1 || c.sound_speed <= 0 || c.rho_floor <= 0)
        throw std::invalid_argument("gamma, sound_speed and rho_floor must be positive");
    if (c.boundary != "periodic" && c.boundary != "outflow")
        throw std::invalid_argument("unknown boundary");
    if (c.initial != "sod" && c.initial != "iso" && c.initial != "sine")
        throw std::invalid_argument("unknown initial condition");
    if (c.snapshot_every < 0 || c.checkpoint_every < 0 || c.output_prefix.empty())
        throw std::invalid_argument("invalid output settings");
}

Cell1D add(const Cell1D& a, const Cell1D& b, double scale) {
    return {a.rho + scale * b.rho, a.mom + scale * b.mom,
            a.energy + scale * b.energy};
}

double mc(double left, double right) {
    if (left * right <= 0) return 0;
    return std::copysign(std::min({2 * std::abs(left), 2 * std::abs(right),
                                   0.5 * std::abs(left + right)}), left + right);
}

struct Primitive { double rho, velocity, pressure; };

Primitive shifted(const Primitive& p, const Primitive& slope, double scale) {
    return {p.rho + scale * slope.rho, p.velocity + scale * slope.velocity,
            p.pressure + scale * slope.pressure};
}

template <typename T>
void write_value(std::ofstream& stream, const T& value) {
    stream.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

template <typename T>
void read_value(std::ifstream& stream, T& value) {
    stream.read(reinterpret_cast<char*>(&value), sizeof(value));
}

}  // namespace

Config1D load_config_1d(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open config: " + path);
    Config1D c;
    std::unordered_set<std::string> seen;
    std::string line;
    int line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        line = trim(line.substr(0, line.find('#')));
        if (line.empty()) continue;
        const auto equal = line.find('=');
        if (equal == std::string::npos) throw std::invalid_argument("config line " + std::to_string(line_number) + ": expected key=value");
        const std::string key = trim(line.substr(0, equal));
        const std::string value = trim(line.substr(equal + 1));
        if (!seen.insert(key).second) throw std::invalid_argument("duplicate config key: " + key);
        if (key == "cells") c.cells = integer(value, key);
        else if (key == "length") c.length = number(value, key);
        else if (key == "end_time") c.end_time = number(value, key);
        else if (key == "cfl") c.cfl = number(value, key);
        else if (key == "order") c.order = integer(value, key);
        else if (key == "gas") c.gas = value;
        else if (key == "gamma") c.gamma = number(value, key);
        else if (key == "sound_speed") c.sound_speed = number(value, key);
        else if (key == "rho_floor") c.rho_floor = number(value, key);
        else if (key == "boundary") c.boundary = value;
        else if (key == "initial") c.initial = value;
        else if (key == "snapshot_every") c.snapshot_every = integer(value, key);
        else if (key == "checkpoint_every") c.checkpoint_every = integer(value, key);
        else if (key == "output_prefix") c.output_prefix = value;
        else if (key == "restart") c.restart = value;
        else throw std::invalid_argument("unknown config key: " + key);
    }
    validate(c);
    return c;
}

Sim1D::Sim1D(Config1D config) : config_(std::move(config)), cells_(config_.cells) {
    validate(config_);
    constexpr double pi = 3.14159265358979323846;
    for (int i = 0; i < config_.cells; ++i) {
        const double x = (i + 0.5) / config_.cells;
        double rho = 1.0, p = 1.0, velocity = 0.0;
        if (config_.initial == "sod") {
            if (x >= 0.5) { rho = 0.125; p = 0.1; }
        } else if (config_.initial == "iso") {
            if (x >= 0.5) rho = 0.1;
            p = config_.sound_speed * config_.sound_speed * rho;
        } else {
            rho = 1.0 + 0.1 * std::sin(2 * pi * x);
            p = config_.gas == "adiabatic" ? 1.0 : config_.sound_speed * config_.sound_speed * rho;
        }
        cells_[i] = {rho, rho * velocity,
                     config_.gas == "adiabatic" ? p / (config_.gamma - 1) + 0.5 * rho * velocity * velocity : 0.0};
    }
    if (!config_.restart.empty()) load_checkpoint(config_.restart);
}

double Sim1D::pressure(const Cell1D& cell) const {
    if (config_.gas == "dust") return 0;
    if (config_.gas == "isothermal") return config_.sound_speed * config_.sound_speed * cell.rho;
    const double kinetic = 0.5 * cell.mom * cell.mom / cell.rho;
    return std::max((config_.gamma - 1) * (cell.energy - kinetic),
                    (config_.gamma - 1) * cell.rho * 1e-12);
}

void Sim1D::sync_energy(std::vector<Cell1D>& cells) const {
    if (config_.gas != "adiabatic") return;
    for (auto& cell : cells)
        cell.energy = pressure(cell) / (config_.gamma - 1) + 0.5 * cell.mom * cell.mom / cell.rho;
}

double Sim1D::timestep() const {
    double speed = 0;
    for (const auto& cell : cells_) {
        const double sound = config_.gas == "dust" ? 0 :
            config_.gas == "isothermal" ? config_.sound_speed :
            std::sqrt(config_.gamma * pressure(cell) / cell.rho);
        speed = std::max(speed, std::abs(cell.mom / cell.rho) + sound);
    }
    return speed == 0 ? std::numeric_limits<double>::infinity() :
        config_.cfl * config_.length / config_.cells / speed;
}

std::vector<Cell1D> Sim1D::rhs(const std::vector<Cell1D>& cells) const {
    const int n = config_.cells;
    auto index = [&](int i) {
        if (config_.boundary == "periodic") return (i % n + n) % n;
        return std::max(0, std::min(n - 1, i));
    };
    std::vector<Primitive> w(n);
    for (int i = 0; i < n; ++i)
        w[i] = {cells[i].rho, cells[i].mom / cells[i].rho, pressure(cells[i])};
    std::vector<Primitive> slope(n);
    if (config_.order == 2) {
        for (int i = 0; i < n; ++i) {
            const auto& l = w[index(i - 1)];
            const auto& c = w[i];
            const auto& r = w[index(i + 1)];
            slope[i] = {mc(c.rho - l.rho, r.rho - c.rho),
                        mc(c.velocity - l.velocity, r.velocity - c.velocity),
                        mc(c.pressure - l.pressure, r.pressure - c.pressure)};
        }
    }
    std::vector<Cell1D> flux(n + 1);
    for (int face = 0; face <= n; ++face) {
        const int li = index(face - 1), ri = index(face);
        const Primitive l = shifted(w[li], slope[li], 0.5);
        const Primitive r = shifted(w[ri], slope[ri], -0.5);
        auto conserved = [&](const Primitive& s) {
            return Cell1D{s.rho, s.rho * s.velocity,
                config_.gas == "adiabatic" ? s.pressure / (config_.gamma - 1) + 0.5 * s.rho * s.velocity * s.velocity : 0.0};
        };
        auto physical_flux = [&](const Primitive& s) {
            const double energy = config_.gas == "adiabatic" ?
                s.pressure / (config_.gamma - 1) + 0.5 * s.rho * s.velocity * s.velocity : 0.0;
            return Cell1D{s.rho * s.velocity, s.rho * s.velocity * s.velocity + s.pressure,
                          (energy + s.pressure) * s.velocity};
        };
        auto sound = [&](const Primitive& s) {
            return config_.gas == "dust" ? 0.0 : config_.gas == "isothermal" ?
                config_.sound_speed : std::sqrt(config_.gamma * s.pressure / s.rho);
        };
        const double sl = std::min({l.velocity - sound(l), r.velocity - sound(r), 0.0});
        const double sr = std::max({l.velocity + sound(l), r.velocity + sound(r), 0.0});
        const Cell1D fl = physical_flux(l), fr = physical_flux(r);
        const Cell1D ul = conserved(l), ur = conserved(r);
        const double width = sr - sl > 0 ? sr - sl : 1.0;
        flux[face] = {(sr * fl.rho - sl * fr.rho + sl * sr * (ur.rho - ul.rho)) / width,
                      (sr * fl.mom - sl * fr.mom + sl * sr * (ur.mom - ul.mom)) / width,
                      (sr * fl.energy - sl * fr.energy + sl * sr * (ur.energy - ul.energy)) / width};
    }
    const double inv_dx = n / config_.length;
    std::vector<Cell1D> out(n);
    for (int i = 0; i < n; ++i) {
        out[i] = {(flux[i].rho - flux[i + 1].rho) * inv_dx,
                  (flux[i].mom - flux[i + 1].mom) * inv_dx,
                  (flux[i].energy - flux[i + 1].energy) * inv_dx};
    }
    return out;
}

void Sim1D::step(double dt) {
    if (!(dt > 0 && std::isfinite(dt))) throw std::invalid_argument("invalid timestep");
    const auto k1 = rhs(cells_);
    std::vector<Cell1D> next(cells_.size());
    for (std::size_t i = 0; i < cells_.size(); ++i) {
        next[i] = add(cells_[i], k1[i], dt);
        next[i].rho = std::max(next[i].rho, config_.rho_floor);
    }
    if (config_.order == 2) {
        const auto k2 = rhs(next);
        for (std::size_t i = 0; i < cells_.size(); ++i) {
            const Cell1D second = add(next[i], k2[i], dt);
            next[i] = add(cells_[i], second, 1.0);
            next[i].rho = std::max(0.5 * next[i].rho, config_.rho_floor);
            next[i].mom *= 0.5;
            next[i].energy *= 0.5;
        }
    }
    sync_energy(next);
    cells_.swap(next);
    time_ += dt;
    ++steps_;
}

void Sim1D::run() {
    while (time_ < config_.end_time - 1e-14) {
        const double dt = std::min(timestep(), config_.end_time - time_);
        step(dt);
        if (config_.snapshot_every && steps_ % config_.snapshot_every == 0)
            save_snapshot(config_.output_prefix + "_step" + std::to_string(steps_) + ".csv");
        if (config_.checkpoint_every && steps_ % config_.checkpoint_every == 0)
            save_checkpoint(config_.output_prefix + ".chk");
    }
    save_snapshot(config_.output_prefix + ".csv");
    save_checkpoint(config_.output_prefix + ".chk");
}

double Sim1D::mass() const {
    double sum = 0;
    for (const auto& cell : cells_) sum += cell.rho;
    return sum * config_.length / config_.cells;
}

void Sim1D::save_snapshot(const std::string& path) const {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot write snapshot: " + path);
    out << std::setprecision(17) << "# time=" << time_ << " steps=" << steps_ << '\n';
    out << "x,rho,mom,energy,pressure\n";
    for (int i = 0; i < config_.cells; ++i) {
        const auto& cell = cells_[i];
        out << (i + 0.5) * config_.length / config_.cells << ',' << cell.rho << ','
            << cell.mom << ',' << cell.energy << ',' << pressure(cell) << '\n';
    }
    if (!out) throw std::runtime_error("snapshot write failed: " + path);
}

void Sim1D::save_checkpoint(const std::string& path) const {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("cannot write checkpoint: " + path);
    out.write("SVZ1D002", 8);
    const std::int32_t n = config_.cells, steps = steps_;
    write_value(out, n); write_value(out, steps); write_value(out, time_);
    write_value(out, config_.length); write_value(out, config_.gamma);
    write_value(out, config_.sound_speed);
    const std::int32_t gas = config_.gas == "dust" ? 0 : config_.gas == "isothermal" ? 1 : 2;
    write_value(out, gas);
    const std::int32_t order = config_.order;
    const std::int32_t boundary = config_.boundary == "periodic" ? 0 : 1;
    write_value(out, order); write_value(out, boundary);
    write_value(out, config_.cfl); write_value(out, config_.rho_floor);
    for (const auto& cell : cells_) {
        write_value(out, cell.rho); write_value(out, cell.mom); write_value(out, cell.energy);
    }
    if (!out) throw std::runtime_error("checkpoint write failed: " + path);
}

void Sim1D::load_checkpoint(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open checkpoint: " + path);
    char magic[8]{};
    in.read(magic, 8);
    if (std::string(magic, 8) != "SVZ1D002") throw std::runtime_error("bad checkpoint header");
    std::int32_t n, steps, gas, order, boundary;
    double time, length, gamma, sound_speed, cfl, rho_floor;
    read_value(in, n); read_value(in, steps); read_value(in, time);
    read_value(in, length); read_value(in, gamma); read_value(in, sound_speed);
    read_value(in, gas);
    read_value(in, order); read_value(in, boundary);
    read_value(in, cfl); read_value(in, rho_floor);
    const int expected_gas = config_.gas == "dust" ? 0 : config_.gas == "isothermal" ? 1 : 2;
    const int expected_boundary = config_.boundary == "periodic" ? 0 : 1;
    if (!in || n != config_.cells || length != config_.length || gamma != config_.gamma ||
        sound_speed != config_.sound_speed || gas != expected_gas ||
        order != config_.order || boundary != expected_boundary ||
        cfl != config_.cfl || rho_floor != config_.rho_floor || steps < 0 ||
        !std::isfinite(time) || time < 0 || time > config_.end_time)
        throw std::runtime_error("checkpoint does not match config");
    std::vector<Cell1D> loaded(n);
    for (auto& cell : loaded) {
        read_value(in, cell.rho); read_value(in, cell.mom); read_value(in, cell.energy);
        if (!in || !std::isfinite(cell.rho) || cell.rho <= 0 || !std::isfinite(cell.mom) ||
            !std::isfinite(cell.energy)) throw std::runtime_error("truncated or invalid checkpoint");
    }
    char extra;
    if (in.read(&extra, 1)) throw std::runtime_error("checkpoint has trailing bytes");
    cells_.swap(loaded);
    time_ = time;
    steps_ = steps;
}

}  // namespace svcore
