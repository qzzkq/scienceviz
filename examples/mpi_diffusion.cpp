#include "mpi/halo.hpp"
#include "mpi/slab.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr double pi = 3.14159265358979323846;
constexpr auto rho = svcore::Field::Rho;

struct Options {
    int nx = 24, ny = 24, nz = 24, halo = 1, steps = 20;
    std::string output;
};

int integer(const std::string& value) {
    std::size_t end;
    const int number = std::stoi(value, &end);
    if (end != value.size()) throw std::invalid_argument("expected integer: " + value);
    return number;
}

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (++i >= argc) throw std::invalid_argument("missing value for " + key);
        const std::string value = argv[i];
        if (key == "--output") o.output = value;
        else if (key == "--nx") o.nx = integer(value);
        else if (key == "--ny") o.ny = integer(value);
        else if (key == "--nz") o.nz = integer(value);
        else if (key == "--size") o.nx = o.ny = o.nz = integer(value);
        else if (key == "--halo") o.halo = integer(value);
        else if (key == "--steps") o.steps = integer(value);
        else throw std::invalid_argument("unknown option: " + key);
    }
    // Прототип с полной последовательной проверкой на rank 0, не production solver.
    if (o.nx < o.halo || o.ny < o.halo || o.nz < o.halo || o.halo < 1 || o.halo > 2
        || o.nx > 128 || o.ny > 128 || o.nz > 128 || o.steps < 0 || o.steps > 10000) {
        throw std::invalid_argument("sizes must be in [halo,128], halo=1/2, steps in [0,10000]");
    }
    return o;
}

std::size_t index(int x, int y, int z, const Options& o) {
    return static_cast<std::size_t>(x) + static_cast<std::size_t>(o.nx) * (y + o.ny * z);
}

double mode(int x, int n, bool cosine) {
    const double phase = 2.0 * pi * x / n;
    return cosine ? std::cos(phase) : std::sin(phase);
}

float initial(int x, int y, int z, const Options& o) {
    return static_cast<float>(1.0 + 0.1 * (mode(x, o.nx, false)
        + 0.7 * mode(y, o.ny, true) + 0.3 * mode(z, o.nz, false)));
}

// Независимый последовательный путь: нет Grid3D и нет гало, соседи через modulo.
std::vector<float> serial_reference(const Options& o, double dt) {
    std::vector<float> u(static_cast<std::size_t>(o.nx) * o.ny * o.nz);
    auto next = u;
    for (int z = 0; z < o.nz; ++z)
        for (int y = 0; y < o.ny; ++y)
            for (int x = 0; x < o.nx; ++x) u[index(x, y, z, o)] = initial(x, y, z, o);
    for (int step = 0; step < o.steps; ++step) {
        for (int z = 0; z < o.nz; ++z) {
            for (int y = 0; y < o.ny; ++y) {
                for (int x = 0; x < o.nx; ++x) {
                    const auto i = index(x, y, z, o);
                    const double c = u[i];
                    const double lap = o.nx * o.nx * (double(u[index((x+1)%o.nx,y,z,o)])
                        - 2*c + u[index((x+o.nx-1)%o.nx,y,z,o)])
                        + o.ny * o.ny * (double(u[index(x,(y+1)%o.ny,z,o)])
                        - 2*c + u[index(x,(y+o.ny-1)%o.ny,z,o)])
                        + o.nz * o.nz * (double(u[index(x,y,(z+1)%o.nz,o)])
                        - 2*c + u[index(x,y,(z+o.nz-1)%o.nz,o)]);
                    next[i] = static_cast<float>(c + dt * lap);
                }
            }
        }
        u.swap(next);
    }
    return u;
}

int run(int argc, char** argv) {
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    const Options o = parse(argc, argv);
    const auto slab = svmpi::split_z(o.nz, ranks, rank, o.halo);
    svcore::Grid3D grid(o.nx, o.ny, slab.nz, o.halo);
    for (int z = 0; z < slab.nz; ++z)
        for (int y = 0; y < o.ny; ++y)
            for (int x = 0; x < o.nx; ++x) grid.at(rho, x, y, z) = initial(x,y,z+slab.z0,o);
    svmpi::HaloExchange exchange(grid, {rho});
    // u_t = Laplacian(u), периодический единичный куб; явная схема, D=1.
    // Условие устойчивости dt <= 1/(2*(nx²+ny²+nz²)). Запас 10%.
    const double dt = svmpi::global_time_step(0.45 / (o.nx*o.nx + o.ny*o.ny + o.nz*o.nz));
    std::vector<float> next(grid.own_count());
    double exchange_seconds = 0, compute_seconds = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (int step = 0; step < o.steps; ++step) {
        const double halo_start = MPI_Wtime();
        exchange.exchange();
        const double compute_start = MPI_Wtime();
        exchange_seconds += compute_start - halo_start;
        for (int z = 0; z < slab.nz; ++z) {
            for (int y = 0; y < o.ny; ++y) {
                for (int x = 0; x < o.nx; ++x) {
                    const double c = grid.at(rho, x, y, z);
                    const double lap = o.nx * o.nx * (double(grid.at(rho,x+1,y,z))
                        - 2*c + grid.at(rho,x-1,y,z))
                        + o.ny * o.ny * (double(grid.at(rho,x,y+1,z))
                        - 2*c + grid.at(rho,x,y-1,z))
                        + o.nz * o.nz * (double(grid.at(rho,x,y,z+1))
                        - 2*c + grid.at(rho,x,y,z-1));
                    next[index(x,y,z,o)] = static_cast<float>(c + dt * lap);
                }
            }
        }
        for (int z = 0; z < slab.nz; ++z)
            for (int y = 0; y < o.ny; ++y)
                for (int x = 0; x < o.nx; ++x) grid.at(rho,x,y,z) = next[index(x,y,z,o)];
        compute_seconds += MPI_Wtime() - compute_start;
    }
    const double elapsed = MPI_Wtime() - start;
    double times[] = {compute_seconds, exchange_seconds, elapsed};
    double maxima[3];
    MPI_Reduce(times, maxima, 3, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<float> packed(grid.own_count());
    for (int z = 0; z < slab.nz; ++z)
        for (int y = 0; y < o.ny; ++y)
            for (int x = 0; x < o.nx; ++x) packed[index(x,y,z,o)] = grid.at(rho,x,y,z);
    std::vector<int> counts(ranks), offsets(ranks);
    for (int r = 0; r < ranks; ++r) {
        const auto part = svmpi::split_z(o.nz,ranks,r,o.halo);
        counts[r] = o.nx * o.ny * part.nz;
        offsets[r] = o.nx * o.ny * part.z0;
    }
    std::vector<float> global(rank == 0 ? static_cast<std::size_t>(o.nx)*o.ny*o.nz : 0);
    svmpi::check(MPI_Gatherv(packed.data(), counts[rank], MPI_FLOAT, global.data(),
                            counts.data(), offsets.data(), MPI_FLOAT, 0, MPI_COMM_WORLD),
                 "gather verification field");
    int ok = 1;
    if (rank == 0) {
        const auto reference = serial_reference(o,dt);
        double max_error = 0, mode_error = 0, initial_mass = 0, final_mass = 0;
        // Точный множитель затухания каждой моды именно дискретной схемы.
        auto decay = [&](int n) {
            const double s = std::sin(pi/n);
            return std::pow(1.0 - 4.0*dt*n*n*s*s, o.steps);
        };
        const double dx = decay(o.nx), dy = decay(o.ny), dz = decay(o.nz);
        for (int z = 0; z < o.nz; ++z) {
            for (int y = 0; y < o.ny; ++y) {
                for (int x = 0; x < o.nx; ++x) {
                    const auto i = index(x,y,z,o);
                    ok &= std::isfinite(global[i]);
                    max_error = std::max(max_error,std::abs(double(global[i])-reference[i]));
                    const double exact = 1.0 + 0.1*(dx*mode(x,o.nx,false)
                        + 0.7*dy*mode(y,o.ny,true) + 0.3*dz*mode(z,o.nz,false));
                    mode_error = std::max(mode_error,std::abs(global[i]-exact));
                    initial_mass += initial(x,y,z,o);
                    final_mass += global[i];
                }
            }
        }
        const double mass_drift = std::abs(final_mass-initial_mass)/initial_mass;
        ok &= max_error <= 1e-6 && mode_error <= 5e-6 && mass_drift <= 1e-6;
        if (!o.output.empty()) {
            std::ofstream out(o.output,std::ios::binary);
            out.write(reinterpret_cast<const char*>(global.data()),
                      static_cast<std::streamsize>(global.size()*sizeof(float)));
            if (!out) throw std::runtime_error("cannot write " + o.output);
        }
        std::cout << std::setprecision(12)
            << "{\"ranks\":" << ranks << ",\"nx\":" << o.nx << ",\"ny\":" << o.ny
            << ",\"nz\":" << o.nz << ",\"halo\":" << o.halo << ",\"steps\":" << o.steps
            << ",\"dt\":" << dt << ",\"max_abs_error_serial\":" << max_error
            << ",\"max_abs_error_mode\":" << mode_error << ",\"relative_mass_drift\":" << mass_drift
            << ",\"compute_s_max\":" << maxima[0] << ",\"halo_s_max\":" << maxima[1]
            << ",\"loop_s_max\":" << maxima[2] << ",\"verified\":" << (ok ? "true" : "false")
            << "}\n";
    }
    MPI_Bcast(&ok,1,MPI_INT,0,MPI_COMM_WORLD);
    return ok ? 0 : 1;
}
}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc,&argv);
    int result = 1;
    try { result = run(argc,argv); }
    catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD,1);
    }
    MPI_Finalize();
    return result;
}
