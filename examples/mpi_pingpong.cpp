#include <mpi.h>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (ranks != 2) {
        if (rank == 0) std::cerr << "mpi_pingpong requires exactly 2 ranks\n";
        MPI_Finalize();
        return 1;
    }
    char host[MPI_MAX_PROCESSOR_NAME] = {};
    int length;
    MPI_Get_processor_name(host, &length);
    char peer[MPI_MAX_PROCESSOR_NAME] = {};
    MPI_Sendrecv(host, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 1 - rank, 9,
                 peer, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 1 - rank, 9,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (rank == 0) {
        std::cout << "# rank0=" << host << " rank1=" << peer << '\n';
        std::cout << "# one_way_us = round_trip / 2; throughput = payload / one_way\n";
        std::cout << "bytes,iterations,one_way_us,payload_MB_s\n" << std::setprecision(9);
    }
    int failed = 0;
    for (int bytes : {1, 8, 64, 512, 4096, 32768, 262144, 1048576}) {
        std::vector<unsigned char> buffer(bytes, 42);
        const int iterations = std::min(1000, std::max(50, 8000000 / bytes));
        auto round_trip = [&]() {
            if (rank == 0) {
                MPI_Send(buffer.data(), bytes, MPI_BYTE, 1, 0, MPI_COMM_WORLD);
                MPI_Recv(buffer.data(), bytes, MPI_BYTE, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else {
                MPI_Recv(buffer.data(), bytes, MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(buffer.data(), bytes, MPI_BYTE, 0, 1, MPI_COMM_WORLD);
            }
        };
        for (int warmup = 0; warmup < 10; ++warmup) round_trip();
        MPI_Barrier(MPI_COMM_WORLD);
        const double start = MPI_Wtime();
        for (int i = 0; i < iterations; ++i) round_trip();
        const double seconds = (MPI_Wtime() - start) / (2.0 * iterations);
        failed |= !std::all_of(buffer.begin(), buffer.end(), [](auto c) { return c == 42; });
        if (rank == 0) {
            std::cout << bytes << ',' << iterations << ',' << seconds * 1e6 << ','
                      << bytes / seconds / 1e6 << '\n';
        }
    }
    int any_failed;
    MPI_Allreduce(&failed, &any_failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Finalize();
    return any_failed ? 1 : 0;
}
