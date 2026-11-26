#include <mpi.h>

#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int minimum_hosts = 1;
    try {
        if (argc == 3 && std::string(argv[1]) == "--min-hosts") {
            minimum_hosts = std::stoi(argv[2]);
            if (minimum_hosts < 1) throw std::invalid_argument("positive host count required");
        } else if (argc != 1) {
            throw std::invalid_argument("usage: mpi_hello [--min-hosts N]");
        }
    } catch (const std::exception& e) {
        if (rank == 0) std::cerr << e.what() << '\n';
        MPI_Finalize();
        return 1;
    }
    char name[MPI_MAX_PROCESSOR_NAME] = {};
    int length = 0;
    MPI_Get_processor_name(name, &length);
    std::vector<char> names(static_cast<std::size_t>(ranks) * MPI_MAX_PROCESSOR_NAME);
    MPI_Gather(name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, names.data(),
               MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 0, MPI_COMM_WORLD);
    int ok = 1;
    if (rank == 0) {
        std::set<std::string> hosts;
        for (int r = 0; r < ranks; ++r) {
            const std::string host(names.data() + r * MPI_MAX_PROCESSOR_NAME);
            hosts.insert(host);
            std::cout << "rank=" << r << '/' << ranks << " host=" << host << '\n';
        }
        ok = static_cast<int>(hosts.size()) >= minimum_hosts;
        std::cout << "unique_hosts=" << hosts.size() << " required=" << minimum_hosts
                  << " result=" << (ok ? "PASS" : "FAIL") << '\n';
    }
    MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return ok ? 0 : 1;
}
