#include "mpi/halo.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace svmpi {

void check(int result, const char* operation) {
    if (result == MPI_SUCCESS) {
        return;
    }
    char message[MPI_MAX_ERROR_STRING];
    int length = 0;
    MPI_Error_string(result, message, &length);
    throw std::runtime_error(std::string(operation) + ": " + std::string(message, length));
}

HaloExchange::HaloExchange(svcore::Grid3D& grid, std::vector<svcore::Field> fields,
                           MPI_Comm comm)
    : grid_(grid) {
    int mask = 0;
    int invalid = grid.empty() || grid.z_slab_count() > static_cast<std::size_t>(
        std::numeric_limits<int>::max());
    for (auto field : fields) {
        const int index = static_cast<int>(field);
        if (index < 0 || index >= svcore::kFieldCount) {
            invalid = 1;
        } else if ((mask & (1 << index)) != 0) {
            invalid = 1;
        } else {
            mask |= 1 << index;
        }
    }
    invalid |= mask == 0;
    int any_invalid = 0;
    check(MPI_Allreduce(&invalid, &any_invalid, 1, MPI_INT, MPI_MAX, comm), "halo validation");
    // Один порядок полей на всех процессах, независимо от порядка аргументов.
    int signature[] = {grid.nx(), grid.ny(), grid.halo(), mask};
    int minimum[4], maximum[4];
    check(MPI_Allreduce(signature, minimum, 4, MPI_INT, MPI_MIN, comm), "halo layout min");
    check(MPI_Allreduce(signature, maximum, 4, MPI_INT, MPI_MAX, comm), "halo layout max");
    if (any_invalid || !std::equal(minimum, minimum + 4, maximum)) {
        throw std::invalid_argument("HaloExchange: invalid or inconsistent xy/halo/fields");
    }
    for (int i = 0; i < svcore::kFieldCount; ++i) {
        if ((mask & (1 << i)) != 0) {
            fields_.push_back(static_cast<svcore::Field>(i));
        }
    }
    int rank, ranks;
    check(MPI_Comm_rank(comm, &rank), "rank");
    check(MPI_Comm_size(comm, &ranks), "size");
    count_ = static_cast<int>(grid.z_slab_count());
    lower_ = (rank + ranks - 1) % ranks;
    upper_ = (rank + 1) % ranks;
    check(MPI_Comm_dup(comm, &comm_), "halo communicator");
    MPI_Comm_set_errhandler(comm_, MPI_ERRORS_RETURN);
}

HaloExchange::~HaloExchange() {
    if (comm_ != MPI_COMM_NULL) {
        MPI_Comm_free(&comm_);
    }
}

void HaloExchange::exchange() {
    for (auto field : fields_) {
        grid_.fill_halo_periodic_xy(field);
        check(MPI_Sendrecv(grid_.z_send_hi(field), count_, MPI_FLOAT, upper_, 0,
                           grid_.z_recv_lo(field), count_, MPI_FLOAT, lower_, 0,
                           comm_, MPI_STATUS_IGNORE), "halo up");
        check(MPI_Sendrecv(grid_.z_send_lo(field), count_, MPI_FLOAT, lower_, 1,
                           grid_.z_recv_hi(field), count_, MPI_FLOAT, upper_, 1,
                           comm_, MPI_STATUS_IGNORE), "halo down");
    }
}

double global_time_step(double local_dt, MPI_Comm comm) {
    int invalid = !std::isfinite(local_dt) || local_dt <= 0.0;
    int any_invalid = 0;
    check(MPI_Allreduce(&invalid, &any_invalid, 1, MPI_INT, MPI_MAX, comm), "dt validation");
    if (any_invalid) {
        throw std::invalid_argument("global_time_step: dt must be finite and positive on every rank");
    }
    double result;
    check(MPI_Allreduce(&local_dt, &result, 1, MPI_DOUBLE, MPI_MIN, comm), "dt minimum");
    return result;
}

}  // namespace svmpi
