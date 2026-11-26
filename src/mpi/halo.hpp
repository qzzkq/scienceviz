#pragma once

#include "core/grid.hpp"

#include <mpi.h>
#include <vector>

namespace svmpi {

void check(int result, const char* operation);

// Коллективное создание/уничтожение. Сетка живёт дольше обменника и не меняет
// размеры. Все процессы вызывают exchange в одинаковом порядке. Собственный
// коммуникатор изолирует теги 0/1 от других частей приложения.
class HaloExchange {
public:
    HaloExchange(svcore::Grid3D& grid, std::vector<svcore::Field> fields,
                 MPI_Comm comm = MPI_COMM_WORLD);
    ~HaloExchange();
    HaloExchange(const HaloExchange&) = delete;
    HaloExchange& operator=(const HaloExchange&) = delete;

    void exchange();

private:
    svcore::Grid3D& grid_;
    std::vector<svcore::Field> fields_;
    MPI_Comm comm_ = MPI_COMM_NULL;
    int count_ = 0;
    int lower_ = 0;
    int upper_ = 0;
};

// Коллективный минимум положительных конечных локальных ограничений dt.
// При недопустимом значении хотя бы на одном процессе ошибку получают все.
double global_time_step(double local_dt, MPI_Comm comm = MPI_COMM_WORLD);

}  // namespace svmpi
