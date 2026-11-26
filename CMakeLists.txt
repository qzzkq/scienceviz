cmake_minimum_required(VERSION 3.16)
project(scienceviz LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

add_library(scienceviz_lib
    src/volume.cpp
    src/synthetic.cpp
    src/npy.cpp
    src/render.cpp
)
target_include_directories(scienceviz_lib PUBLIC src)
target_include_directories(scienceviz_lib SYSTEM PUBLIC third_party)
find_package(OpenMP QUIET COMPONENTS CXX)
if(OpenMP_CXX_FOUND)
    target_link_libraries(scienceviz_lib PRIVATE OpenMP::OpenMP_CXX)
endif()

# Ядро симуляции: сетка, газодинамика, конфиг, снимки. От рендера не зависит.
add_library(svcore
    src/core/grid.cpp
)
target_include_directories(svcore PUBLIC src)

add_library(grid_volume src/grid_volume.cpp)
target_link_libraries(grid_volume PUBLIC scienceviz_lib svcore)

add_executable(scienceviz src/main.cpp)
target_link_libraries(scienceviz PRIVATE grid_volume)

add_executable(scienceviz_tests tests/test_all.cpp)
target_link_libraries(scienceviz_tests PRIVATE scienceviz_lib)

add_executable(svcore_tests tests/test_core.cpp)
target_link_libraries(svcore_tests PRIVATE svcore)

add_executable(grid_volume_tests tests/test_grid_volume.cpp)
target_link_libraries(grid_volume_tests PRIVATE grid_volume)

enable_testing()
add_test(NAME scienceviz_tests COMMAND scienceviz_tests)
add_test(NAME svcore_tests COMMAND svcore_tests)
add_test(NAME grid_volume_tests COMMAND grid_volume_tests)

option(SV_ENABLE_MPI "Build distributed grid tools and MPI tests" OFF)
if(SV_ENABLE_MPI)
    find_package(MPI REQUIRED COMPONENTS CXX)
    add_library(svmpi src/mpi/halo.cpp)
    target_include_directories(svmpi PUBLIC src)
    target_link_libraries(svmpi PUBLIC svcore MPI::MPI_CXX)

    add_executable(svmpi_tests tests/test_mpi.cpp)
    target_link_libraries(svmpi_tests PRIVATE svmpi)
    add_executable(mpi_hello examples/mpi_hello.cpp)
    target_link_libraries(mpi_hello PRIVATE MPI::MPI_CXX)
    add_executable(mpi_pingpong examples/mpi_pingpong.cpp)
    target_link_libraries(mpi_pingpong PRIVATE MPI::MPI_CXX)
    add_executable(mpi_diffusion examples/mpi_diffusion.cpp)
    target_link_libraries(mpi_diffusion PRIVATE svmpi)

    foreach(ranks 1 2 3)
        add_test(NAME svmpi_${ranks}
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:svmpi_tests> ${MPIEXEC_POSTFLAGS})
        set_tests_properties(svmpi_${ranks} PROPERTIES TIMEOUT 45 PROCESSORS ${ranks})
        foreach(halo 1 2)
            add_test(NAME diffusion_${ranks}_halo${halo}
                COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
                    ${MPIEXEC_PREFLAGS} $<TARGET_FILE:mpi_diffusion> ${MPIEXEC_POSTFLAGS}
                    --nx 17 --ny 13 --nz 19 --halo ${halo} --steps 20)
            set_tests_properties(diffusion_${ranks}_halo${halo}
                PROPERTIES TIMEOUT 60 PROCESSORS ${ranks})
        endforeach()
    endforeach()
endif()

find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
    execute_process(COMMAND ${Python3_EXECUTABLE} -c "import numpy"
        RESULT_VARIABLE NUMPY_RESULT OUTPUT_QUIET ERROR_QUIET)
    if(NUMPY_RESULT EQUAL 0)
        add_test(NAME snapshot_pipeline
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/test_pipeline.py $<TARGET_FILE:scienceviz>)
        add_test(NAME video_interpolation
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/test_video_interpolation.py)
    endif()
endif()
