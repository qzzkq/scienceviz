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
    src/core/sim1d.cpp
)
target_include_directories(svcore PUBLIC src)

add_library(grid_volume src/grid_volume.cpp)
target_link_libraries(grid_volume PUBLIC scienceviz_lib svcore)

add_executable(scienceviz src/main.cpp)
target_link_libraries(scienceviz PRIVATE grid_volume)

add_executable(scienceviz_sim1d src/sim1d_main.cpp)
target_link_libraries(scienceviz_sim1d PRIVATE svcore)

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

find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
    execute_process(COMMAND ${Python3_EXECUTABLE} -c "import numpy"
        RESULT_VARIABLE NUMPY_RESULT OUTPUT_QUIET ERROR_QUIET)
    if(NUMPY_RESULT EQUAL 0)
        add_test(NAME snapshot_pipeline
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/test_pipeline.py $<TARGET_FILE:scienceviz>)
        add_test(NAME video_interpolation
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/test_video_interpolation.py)
        add_test(NAME sim1d_reference
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/test_sim1d.py $<TARGET_FILE:scienceviz_sim1d>)
    endif()
endif()
