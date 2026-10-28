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

add_executable(scienceviz src/main.cpp)
target_link_libraries(scienceviz PRIVATE scienceviz_lib)

add_executable(scienceviz_tests tests/test_all.cpp)
target_link_libraries(scienceviz_tests PRIVATE scienceviz_lib)

enable_testing()
add_test(NAME scienceviz_tests COMMAND scienceviz_tests)
