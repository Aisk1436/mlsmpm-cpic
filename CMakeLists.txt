cmake_minimum_required(VERSION 3.20)

project(MPM CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

add_subdirectory(Solver)
add_subdirectory(Demo)
