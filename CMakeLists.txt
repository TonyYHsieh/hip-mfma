cmake_minimum_required(VERSION 3.5)
project(gemm)

enable_language(CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_COMPILER "/opt/rocm/bin/hipcc")
set(CMAKE_CXX_FLAGS=" -O3 --amdgpu-target=gfx908 -gline-tables-only -save-temps=$PWD")

add_executable(gemm gemm.cpp kernel.cpp)
