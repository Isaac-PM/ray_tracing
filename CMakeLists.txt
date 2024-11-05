cmake_minimum_required(VERSION 3.10)
project(RayTracing)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set the CUDA standard
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

# Set CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 50 60 70 75)

# Include directories
include_directories(include)

# Source files
file(GLOB_RECURSE SOURCES "src/*.cpp" "src/*.cu")

# Executable
add_executable(ray_tracing ${SOURCES})

# CUDA flags
set_target_properties(ray_tracing PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
