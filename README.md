# ray_tracing

> My implementation of "Ray Tracing in One Weekend" using CUDA.

![](expected_output.png)

## Overview

[_Ray Tracing in One Weekend_](https://raytracing.github.io/books/RayTracingInOneWeekend.html) is a book by Peter Shirley, Trevor David Black, and Steve Hollasch that introduces the fundamentals of ray tracing. This project is my CUDA-based implementation of the concepts from the book.

The purpose of this project is to learn the basics of ray tracing and to apply fundamental CUDA principles to improve the performance of the ray tracing algorithm.

## Running

To run the project, the NVIDIA CUDA Toolkit must be installed. The compilation process is handled with CMake as follows:

```bash
# Clone the repository
# Navigate to the project directory
mkdir build
cd build
cmake ..
make
./ray_tracing <1, 2, or 3> <gpu or cpu>
```

The first argument specifies the quality level of the output image, where 1 is the lowest quality and 3 is the highest. The second argument specifies the device (GPU or CPU) to be used for running the ray tracing algorithm.

## Performance Analysis