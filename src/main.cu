#include <iostream>
#include <string>
#include <cstdlib>
#include "Renderer.cuh"
#include "Timer.cuh"

void printUsage(const std::string &programName)
{
    std::cout << "Usage: " << programName << " <benchmark_quality>\n";
    std::cout << "benchmark_quality:\n";
    std::cout << "  1 -> LOW\n";
    std::cout << "  2 -> MEDIUM\n";
    std::cout << "  3 -> HIGH\n";
}

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        printUsage(argv[0]);
        return 1;
    }

    int quality = std::atoi(argv[1]);
    if (quality < 1 || quality > 3)
    {
        printUsage(argv[0]);
        return 1;
    }

    Renderer *renderer = Renderer::loadBenchmark(static_cast<BenchmarkQuality>(quality));
    Timer renderCPU;
    renderCPU.resume();
    renderer->renderCPU();
    renderCPU.pause();
    renderCPU.print("Render CPU", TimeUnit::SECONDS);
    renderer->saveRenderedImage();

    // TODO: Free memory
    return EXIT_SUCCESS;

    return 0;
}


/*
#include <cuda_runtime.h>
#include "Renderer.cuh"
#include "Vec3.cuh"
#include "Ray.cuh"
#include "Hittable.cuh"

// Kernel function to perform ray tracing for each pixel
__global__ void rayTraceKernel(Vec3 *outputImage, int width, int height, Camera camera, Hittable **sceneObjects, int numObjects)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int pixelIndex = y * width + x;
    Ray ray = camera.getRay(x, y, width, height);

    Vec3 color = traceRay(ray, sceneObjects, numObjects);
    outputImage[pixelIndex] = color;
}

// Function to trace a ray and compute the color
__device__ Vec3 traceRay(const Ray &ray, Hittable **sceneObjects, int numObjects)
{
    // Perform intersection tests and shading calculations
    // This is a simplified example, actual implementation will vary
    Vec3 color(0, 0, 0);
    for (int i = 0; i < numObjects; ++i)
    {
        HitRecord record;
        if (sceneObjects[i]->hit(ray, 0.001f, FLT_MAX, record))
        {
            color = record.color;
            break;
        }
    }
    return color;
}

// Host function to set up and launch the kernel
void renderSceneCUDA(Vec3 *outputImage, int width, int height, Camera camera, Hittable **sceneObjects, int numObjects)
{
    Vec3 *d_outputImage;
    Hittable **d_sceneObjects;

    // Allocate memory on the GPU
    cudaMalloc(&d_outputImage, width * height * sizeof(Vec3));
    cudaMalloc(&d_sceneObjects, numObjects * sizeof(Hittable*));
    cudaMemcpy(d_sceneObjects, sceneObjects, numObjects * sizeof(Hittable*), cudaMemcpyHostToDevice);

    // Configure grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    // Launch the kernel
    rayTraceKernel<<<gridDim, blockDim>>>(d_outputImage, width, height, camera, d_sceneObjects, numObjects);

    // Copy the results back to the CPU
    cudaMemcpy(outputImage, d_outputImage, width * height * sizeof(Vec3), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_outputImage);
    cudaFree(d_sceneObjects);
}


*/