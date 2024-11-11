#ifndef RENDERER_H
#define RENDERER_H

#include "Hittables.cuh"
#include "Material.cuh"
#include "PPMImage.cuh"
#include "Ray.cuh"
#include "Sphere.cuh"
#include "Viewport.cuh"
#include <filesystem>

#define cudaCheckError()                                                                 \
    {                                                                                    \
        cudaError_t e = cudaGetLastError();                                              \
        if (e != cudaSuccess)                                                            \
        {                                                                                \
            printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
            exit(EXIT_FAILURE);                                                          \
        }                                                                                \
    }

#define THREAD_X_COUNT 32
#define THREAD_Y_COUNT 32

using namespace geometry;
using namespace graphics;

enum BenchmarkQuality
{
    LOW = 1,
    MEDIUM = 2,
    HIGH = 3
};

__host__ __device__ Vec3 sampleSquare(
    size_t i,
    size_t j,
    size_t sample,
    LinearCongruentialGenerator *lcg,
    const PPMImage *image,
    uint samplesPerPixel);

__host__ __device__ Point defocusDiskSample(
    LinearCongruentialGenerator &lcg,
    const Viewport *viewport);

__host__ __device__ Ray getRay(
    size_t i,
    size_t j,
    size_t sample,
    LinearCongruentialGenerator *lcg,
    const PPMImage *image,
    uint samplesPerPixel,
    const Viewport *viewport);

__host__ Color rayColorCPU(
    const Ray *ray,
    uint depth,
    LinearCongruentialGenerator *lgc,
    Hittables *world);

__device__ Color rayColorGPU(
    const Ray *ray,
    uint depth,
    LinearCongruentialGenerator *lgc,
    Hittables *world);

__global__ void launchGPU(
    size_t imageWidth,
    PPMImage *image,
    Viewport *viewport,
    Hittables *world,
    uint samplesPerPixel,
    float pixelSamplesScale,
    uint maxNumberOfBounces);

class Renderer
{
public:
    // ----------------------------------------------------------------
    // --- Public methods
    __host__ __device__ Renderer(
        size_t imageWidth,
        uint samplesPerPixel,
        uint maxNumberOfBounces,
        size_t hittableSpheresSize,
        const Camera &camera,
        bool usingCUDA = false)
        : m_imageWidth(imageWidth),
          m_samplesPerPixel(samplesPerPixel),
          m_maxNumberOfBounces(maxNumberOfBounces),
          m_usingCUDA(usingCUDA),
          m_image(m_imageWidth),
          m_viewport(m_image, camera),
          m_world(Hittables(hittableSpheresSize)),
          m_pixelSamplesScale(1.0f / m_samplesPerPixel)
    {
    }

    __host__ void clear()
    {
        // TODO: Implement as destructor alternative with CUDA support
    }

    __host__ void addSphereToWorld(const Sphere &sphere)
    {
        m_world.addSphere(sphere);
    }

    __host__ void renderCPU()
    {
        for (size_t j = 0; j < m_image.height(); j++)
        {
            for (size_t i = 0; i < m_image.width(); i++)
            {
                LinearCongruentialGenerator lcg = LinearCongruentialGenerator(j * m_image.width() + i);
                Color pixelColor(0.0f, 0.0f, 0.0f);
                for (size_t sample = 0; sample < m_samplesPerPixel; sample++)
                {
                    Ray ray = getRay(i, j, sample, &lcg, &m_image, m_samplesPerPixel, &m_viewport);
                    pixelColor += rayColorCPU(&ray, m_maxNumberOfBounces, &lcg, &m_world);
                }
                m_image.setPixel(j, i, graphics::RGBPixel(m_pixelSamplesScale * pixelColor));
            }
        }
        saveRenderedImage();
    }

    __host__ void renderGPU()
    {
        PPMImage *d_image;
        cudaMallocManaged(&d_image, sizeof(m_image));
        cudaCheckError();
        cudaMemcpy(d_image, &m_image, sizeof(m_image), cudaMemcpyHostToDevice);
        cudaCheckError();

        Viewport *d_viewport;
        cudaMallocManaged(&d_viewport, sizeof(m_viewport));
        cudaCheckError();
        cudaMemcpy(d_viewport, &m_viewport, sizeof(m_viewport), cudaMemcpyHostToDevice);
        cudaCheckError();

        Hittables *d_world;
        cudaMallocManaged(&d_world, sizeof(m_world));
        cudaCheckError();
        cudaMemcpy(d_world, &m_world, sizeof(m_world), cudaMemcpyHostToDevice);
        cudaCheckError();

        dim3 dimGrid(ceil(m_image.width() / (float)THREAD_X_COUNT), ceil(m_image.height() / (float)THREAD_Y_COUNT), 1);
        dim3 dimBlock(THREAD_X_COUNT, THREAD_Y_COUNT, 1);

        launchGPU<<<dimGrid, dimBlock>>>(m_image.width(), d_image, d_viewport, d_world, m_samplesPerPixel, m_pixelSamplesScale, m_maxNumberOfBounces);
        cudaDeviceSynchronize();

        saveRenderedImage(d_image);

        // TODO Free memory
    }

    __host__ void saveRenderedImage(PPMImage *image = nullptr) const
    {
        std::cout << "Saving image to " << M_DEFAULT_IMAGE_PATH << '\n';
        image != nullptr ? image->save(M_DEFAULT_IMAGE_PATH) : m_image.save(M_DEFAULT_IMAGE_PATH);
    }

    __host__ __device__ const Hittables &world() const
    {
        return m_world;
    }

    __host__ static void generateBenchmark(BenchmarkQuality quality);

    __host__ static Renderer *loadBenchmark(BenchmarkQuality quality);

    // ----------------------------------------------------------------
    // --- Public attributes

    // ----------------------------------------------------------------
    // --- Public class constants

private:
    // ----------------------------------------------------------------
    // --- Private methods

    // ----------------------------------------------------------------
    // --- Private attributes
    size_t m_imageWidth;
    PPMImage m_image;
    Viewport m_viewport;
    Hittables m_world;
    bool m_usingCUDA;
    uint m_samplesPerPixel;
    float m_pixelSamplesScale;
    uint m_maxNumberOfBounces;

    // ----------------------------------------------------------------
    // --- Private class constants
    static const std::string M_DEFAULT_IMAGE_PATH;
    static const std::string M_BENCHMARK_PATH;
    static constexpr float M_ENERGY_REFLECTION = 0.5f;
};

__host__ __device__ inline Vec3 sampleSquare(
    size_t i,
    size_t j,
    size_t sample,
    LinearCongruentialGenerator *lcg,
    const PPMImage *image,
    uint samplesPerPixel)
{
    size_t index = ((j * image->width() + i) * samplesPerPixel + sample) * 2;
    return Vec3(lcg->nextFloat() - 0.5f, lcg->nextFloat() - 0.5f, 0.0f);
}

__host__ __device__ inline Point defocusDiskSample(
    LinearCongruentialGenerator *lcg,
    const Viewport *viewport)
{
    Vec3 p = randomInUnitDisk(*lcg);
    return viewport->camera.center() + (p[0] * viewport->camera.defocusDiskU) + (p[1] * viewport->camera.defocusDiskV);
}

__host__ __device__ inline Ray getRay(
    size_t i,
    size_t j,
    size_t sample,
    LinearCongruentialGenerator *lcg,
    const PPMImage *image,
    uint samplesPerPixel,
    const Viewport *viewport)
{
    Vec3 offset = sampleSquare(i, j, sample, lcg, image, samplesPerPixel);
    auto pixelSample =
        viewport->pixel00Location + ((i + offset.x()) * viewport->pixelDeltaU) + ((j + offset.y()) * viewport->pixelDeltaV);
    auto rayOrigin = (viewport->camera.defocusAngle <= 0) ? viewport->camera.center() : defocusDiskSample(lcg, viewport);
    auto rayDirection = pixelSample - rayOrigin;
    return Ray(rayOrigin, rayDirection);
}

__host__ inline Color rayColorCPU(
    const Ray *ray,
    uint depth,
    LinearCongruentialGenerator *lgc,
    Hittables *world)
{
    if (depth <= 0)
    {
        return COLOR_BLACK();
    }

    HitRecord record;
    if (world->hit(*ray, Interval(0.001f, INFINITY_VALUE), record))
    {
        Ray scattered;
        Color attenuation;
        if (record.material->scatter(*ray, record, attenuation, scattered, *lgc))
        {
            return attenuation * rayColorCPU(&scattered, depth - 1, lgc, world);
        }
        return COLOR_BLACK();
    }
    Vec3 unitDirection = unitVector(ray->direction());
    // A blue to white gradient is generated using linear blending
    float a = 0.5f * (unitDirection.y() + 1.0f);
    // blendedValue = (1 - a) * startValue + a * endValue
    return (1.0f - a) * COLOR_WHITE() + a * COLOR_LIGHT_BLUE();
}

__device__ inline Color rayColorGPU(
    const Ray *ray,
    uint depth,
    LinearCongruentialGenerator *lgc,
    Hittables *world)
{
    Color accumulatedColor = COLOR_WHITE();
    Ray currentRay = *ray;
    uint currentDepth = depth;

    while (currentDepth > 0)
    {
        HitRecord record;
        if (world->hit(currentRay, Interval(0.001f, INFINITY_VALUE), record))
        {
            Ray scattered;
            Color attenuation;
            if (record.material->scatter(currentRay, record, attenuation, scattered, *lgc))
            {
                printf("Scattered\n");
                accumulatedColor *= attenuation; // Accumulate color with attenuation
                // currentRay = scattered;          // Move to the next scattered ray // *******
                currentDepth--; // Decrease depth for the next iteration
            }
            else
            {
                return COLOR_BLACK(); // Stop if scatter fails
            }
        }
        else
        {
            Vec3 unitDirection = unitVector(currentRay.direction());
            float a = 0.5f * (unitDirection.y() + 1.0f);
            accumulatedColor *= (1.0f - a) * COLOR_WHITE() + a * COLOR_LIGHT_BLUE();
            break;
        }
    }

    return accumulatedColor;
}

__global__ inline void launchGPU(
    size_t imageWidth,
    PPMImage *image,
    Viewport *viewport,
    Hittables *world,
    uint samplesPerPixel,
    float pixelSamplesScale,
    uint maxNumberOfBounces)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= image->width() || j >= image->height())
    {
        return;
    }
    LinearCongruentialGenerator lcg = LinearCongruentialGenerator(j * image->width() + i);
    Color pixelColor(0.0f, 0.0f, 0.0f);
    for (size_t sample = 0; sample < samplesPerPixel; sample++)
    {
        Ray ray = getRay(i, j, sample, &lcg, image, samplesPerPixel, viewport);
        pixelColor += rayColorGPU(&ray, maxNumberOfBounces, &lcg, world);
    }
    image->setPixel(j, i, graphics::RGBPixel(1.0f / samplesPerPixel * pixelColor));
}

#endif // RENDERER_H