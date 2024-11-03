#ifndef RENDERER_H
#define RENDERER_H

#include "PPMImage.cuh"
#include "Ray.cuh"
#include "Viewport.cuh"
#include "Sphere.cuh"
#include "Material.cuh"

using namespace geometry;
using namespace graphics;

class Renderer
{
public:
    // ----------------------------------------------------------------
    // --- Public methods
    __host__ __device__ Renderer(const Camera &camera, bool usingCUDA = false) // TODO Implement CUDA support (initializations)
    {
        m_usingCUDA = usingCUDA;
        m_imageWidth = 512; // 512 - 1280 - 1920
        m_image = PPMImage(m_imageWidth);
        m_viewport = Viewport(m_image, camera);
        m_world = HittableList();
        m_samplesPerPixel = 500; // 10 - 500
        m_pixelSamplesScale = 1.0f / m_samplesPerPixel;
        m_maxNumberOfBounces = 50;
    }

    __host__ void generateSampleSquareDistribution()
    {
        size_t distributionSize = m_image.height() * m_image.width() * m_samplesPerPixel * 2;
        m_sampleSquareDistributionSize = distributionSize;
        m_sampleSquareDistribution = new float[distributionSize];
        for (size_t i = 0; i < distributionSize; i++)
        {
            m_sampleSquareDistribution[i] = randomFloat();
        }
    }

    __host__ __device__ void clear()
    {
        // TODO Implement as destructor alternative with CUDA support
        return;
    }

    __host__ __device__ void addToWorld(const Hittable &object)
    {
        Hittable *objectPtr = object.clone();
        m_world.add(objectPtr);
    }

    __host__ void renderCPU()
    {
        for (size_t j = 0; j < m_image.height(); j++)
        {
            for (size_t i = 0; i < m_image.width(); i++)
            {
                LinearCongruentialGenerator lcg = LinearCongruentialGenerator(j * m_image.width() + i); // TODO To be calculated in each thread
                Color pixelColor(0.0f, 0.0f, 0.0f);
                for (size_t sample = 0; sample < m_samplesPerPixel; sample++)
                {
                    Ray ray = getRay(i, j, sample, lcg);
                    pixelColor += rayColor(ray, m_maxNumberOfBounces, lcg);
                }
                m_image.setPixel(j, i, graphics::RGBPixel(m_pixelSamplesScale * pixelColor));
            }
        }
    }

    __host__ void saveRenderedImage() const
    {
        std::clog << "Saving image to " << M_DEFAULT_IMAGE_PATH << '\n';
        m_image.save(M_DEFAULT_IMAGE_PATH);
    }

    // ----------------------------------------------------------------
    // --- Public attributes

    // ----------------------------------------------------------------
    // --- Public class constants

private:
    // ----------------------------------------------------------------
    // --- Private methods
    __host__ __device__ Color rayColor(const Ray &ray, uint depth, LinearCongruentialGenerator &lgc) const
    {
        if (depth <= 0)
        {
            return COLOR_BLACK();
        }

        HitRecord record;
        if (m_world.hit(ray, Interval(0.001f, INFINITY_VALUE), record))
        {
            Ray scattered;
            Color attenuation;
            if (record.material->scatter(ray, record, attenuation, scattered, lgc))
            {
                return attenuation * rayColor(scattered, depth - 1, lgc);
            }
            return COLOR_BLACK();
        }
        Vec3 unitDirection = unitVector(ray.direction());
        // A blue to white gradient is generated using linear blending
        float a = 0.5f * (unitDirection.y() + 1.0f);
        // blendedValue = (1 - a) * startValue + a * endValue
        return (1.0f - a) * COLOR_WHITE() + a * COLOR_LIGHT_BLUE();
    }

    __host__ __device__ Vec3 sampleSquare(size_t i, size_t j, size_t sample) const
    {
        size_t index = ((j * m_image.width() + i) * m_samplesPerPixel + sample) * 2;
        return Vec3(m_sampleSquareDistribution[index] - 0.5f, m_sampleSquareDistribution[index + 1] - 0.5f, 0.0f);
    }

    __host__ __device__ Point defocusDiskSample(LinearCongruentialGenerator &lcg) const
    {
        Vec3 p = randomInUnitDisk(lcg);
        return m_viewport.camera.center() + (p[0] * m_viewport.camera.defocusDiskU) + (p[1] * m_viewport.camera.defocusDiskV);
    }

    __host__ __device__ Ray getRay(size_t i, size_t j, size_t sample, LinearCongruentialGenerator &lcg) const
    {
        Vec3 offset = sampleSquare(i, j, sample);
        auto pixelSample =
            m_viewport.pixel00Location + ((i + offset.x()) * m_viewport.pixelDeltaU) + ((j + offset.y()) * m_viewport.pixelDeltaV);
        auto rayOrigin = (m_viewport.camera.defocusAngle <= 0) ? m_viewport.camera.center() : defocusDiskSample(lcg);
        auto rayDirection = pixelSample - rayOrigin;
        return Ray(rayOrigin, rayDirection);
    }

    // TODO implement time tracking on both CPU and GPU

    // __???__ void renderGPU()
    // {
    //     // TODO Implement CUDA support
    //     return;
    // }

    // ----------------------------------------------------------------
    // --- Private attributes
    size_t m_imageWidth;
    PPMImage m_image;
    Viewport m_viewport;
    HittableList m_world;
    bool m_usingCUDA;
    uint m_samplesPerPixel;
    float m_pixelSamplesScale;
    size_t m_sampleSquareDistributionSize;
    float *m_sampleSquareDistribution;
    uint m_maxNumberOfBounces;

    // ----------------------------------------------------------------
    // --- Private class constants
    static const std::string M_DEFAULT_IMAGE_PATH;
    static constexpr float M_ENERGY_REFLECTION = 0.5f;
};

#endif // RENDERER_H