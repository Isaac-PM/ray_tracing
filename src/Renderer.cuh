#ifndef RENDERER_H
#define RENDERER_H

#include "PPMImage.cuh"
#include "Ray.cuh"
#include "Viewport.cuh"
#include "Sphere.cuh"

using namespace geometry;
using namespace graphics;

class Renderer
{
public:
    // ----------------------------------------------------------------
    // --- Public methods
    __host__ __device__ Renderer(bool usingCUDA = false) // TODO Implement CUDA support (initializations)
    {
        m_usingCUDA = usingCUDA;
        m_image = PPMImage();
        m_viewport = Viewport(m_image);
        m_world = HittableList();
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
                auto pixelCenter =
                    m_viewport.pixel00Location + (i * m_viewport.pixelDeltaU) + (j * m_viewport.pixelDeltaV);

                auto rayDirection = pixelCenter - m_viewport.camera.center();

                graphics::Color pixelColor = rayColor(geometry::Ray(m_viewport.camera.center(), rayDirection));

                m_image.setPixel(j, i, graphics::RGBPixel(pixelColor));
            }
        }
    }

    __host__ void saveRenderedImage()
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
    __host__ __device__ Color rayColor(const Ray &ray) const
    {
        HitRecord record;
        if (m_world.hit(ray, Interval(0, INFINITY_VALUE), record))
        {
            return 0.5f * (record.normal + COLOR_WHITE());
        }
        Vec3 unitDirection = unitVector(ray.direction());
        // A blue to white gradient is generated using linear blending
        float a = 0.5f * (unitDirection.y() + 1.0f);
        // blendedValue = (1 - a) * startValue + a * endValue
        return (1.0f - a) * COLOR_WHITE() + a * COLOR_LIGHT_BLUE();
    }

    // TODO implement time tracking on both CPU and GPU

    // __???__ void renderGPU()
    // {
    //     // TODO Implement CUDA support
    //     return;
    // }

    // ----------------------------------------------------------------
    // --- Private attributes
    PPMImage m_image;
    Viewport m_viewport;
    HittableList m_world;
    bool m_usingCUDA;

    // ----------------------------------------------------------------
    // --- Private class constants
    static const std::string M_DEFAULT_IMAGE_PATH;
};

#endif // RENDERER_H