#ifndef HITTABLES_H
#define HITTABLES_H

#include "Hittable.cuh"
#include "Interval.cuh"
#include "Ray.cuh"
#include "Sphere.cuh"

namespace graphics
{
    class Hittables
    {
    public:
        // ----------------------------------------------------------------
        // --- Public methods
        __host__ __device__ Hittables(
            size_t spheresSize = DEFAULT_SIZE,
            size_t spheresCount = 0,
            Sphere *spheresPtr = nullptr)
            : spheresSize(spheresSize),
              spheresCount(spheresCount),
              spheres(spheresPtr)
        {
            if (spheresPtr == nullptr)
            {
                spheres = new Sphere[spheresSize];
            }
        }

        __host__ void clear(bool usingCUDA = false)
        {
            // TODO: Implement as destructor alternative with CUDA support
        }

        __host__ void addSphere(const Sphere &sphere)
        {
            if (spheresCount < spheresSize)
            {
                spheres[spheresCount++] = sphere;
            }
        }

        // __host__ void setupCUDA()
        // {
        //     Sphere *d_spheres;
        //     cudaMallocManaged(&d_spheres, spheresSize * sizeof(Sphere));
        //     cudaMemcpy(d_spheres, spheres, spheresSize * sizeof(Sphere), cudaMemcpyHostToDevice);
        //     delete[] spheres;
        //     spheres = d_spheres;
        // }

        __host__ __device__ bool hit(
            const geometry::Ray &ray,
            const geometry::Interval &interval,
            HitRecord &record) const
        {
            HitRecord tempRecord;
            bool hitAnything = false;
            float closestSoFar = interval.max;

            for (size_t i = 0; i < spheresCount; i++)
            {
                if (spheres[i].hit(ray, geometry::Interval(interval.min, closestSoFar), tempRecord))
                {
                    hitAnything = true;
                    closestSoFar = tempRecord.t;
                    record = tempRecord;
                }
            }

            return hitAnything;
        }

        __host__ __device__ Sphere &getSphere(size_t index) const
        {
            return spheres[index];
        }

        // ----------------------------------------------------------------
        // --- Public attributes
        size_t spheresSize;
        size_t spheresCount;
        Sphere *spheres;
        // Declare other hittable object arrays here...

        // ----------------------------------------------------------------
        // --- Public class constants
        static const size_t DEFAULT_SIZE = 10;

    private:
        // ----------------------------------------------------------------
        // --- Private methods

        // ----------------------------------------------------------------
        // --- Private attributes

        // ----------------------------------------------------------------
        // --- Private class constants
    };
} // namespace graphics

#endif // HITTABLES_H