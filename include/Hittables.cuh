#ifndef HITTABLES_H
#define HITTABLES_H

#include "HitRecord.cuh"
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

        __host__ void addSphere(const Sphere &sphere)
        {
            if (spheresCount < spheresSize)
            {
                spheres[spheresCount++] = sphere;
            }
        }

        __host__ __device__ Sphere &getSphere(size_t index) const
        {
            return spheres[index];
        }

        __host__ __device__ bool hit(
            const geometry::Ray &ray,
            const geometry::Interval &interval,
            HitRecord &record)
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

        // ----------------------------------------------------------------
        // --- Public attributes
        size_t spheresSize;
        size_t spheresCount;
        Sphere *spheres;
        // Declare other hittable objects arrays here...

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