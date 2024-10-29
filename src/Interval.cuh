#ifndef INTERVAL_H
#define INTERVAL_H

#include "Vec3.cuh"

namespace geometry
{
    class Interval
    {
    public:
        // ----------------------------------------------------------------
        // --- Public methods
        __host__ __device__ Interval(
            float min = -INFINITY_VALUE,
            float max = INFINITY_VALUE)
            : min(min),
              max(max)
        {
        }

        __host__ __device__ float size() const
        {
            return max - min;
        }

        __host__ __device__ bool contains(float x) const
        {
            return x >= min && x <= max;
        }

        __host__ __device__ bool surrounds(float x) const
        {
            return x > min && x < max;
        }

        // ----------------------------------------------------------------
        // --- Public attributes
        float min;
        float max;

        // ----------------------------------------------------------------
        // --- Public class constants

        static const Interval EMPTY;
        static const Interval UNIVERSE;

    private:
        // ----------------------------------------------------------------
        // --- Private methods

        // ----------------------------------------------------------------
        // --- Private attributes

        // ----------------------------------------------------------------
        // --- Private class constants
    };
} // namespace geometry

#endif // INTERVAL_H