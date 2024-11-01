#ifndef RAY_H
#define RAY_H

#include "Vec3.cuh"

namespace geometry
{
    class Ray
    {
        /*
        A ray is a function with the form P(t) = A + tb, where:
        - P is a 3D position along a line in 3D;
        - A is the ray origin;
        - b is the ray direction;
        - t moves the point P along the ray:
            - For negative t values, parts behind A are obtained;
            - For positive t values, parts in front of A are obtained.
        *
        *   <\
        *       \*  <-- t = -1
        *          \
        *             \
        *                \
        *                   \*  <-- t = 0, A
        *                      \
        *                         \
        *                            \
        *                               \*  <-- t = 1, b
        *                                  \
        *                                     \
        *                                        \
        *                                           \*  <-- t = 2
        *                                              \>
        *
        */
    public:
        // ----------------------------------------------------------------
        // --- Public methods
        __host__ __device__ Ray(
            const Point &origin = Point(),
            const Vec3 &direction = Vec3())
            : m_origin(origin),
              m_direction(direction) {}

        __host__ __device__ const Vec3 &origin() const { return m_origin; }

        __host__ __device__ const Vec3 &direction() const { return m_direction; }

        __host__ __device__ Point at(float t) const
        {
            return m_origin + t * m_direction;
        }

        // ----------------------------------------------------------------
        // --- Public attributes

        // ----------------------------------------------------------------
        // --- Public class constants

    private:
        // ----------------------------------------------------------------
        // --- Private methods

        // ----------------------------------------------------------------
        // --- Private attributes
        Point m_origin;
        Vec3 m_direction;

        // ----------------------------------------------------------------
        // --- Private class constants
    };
} // namespace geometry

#endif // RAY_H