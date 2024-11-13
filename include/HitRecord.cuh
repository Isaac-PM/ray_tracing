#ifndef HITRECORD_H
#define HITRECORD_H

#include "Ray.cuh"
#include "Interval.cuh"

namespace graphics
{
    class Material;

    class HitRecord
    {
    public:
        // ----------------------------------------------------------------
        // --- Public methods
        __host__ __device__ HitRecord() {}

        __host__ __device__ HitRecord(
            const geometry::Point &point,
            const geometry::Vec3 &normal,
            float t)
            : point(point),
              normal(normal),
              t(t)
        {
        }

        __host__ __device__ void setFaceNormal(const geometry::Ray &ray, const geometry::Vec3 &outwardNormal)
        {
            // Sets the hit record normal vector.
            // NOTE: the parameter 'outwardNormal' is assumed to have unit length.
            frontFace = geometry::dot(ray.direction(), outwardNormal) < 0;
            normal = frontFace ? outwardNormal : -outwardNormal;
        }

        // ----------------------------------------------------------------
        // --- Public attributes
        geometry::Point point;
        geometry::Vec3 normal;
        /*
        Surface normal for shading, vector that is perpendicular
        to the surface at the point of intersection.
        *
        *             ^
        *              \ <-- Normal vector
        *               \
        *        ********\******
        *      ***        \    ***
        *     **           .p    **
        *    ** _---------------_ **
        *   ** -_               _- **
        *    **  ---------------  **
        *     **                 **
        *      ***             *** <-- A sphere
        *        ***************
        *
        */
        float t; // Distance along the ray.
        bool frontFace;
        Material *material; // Is deleted by the struck object 

        // ----------------------------------------------------------------
        // --- Public class constants

    private:
        // ----------------------------------------------------------------
        // --- Private methods

        // ----------------------------------------------------------------
        // --- Private attributes

        // ----------------------------------------------------------------
        // --- Private class constants
    };
} // namespace graphics

#endif // HITRECORD_H