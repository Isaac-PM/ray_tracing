#ifndef HITTABLE_H
#define HITTABLE_H

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
        Material *material;

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

    class Hittable
    {
    public:
        // ----------------------------------------------------------------
        // --- Public methods
        __host__ __device__ virtual ~Hittable() = default;

        __host__ __device__ virtual bool hit(
            const geometry::Ray &ray,
            const geometry::Interval &interval,
            HitRecord &record) const = 0;

        __host__ __device__ virtual Hittable *clone(bool usingCUDA = false) const = 0;

        // ----------------------------------------------------------------
        // --- Public attributes

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

    class HittableList : public Hittable
    {
    public:
        // ----------------------------------------------------------------
        // --- Public methods
        __host__ __device__ HittableList(
            size_t size = DEFAULT_SIZE,
            size_t count = 0,
            Hittable **hittablesPtr = nullptr)
            : size(size),
              count(count),
              hittables(hittablesPtr)
        {
            if (hittables == nullptr)
            {
                hittables = new Hittable *[size];
                for (size_t i = 0; i < size; i++)
                {
                    hittables[i] = nullptr;
                }
            }
        }

        __host__ __device__ void clear(bool usingCUDA = false)
        {
            // TODO: Implement as destructor alternative with CUDA support
            for (size_t i = 0; i < count; i++)
            {
                delete hittables[i];
            }
            delete[] hittables;
        }

        __host__ __device__ void add(Hittable *hittable)
        {
            hittables[count++] = hittable;
        }

        __host__ __device__ bool hit(
            const geometry::Ray &ray,
            const geometry::Interval &interval,
            HitRecord &record) const override
        {
            HitRecord tempRecord;
            bool hitAnything = false;
            float closestSoFar = interval.max;

            for (size_t i = 0; i < count; i++)
            {
                Hittable *hittable = hittables[i];
                if (hittable->hit(ray, geometry::Interval(interval.min, closestSoFar), tempRecord))
                {
                    hitAnything = true;
                    closestSoFar = tempRecord.t;
                    record = tempRecord;
                }
            }
            return hitAnything;
        }

        __host__ __device__ Hittable *clone(bool usingCUDA = false) const override
        {
            // Not intended to be used.
            return nullptr;
        }

        __host__ __device__ Hittable *operator[](size_t index) const
        {
            return hittables[index];
        }

        // ----------------------------------------------------------------
        // --- Public attributes
        size_t size;
        size_t count;
        Hittable **hittables;

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

#endif // HITTABLE_H