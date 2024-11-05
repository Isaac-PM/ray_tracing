#ifndef SPHERE_H
#define SPHERE_H

#include "Vec3.cuh"
#include "Ray.cuh"
#include "Hittable.cuh"
#include "Interval.cuh"
#include <fstream>

namespace geometry
{
    class Sphere : public graphics::Hittable
    {
    public:
        // ----------------------------------------------------------------
        // --- Public methods
        __host__ __device__ Sphere(
            const Vec3 &center = Vec3(0.0f, 0.0f, 0.0f),
            float radius = 0.0f,
            graphics::Material *material = nullptr)
            : center(center),
              radius(radius),
              material(material)
        {
            // TODO: Initialize the material pointer.
        }

        __host__ __device__ bool hit(
            const geometry::Ray &ray,
            const geometry::Interval &interval,
            graphics::HitRecord &record) const
        {
            /*
            Any point P on the sphere satisfies the following equation: P(t) = Q + t * d, where:
            - Q is the ray's origin
            */

            const Vec3 diffCQ = center - ray.origin();
            auto a = ray.direction().lengthSquared;
            auto h = dot(ray.direction(), diffCQ);
            auto c = diffCQ.lengthSquared - radius * radius;

            auto discriminant = h * h - a * c;
            if (discriminant < 0)
            {
                return false;
            }

            auto sqrtDiscriminant = sqrtf(discriminant);
            auto root = (h - sqrtDiscriminant) / a;
            if (!interval.surrounds(root))
            {
                root = (h + sqrtDiscriminant) / a;
                if (!interval.surrounds(root))
                {
                    return false;
                }
            }
            record.t = root;
            record.point = ray.at(record.t);
            Vec3 outwardNormal = (record.point - center) / radius;
            record.setFaceNormal(ray, outwardNormal);
            record.material = material;
            return true;
        }

        __host__ __device__ Hittable *clone(bool usingCUDA = false) const override // TODO: Implement CUDA support
        {
            return new Sphere(center, radius, material);
        }

        // ----------------------------------------------------------------
        // --- Public attributes
        Vec3 center;
        float radius;
        graphics::Material *material;

        // ----------------------------------------------------------------
        // --- Public class constants
        static constexpr float NO_HIT = -1.0f;

    private:
        // ----------------------------------------------------------------
        // --- Private methods

        // ----------------------------------------------------------------
        // --- Private attributes

        // ----------------------------------------------------------------
        // --- Private class constants
    };
} // namespace geometry

#endif // SPHERE_H