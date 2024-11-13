#ifndef SPHERE_H
#define SPHERE_H

#include "Vec3.cuh"
#include "Ray.cuh"
#include "HitRecord.cuh"
#include "Interval.cuh"
#include "Material.cuh"

namespace graphics
{
    class Sphere
    {
    public:
        // ----------------------------------------------------------------
        // --- Public methods
        __host__ __device__ Sphere(
            const geometry::Vec3 &center = geometry::Vec3(0.0f, 0.0f, 0.0f),
            float radius = 0.0f,
            Material *material = nullptr)
            : center(center),
              radius(radius),
              material(material)
        {
            // TODO: Initialize the material pointer.
        }

        __host__ __device__ Sphere(const Sphere &other)
            : center(other.center),
              radius(other.radius),
              material(other.material ? new Material(*other.material) : nullptr)
        {
        }

        __host__ __device__ Sphere &operator=(const Sphere &other)
        {
            if (this != &other)
            {
                delete material;
                center = other.center;
                radius = other.radius;
                material = other.material ? new Material(*other.material) : nullptr;
            }
            return *this;
        }

        __host__ ~Sphere()
        {
            delete material;
        }

        __host__ __device__ bool hit(
            const geometry::Ray &ray,
            const geometry::Interval &interval,
            HitRecord &record) const
        {
            /*
            Any point P on the sphere satisfies the following equation: P(t) = Q + t * d, where:
            - Q is the ray's origin
            */

            const geometry::Vec3 diffCQ = center - ray.origin();
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
            geometry::Vec3 outwardNormal = (record.point - center) / radius;
            record.setFaceNormal(ray, outwardNormal);
            record.material = material;
            return true;
        }

        // ----------------------------------------------------------------
        // --- Public attributes
        geometry::Vec3 center;
        float radius;
        Material *material;

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
} // namespace graphics

namespace geometry
{
    using Sphere = graphics::Sphere;
} // namespace geometry

#endif // SPHERE_H