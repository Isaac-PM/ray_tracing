#ifndef MATERIAL_H
#define MATERIAL_H

#include "RGBPixel.cuh"
#include "Hittable.cuh"

namespace graphics
{
    class Material
    {
    public:
        // ----------------------------------------------------------------
        // --- Public methods
        __host__ __device__ Material(const Color &albedo) : albedo(albedo) {}

        virtual ~Material() = default;

        __host__ __device__ virtual bool scatter(
            const geometry::Ray &rayIn,
            const HitRecord &record,
            Color &attenuation,
            geometry::Ray &scattered,
            LinearCongruentialGenerator &lcg) const
        {
            return false;
        }

        // ----------------------------------------------------------------
        // --- Public attributes
        Color albedo; // Radiation percentage that is reflected by a surface / "fractional reflection".

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

    class Lambertian : public Material
    {
    public:
        // ----------------------------------------------------------------
        // --- Public methods
        Lambertian(const Color &albedo) : Material(albedo) {}

        __host__ __device__ bool scatter(
            const geometry::Ray &rayIn,
            const HitRecord &record,
            Color &attenuation,
            geometry::Ray &scattered,
            LinearCongruentialGenerator &lcg) const override
        {
            /*
            Lambertian reflectance can scatter and attenuate light according
            to the reflectance (R), depending on certain probability (1 - R).
            */
            auto scatterDirection = record.normal + geometry::randomUnitVector(lcg);
            if (scatterDirection.nearZero())
            {
                // Catch degenerate scatter direction (almost opposite to the normal).
                scatterDirection = record.normal;
            }
            scattered = geometry::Ray(record.point, scatterDirection);
            attenuation = albedo;
            return true;
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

        // ----------------------------------------------------------------
        // --- Private class constants
    };

    class Metal : public Material
    {
    public:
        // ----------------------------------------------------------------
        // --- Public methods
        Metal(const Color &albedo, float fuzz)
            : Material(albedo),
              fuzz(fuzz < 1 ? fuzz : 1) {}

        __host__ __device__ bool scatter(
            const geometry::Ray &rayIn,
            const HitRecord &record,
            Color &attenuation,
            geometry::Ray &scattered,
            LinearCongruentialGenerator &lcg) const override
        {
            geometry::Vec3 reflected = reflect(rayIn.direction(), record.normal);
            reflected = geometry::unitVector(reflected + (fuzz * geometry::randomUnitVector(lcg)));
            scattered = geometry::Ray(record.point, reflected);
            attenuation = albedo;
            return (geometry::dot(scattered.direction(), record.normal) > 0);
        }

        // ----------------------------------------------------------------
        // --- Public attributes
        float fuzz;

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

    class Dielectric : public Material
    {
        
    };

} // namespace graphics

#endif // MATERIAL_H