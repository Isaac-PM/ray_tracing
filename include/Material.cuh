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

        __host__ virtual std::string type() const
        {
            return "Material";
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
        __host__ __device__ Lambertian(const Color &albedo) : Material(albedo) {}

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

        __host__ std::string type() const override
        {
            return "Lambertian";
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
        __host__ __device__ Metal(const Color &albedo, float fuzz)
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

        __host__ std::string type() const override
        {
            return "Metal";
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
    public:
        // ----------------------------------------------------------------
        // --- Public methods
        __host__ __device__ Dielectric(float refractionIndex)
            : Material(COLOR_WHITE()),
              refractionIndex(refractionIndex) {}

        __host__ __device__ bool scatter(
            const geometry::Ray &rayIn,
            const HitRecord &record,
            Color &attenuation,
            geometry::Ray &scattered,
            LinearCongruentialGenerator &lcg) const override
        {
            attenuation = albedo;
            float ri = record.frontFace ? (1.0f / refractionIndex) : refractionIndex;
            geometry::Vec3 unitDirection = geometry::unitVector(rayIn.direction());

            float cosTheta = fminf(geometry::dot(-unitDirection, record.normal), 1.0f);
            float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);

            /*
            There are ray angles for which no solution is possible using
            Snell's law, so if the refraction index times the sine of theta
            is greater than 1, then the ray cannot be refracted, so it must
            be reflected.
            */

            bool cannotRefract = ri * sinTheta > 1.0f;
            geometry::Vec3 direction;

            if (cannotRefract || reflectance(cosTheta, ri) > lcg.nextFloat())
            {
                direction = geometry::reflect(unitDirection, record.normal);
            }
            else
            {
                direction = geometry::refract(unitDirection, record.normal, ri, cosTheta);
            }

            scattered = geometry::Ray(record.point, direction);
            return true;
        }

        __host__ std::string type() const override
        {
            return "Dielectric";
        }

        // ----------------------------------------------------------------
        // --- Public attributes
        float refractionIndex;

        // ----------------------------------------------------------------
        // --- Public class constants

    private:
        // ----------------------------------------------------------------
        // --- Private methods
        __host__ __device__ static float reflectance(float cosine, float refractionIndex)
        {
            /*
            Use Schlick's approximation for reflectance, due that glass
            has reflectivity that varies with angle.
            */
            auto r0 = (1 - refractionIndex) / (1 + refractionIndex);
            r0 = r0 * r0;
            return r0 + (1 - r0) * pow((1 - cosine), 5);
        }

        // ----------------------------------------------------------------
        // --- Private attributes

        // ----------------------------------------------------------------
        // --- Private class constants
    };

} // namespace graphics

#endif // MATERIAL_H