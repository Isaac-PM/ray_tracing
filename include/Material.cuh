#ifndef MATERIAL_H
#define MATERIAL_H

#include "HitRecord.cuh"
#include "LinearCongruentialGenerator.cuh"
#include "Ray.cuh"
#include "RGBPixel.cuh"
#include "Vec3.cuh"

namespace graphics
{
    enum class MaterialType
    {
        LAMBERTIAN,
        METAL,
        DIELECTRIC
    };

    struct Lambertian
    {
        // ----------------------------------------------------------------
        // --- Members

        // ----------------------------------------------------------------
        // --- Functions
        __host__ __device__ Lambertian() {}

        __host__ __device__ bool scatter(
            const Color &albedo,
            const geometry::Ray &rayIn,
            const HitRecord &record,
            Color &attenuation,
            geometry::Ray &scattered,
            LinearCongruentialGenerator &lcg) const
        {
            /*
            Lambertian reflectance can scatter and attenuate light according
            to the reflectance (R), depending on certain probability (1 - R).
           */
            geometry::Vec3 scatterDirection = record.normal + geometry::randomUnitVector(lcg);
            if (scatterDirection.nearZero())
            {
                // Catch degenerate scatter direction (almost opposite to the normal).
                scatterDirection = record.normal;
            }
            scattered = geometry::Ray(record.point, scatterDirection);
            attenuation = albedo;
            return true;
        }
    };

    struct Metal
    {
        // ----------------------------------------------------------------
        // --- Members
        float fuzz;

        // ----------------------------------------------------------------
        // --- Functions
        __host__ __device__ Metal(float fuzz) : fuzz(fuzz < 1 ? fuzz : 1) {}

        __host__ __device__ bool scatter(
            const Color &albedo,
            const geometry::Ray &rayIn,
            const HitRecord &record,
            Color &attenuation,
            geometry::Ray &scattered,
            LinearCongruentialGenerator &lcg) const
        {
            geometry::Vec3 reflected = reflect(rayIn.direction(), record.normal);
            reflected = geometry::unitVector(reflected + (fuzz * geometry::randomUnitVector(lcg)));
            scattered = geometry::Ray(record.point, reflected);
            attenuation = albedo;
            return (geometry::dot(scattered.direction(), record.normal) > 0);
        }
    };

    struct Dielectric
    {
        // ----------------------------------------------------------------
        // --- Members
        float refractionIndex;

        // ----------------------------------------------------------------
        // --- Functions
        __host__ __device__ Dielectric(float refractionIndex) : refractionIndex(refractionIndex) {}

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

        __host__ __device__ bool scatter(
            const Color &albedo,
            const geometry::Ray &rayIn,
            const HitRecord &record,
            Color &attenuation,
            geometry::Ray &scattered,
            LinearCongruentialGenerator &lcg) const
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
    };

    struct Material
    {
        // ----------------------------------------------------------------
        // --- Members
        MaterialType type;
        Color albedo;

        union
        {
            Lambertian lambertian;
            Metal metal;
            Dielectric dielectric;
        };

        // ----------------------------------------------------------------
        // --- Functions
        __host__ __device__ Material(const Color &albedo) : type(MaterialType::LAMBERTIAN), albedo(albedo) {}

        __host__ __device__ Material(const Color &albedo, float fuzz) : type(MaterialType::METAL), albedo(albedo), metal(fuzz) {}

        __host__ __device__ Material(float refractionIndex) : type(MaterialType::DIELECTRIC), albedo(COLOR_WHITE()), dielectric(refractionIndex) {}

        ~Material()
        {
            switch (type)
            {
            case MaterialType::LAMBERTIAN:
                lambertian.~Lambertian();
                break;
            case MaterialType::METAL:
                metal.~Metal();
                break;
            case MaterialType::DIELECTRIC:
                dielectric.~Dielectric();
                break;
            }
        }

        __host__ __device__ void printType() const
        {
            switch (type)
            {
            case MaterialType::LAMBERTIAN:
                printf("Lambertian\n");
                break;
            case MaterialType::METAL:
                printf("Metal\n");
                break;
            case MaterialType::DIELECTRIC:
                printf("Dielectric\n");
                break;
            }
        }

        __host__ __device__ bool scatter(
            const geometry::Ray &rayIn,
            const HitRecord &record,
            Color &attenuation,
            geometry::Ray &scattered,
            LinearCongruentialGenerator &lcg) const
        {
            switch (type)
            {
            case MaterialType::LAMBERTIAN:
                return lambertian.scatter(albedo, rayIn, record, attenuation, scattered, lcg);
            case MaterialType::METAL:
                return metal.scatter(albedo, rayIn, record, attenuation, scattered, lcg);
            case MaterialType::DIELECTRIC:
                return dielectric.scatter(albedo, rayIn, record, attenuation, scattered, lcg);
            default:
                return false;
            }
        }
    };
} // namespace graphics

#endif // MATERIAL_H