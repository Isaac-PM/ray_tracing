#ifndef VEC3_H
#define VEC3_H

#include <cmath>
#include <iostream>
#include <limits>
#include <cstdlib>
#include "LinearCongruentialGenerator.cuh"

namespace geometry
{
    constexpr float INFINITY_VALUE = std::numeric_limits<float>::infinity();
    constexpr float PI = 3.141592f;

    __host__ __device__ inline float degreesToRadians(float degrees)
    {
        return degrees * PI / 180.0f;
    }

    __host__ inline float randomFloat()
    {
        // Returns a random real in [0,1).
        return rand() / (RAND_MAX + 1.0);
    }

    __host__ inline float randomFloat(float min, float max)
    {
        // Returns a random real in [min,max).
        return min + (max - min) * randomFloat();
    }

    class Vec3
    {
    public:
        // ----------------------------------------------------------------
        // --- Public methods
        __host__ __device__ Vec3() : values{0.0, 0.0, 0.0} {}

        __host__ __device__ Vec3(float v0, float v1, float v2) : values{v0, v1, v2} {}

        __host__ __device__ float x() const { return values[0]; };
        __host__ __device__ float &x() { return values[0]; };

        __host__ __device__ float y() const { return values[1]; };
        __host__ __device__ float &y() { return values[1]; };

        __host__ __device__ float z() const { return values[2]; };
        __host__ __device__ float &z() { return values[2]; };

        __host__ __device__ Vec3 operator-() const { return Vec3(-x(), -y(), -z()); }

        __host__ __device__ float operator[](uint index) const { return values[index]; }

        __host__ __device__ float &operator[](uint index) { return values[index]; }

        __host__ __device__ Vec3 &operator+=(const Vec3 &v)
        {
            x() += v.x();
            y() += v.y();
            z() += v.z();
            return *this;
        }

        __host__ __device__ Vec3 &operator*=(float t)
        {
            x() *= t;
            y() *= t;
            z() *= t;
            return *this;
        }

        __host__ __device__ Vec3 &operator/=(float t)
        {
            return *this *= 1 / t;
        }

        __host__ __device__ float lengthSquared() const
        {
            return x() * x() + y() * y() + z() * z();
        }

        __host__ __device__ bool nearZero() const
        {
            // Returns true if vector is close to zero in all dimensions.
            auto s = 1e-8;
            return (fabsf(x()) < s && fabsf(y()) < s && fabsf(z()) < s);
        }

        __host__ __device__ float length() const
        {
            return sqrtf(lengthSquared());
        }

        // __host__ __device__ static Vec3 random(LinearCongruentialGenerator &lcg)
        // {
        //     float min = 0.0f;
        //     float max = 1.0f;
        //     return Vec3(lcg.nextFloat(min, max), lcg.nextFloat(min, max), lcg.nextFloat(min, max));
        // }

        __host__ __device__ static Vec3 random(LinearCongruentialGenerator &lcg, float min, float max)
        {
            return Vec3(lcg.nextFloat(min, max), lcg.nextFloat(min, max), lcg.nextFloat(min, max));
        }

        // ----------------------------------------------------------------
        // --- Public attributes,
        float values[3];

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

    __host__ inline std::ostream &operator<<(std::ostream &out, const Vec3 &v)
    {
        return out << v.x() << ' ' << v.y() << ' ' << v.z();
    }

    __host__ __device__ inline Vec3 operator+(const Vec3 &u, const Vec3 &v)
    {
        return Vec3(u.x() + v.x(), u.y() + v.y(), u.z() + v.z());
    }

    __host__ __device__ inline Vec3 operator-(const Vec3 &u, const Vec3 &v)
    {
        return Vec3(u.x() - v.x(), u.y() - v.y(), u.z() - v.z());
    }

    __host__ __device__ inline Vec3 operator*(const Vec3 &u, const Vec3 &v)
    {
        return Vec3(u.x() * v.x(), u.y() * v.y(), u.z() * v.z());
    }

    __host__ __device__ inline Vec3 operator*(float t, const Vec3 &v)
    {
        return Vec3(t * v.x(), t * v.y(), t * v.z());
    }

    __host__ __device__ inline Vec3 operator*(const Vec3 &v, float t)
    {
        return t * v;
    }

    __host__ __device__ inline Vec3 operator/(Vec3 v, float t)
    {
        return (1 / t) * v;
    }

    __host__ __device__ inline float dot(const Vec3 &u, const Vec3 &v)
    {
        return u.x() * v.x() + u.y() * v.y() + u.z() * v.z();
    }

    __host__ __device__ inline Vec3 cross(const Vec3 &u, const Vec3 &v)
    {
        return Vec3(u.y() * v.z() - u.z() * v.y(),
                    u.z() * v.x() - u.x() * v.z(),
                    u.x() * v.y() - u.y() * v.x());
    }

    __host__ __device__ inline Vec3 unitVector(Vec3 v)
    {
        return v / v.length();
    }

    __host__ __device__ inline Vec3 randomUnitVector(LinearCongruentialGenerator &lcg)
    {
        while (true)
        {
            auto p = Vec3::random(lcg, -1.0f, 1.0f);
            auto lengthSquared = p.lengthSquared();
            if (1e-160 < lengthSquared && lengthSquared <= 1)
            {
                return p / sqrtf(lengthSquared);
            }
        }
    }

    __host__ __device__ inline Vec3 randomOnHemisphere(const Vec3 &normal, LinearCongruentialGenerator &lcg)
    {
        Vec3 onUnitSphere = randomUnitVector(lcg);
        if (dot(onUnitSphere, normal) > 0.0f) // In the same hemisphere as the normal.
        {
            return onUnitSphere;
        }
        else
        {
            return -onUnitSphere;
        }
    }

    __host__ __device__ inline Vec3 randomInUnitDisk(LinearCongruentialGenerator &lcg)
    {
        while (true)
        {
            Vec3 p = Vec3(lcg.nextFloat(-1.0f, 1.0f), lcg.nextFloat(-1.0f, 1.0f), 0.0f);
            if (p.lengthSquared() < 1.0f)
            {
                return p;
            }
        }
    }

    __host__ __device__ inline Vec3 reflect(const Vec3 &v, const Vec3 &n)
    {
        return v - 2 * dot(v, n) * n;
    }

    __host__ __device__ inline Vec3 refract(
        const Vec3 &uv,
        const Vec3 &n,
        float etaiOverEtat,
        float cosTheta)
    {
        /*
        Ray refraction is described by Snell's law = η * sin(θ) = η' * sin(θ')
        *
        *                   |                                      |
        *          -_       |                                      |       _^
        *            -_     |                                      |     _-
        *              -_---|--> θ                            θ <--|---_-
        *   η            -_ |                      η               | _-
        *   _______________v|________________      ________________|-_______________
        *                   |\                                    ^|
        *   η'              | \                    η'            / |
        *            θ' <---|--\                                /--|---> θ'
        *                   |   \                              /   |
        *                   |    \                            /    |
        *                   |     v                                |
        *
        θ and θ' are the angles from the normal, and η and η'
        the refractive indexes, for determining the direction of the refracted
        ray it must be solved for sin(θ'): sin(θ') = (η / η') * sin(θ)

        In the refracted side there is a refracted ray R' and a normal n', and there is
        a refracted angle θ' between them, R' can be separated into parts,
        perpendicular and parallel relative to n': R' = R'⊥ + R'∥, solving for both:
            R'⊥ = (η / η') * (R + cos(θ) * n)
            R'∥ = -sqrt(1 - |R'⊥|² * n)
        */
        Vec3 rayOutPerpendicular = etaiOverEtat * (uv + cosTheta * n);
        Vec3 rayOutParallel = -sqrtf(fabsf(1.0f - rayOutPerpendicular.lengthSquared())) * n;
        return rayOutPerpendicular + rayOutParallel;
    }

    using Point = Vec3;

} // namespace geometry

#endif // VEC3_H