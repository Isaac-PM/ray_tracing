#ifndef VEC3_H
#define VEC3_H

#include <cmath>
#include <iostream>
#include <limits>

namespace geometry
{
    constexpr float INFINITY_VALUE = std::numeric_limits<float>::infinity();
    constexpr float PI = 3.141592f;

    __host__ __device__ inline float degreesToRadians(float degrees)
    {
        return degrees * PI / 180.0f;
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

        __host__ __device__ float length() const
        {
            return sqrtf(lengthSquared());
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

    using Point = Vec3;

} // namespace geometry

#endif // VEC3_H