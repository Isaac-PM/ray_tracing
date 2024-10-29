#ifndef RGBPIXEL_H
#define RGBPIXEL_H

#include <iostream>
#include <fstream>
#include <stdint.h>
#include <string>
#include "Vec3.cuh"

namespace graphics
{
    using ColorChannel = uint8_t;

    using Color = geometry::Vec3; // Represents a color with not integer values for each channel, [0, 1] range.

    class RGBPixel
    {
    public:
        // ----------------------------------------------------------------
        // --- Public methods
        __host__ __device__ RGBPixel(ColorChannel r = 0, ColorChannel g = 0, ColorChannel b = 0)
            : r(r), g(g), b(b) {}

        __host__ __device__ RGBPixel(const Color &color)
        {
            r = static_cast<ColorChannel>(255.999f * color.x());
            g = static_cast<ColorChannel>(255.999f * color.y());
            b = static_cast<ColorChannel>(255.999f * color.z());
        }

        // ----------------------------------------------------------------
        // --- Public attributes
        ColorChannel r;
        ColorChannel g;
        ColorChannel b;

        // ----------------------------------------------------------------
        // --- Public class constants
        static const size_t MAX_VALUE = 255;
        static const size_t MIN_VALUE = 0;

    private:
        // ----------------------------------------------------------------
        // --- Private methods

        // ----------------------------------------------------------------
        // --- Private attributes

        // ----------------------------------------------------------------
        // --- Private class constants
    };

    __host__ inline std::ostream &operator<<(std::ostream &out, const RGBPixel &p)
    {
        return out << static_cast<int>(p.r) << ' ' << static_cast<int>(p.g) << ' ' << static_cast<int>(p.b);
    }

    __host__ __device__ inline Color COLOR_LIGHT_BLUE() { return Color(0.5f, 0.7f, 1.0f); }
    __host__ __device__ inline Color COLOR_LIGHT_GREEN() { return Color(0.5f, 1.0f, 0.5f); }
    __host__ __device__ inline Color COLOR_RED() { return Color(1.0f, 0.0f, 0.0f); }
    __host__ __device__ inline Color COLOR_GREEN() { return Color(0.0f, 1.0f, 0.0f); }
    __host__ __device__ inline Color COLOR_BLUE() { return Color(0.0f, 0.0f, 1.0f); }
    __host__ __device__ inline Color COLOR_WHITE() { return Color(1.0f, 1.0f, 1.0f); }
    __host__ __device__ inline Color COLOR_BLACK() { return Color(0.0f, 0.0f, 0.0f); }

} // namespace graphics

#endif // RGBPIXEL_H