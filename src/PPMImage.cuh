#ifndef PPMIMAGE_H
#define PPMIMAGE_H

#include "RGBPixel.cuh"

namespace graphics
{
    const std::string FILE_FORMAT_PPM = "P3";

    class PPMImage
    {
    public:
        // ----------------------------------------------------------------
        // --- Public methods
        __host__ __device__ PPMImage(
            size_t columns = DEFAULT_COLUMNS,
            float aspectRatio = DEFAULT_ASPECT_RATIO,
            RGBPixel *pixelsPtr = nullptr)
            : columns(columns),
              aspectRatio(aspectRatio),
              pixels(pixelsPtr),
              maxChannelValue(RGBPixel::MAX_VALUE)
        {
            rows = size_t(columns / aspectRatio);
            rows = (rows < 1 ? 1 : rows);
            if (pixels == nullptr)
            {
                pixels = new RGBPixel[pixelCount()];
                for (size_t i = 0; i < pixelCount(); i++)
                {
                    pixels[i] = RGBPixel();
                }
            }
        }

        __host__ __device__ void clear(bool usingCUDA = false)
        {
            // TODO Implement as destructor alternative with CUDA support
            delete[] pixels;
        }

        __host__ __device__ size_t pixelCount() const
        {
            return rows * columns;
        }

        __host__ __device__ size_t height() const
        {
            return rows;
        }

        __host__ __device__ size_t width() const
        {
            return columns;
        }

        __host__ __device__ RGBPixel getPixel(size_t row, size_t column) const
        {
            return pixels[row * columns + column];
        }

        __host__ __device__ void setPixel(size_t row, size_t column, const RGBPixel &pixel)
        {
            pixels[row * columns + column] = pixel;
        }

        __host__ static PPMImage load(const std::string &path)
        {

            std::fstream file(path, std::ios::in);
            if (!file.is_open())
            {
                throw std::runtime_error("File not found");
            }

            std::string imageFormat;
            getline(file, imageFormat, '\n');
            if (imageFormat != FILE_FORMAT_PPM)
            {
                throw std::runtime_error("Invalid PPM format");
            }

            PPMImage image;
            file >> image.columns >> image.rows >> image.maxChannelValue;
            if (image.maxChannelValue != RGBPixel::MAX_VALUE)
            {
                throw std::runtime_error("Invalid channel value");
            }

            for (size_t i = 0; i < image.pixelCount(); i++)
            {
                int r, g, b;
                file >> r >> g >> b;
                image.pixels[i] = RGBPixel((ColorChannel)r, (ColorChannel)g, (ColorChannel)b);
            }

            file.close();
            return image;
        }

        __host__ void save(const std::string &path) const
        {
            std::fstream file(path, std::ios::out);
            if (!file.is_open())
            {
                throw std::runtime_error("File not found");
            }

            file << FILE_FORMAT_PPM << '\n';
            file << columns << ' ' << rows << '\n';
            file << maxChannelValue << '\n';

            for (size_t i = 0; i < pixelCount(); i++)
            {
                file << pixels[i] << '\n';
            }

            file.close();
        }

        // ----------------------------------------------------------------
        // --- Public attributes
        size_t rows;
        size_t columns;
        float aspectRatio;
        RGBPixel *pixels;
        size_t maxChannelValue;

        // ----------------------------------------------------------------
        // --- Public class constants
        static const size_t DEFAULT_COLUMNS = 400;
        static constexpr float DEFAULT_ASPECT_RATIO = 16.0f / 9.0f;

    private:
        // ----------------------------------------------------------------
        // --- Private methods

        // ----------------------------------------------------------------
        // --- Private attributes

        // ----------------------------------------------------------------
        // --- Private class constants
    };
} // namespace graphics

#endif // PPMIMAGE_H