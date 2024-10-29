#ifndef VIEWPORT_H
#define VIEWPORT_H

#include "PPMImage.cuh"

namespace graphics
{
    class Camera
    {
    public:
        // ----------------------------------------------------------------
        // --- Public methods
        __host__ __device__ Camera(
            const geometry::Vec3 &center = geometry::Vec3(0.0f, 0.0f, 0.0f))
            : m_center(center) {}

        __host__ __device__ const geometry::Vec3 &center() const { return m_center; }

        // ----------------------------------------------------------------
        // --- Public attributes

        // ----------------------------------------------------------------
        // --- Public class constants

    private:
        // ----------------------------------------------------------------
        // --- Private methods

        // ----------------------------------------------------------------
        // --- Private attributes
        geometry::Vec3 m_center;

        // ----------------------------------------------------------------
        // --- Private class constants
    };

    class Viewport
    {
        /*
        The viewport is a 2D rectangle in 3D space, where a 3D scene is projected.
        It contains the grid of image pixel locations.
        The camera center is a point in 3D space from which all scene rays originate.
        *
        *                   ^ Up
        *                   |
        *                   |
        *                   |                __ Viewport
        *                   |               |  __
        *                   |               |    __
        *                   |               |      __
        *                   |               |        __
        *   Camera center   |---------------->         __
        *                   -_              |__          __
        *                     -_               __          |
        *                       -_               __        |
        *                         -_               __      |
        *                           -_               __    |
        *                             -_               __  |
        *                               v Right          __|
        *
        In this case, the camera center will be (0, 0, 0) (right-handed coordinates),
        where the y-axis goes up, the x-axis goes to the right, and the z-axis goes into the screen.
        This comes with the problem that the coordinate system of the images is top-left oriented.
        For navigating the image grid, two vectors are defined: u and v.
        *
        *   *-------------------------------------> Vu
        *   | Q___________________________________  _
        *   | | *P  *   *   *   *   *   *   *   * |  | deltaV
        *   | | *   *   *   *   *   *   *   *   * | _|
        *   | | *   *   *   *   *   *   *   *   * |
        *   | | *   *   *   *   *   *   *   *   * |
        *   | | *   *   *   *   *   *   *   *   * |
        *   | | *   *   *   *   *   *   *   *   * |
        *   | |_*___*___*___*___*___*___*___*___*_| Viewport
        *   |   |___| deltaU
        *   v Vv
        *
        *   - Q is the upper left corner of the viewport.
        *   - P is a pixel location.
        *
        */
        using Vec3 = geometry::Vec3;

    public:
        // ----------------------------------------------------------------
        // --- Public methods
        __host__ __device__ Viewport() {}

        __host__ __device__ Viewport(
            const PPMImage &image,
            float height = DEFAULT_HEIGHT,
            float focalLength = DEFAULT_FOCAL_LENGTH,
            const Camera &camera = Camera())
            : height(height),
              focalLength(focalLength),
              camera(camera)
        {
            width = height * (float(image.width()) / image.height());

            u = Vec3(width, 0.0f, 0.0f);
            pixelDeltaU = u / image.width();

            v = Vec3(0.0f, -height, 0.0f);
            pixelDeltaV = v / image.height();

            upperLeft = camera.center() - Vec3(0.0f, 0.0f, focalLength) - (u / 2) - (v / 2);

            pixel00Location = upperLeft + 0.5f * (pixelDeltaU + pixelDeltaV);
        }

        // ----------------------------------------------------------------
        // --- Public attributes
        float height;
        float width;
        float focalLength;
        Camera camera;
        Vec3 u;
        Vec3 pixelDeltaU;
        Vec3 v;
        Vec3 pixelDeltaV;
        Vec3 upperLeft;
        Vec3 pixel00Location;

        // ----------------------------------------------------------------
        // --- Public class constants
        static constexpr float DEFAULT_HEIGHT = 2.0f;
        static constexpr float DEFAULT_FOCAL_LENGTH = 1.0f;

    private:
        // ----------------------------------------------------------------
        // --- Private methods

        // ----------------------------------------------------------------
        // --- Private attributes

        // ----------------------------------------------------------------
        // --- Private class constants
    };
} // namespace graphics

#endif // VIEWPORT_H