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
            float verticalFieldOfView = 90,
            const geometry::Point &lookFrom = geometry::Point(0.0f, 0.0f, 0.0f),
            const geometry::Point &lookAt = geometry::Point(0.0f, 0.0f, -1.0f),
            const geometry::Vec3 &viewUp = geometry::Vec3(0.0f, 1.0f, 0.0f),
            float defocusAngle = 0.0f,
            float focusDistance = 0.0f)
            : verticalFieldOfView(verticalFieldOfView),
              lookFrom(lookFrom),
              lookAt(lookAt),
              viewUp(viewUp),
              defocusAngle(defocusAngle),
              focusDistance(focusDistance),
              m_center(lookFrom)
        {
            computeAttributes();
        }

        __host__ __device__ const geometry::Vec3 &center() const { return m_center; }

        __host__ __device__ void computeAttributes()
        {
            m_center = lookFrom;
            w = geometry::unitVector(lookFrom - lookAt);
            u = geometry::unitVector(geometry::cross(viewUp, w));
            v = geometry::cross(w, u);
            auto defocusRadius = focusDistance * tanf(geometry::degreesToRadians(defocusAngle / 2));
            defocusDiskU = u * defocusRadius;
            defocusDiskV = v * defocusRadius;
        }

        // ----------------------------------------------------------------
        // --- Public attributes
        float verticalFieldOfView; // Visual angle from edge to edge of the rendered image, in degrees.

        geometry::Point lookFrom;
        geometry::Point lookAt;
        geometry::Vec3 viewUp;
        geometry::Vec3 u, v, w;
        /*

        *
        *                     ^
        *             View up |    ^
        *                     | V /
        *                  ___|__/________
        *                 /   | /        /
        *                /    |/        /
        *   Look from   / <---*--->    /
        *              /     / \-_    /
        *             /_________\_-__/
        *                        \    -_
        *                       U \     v
        *                          \     * Look at
        *                           v
        *
        *       View up ^
        *            ---|-_____
        *           /   |      -----_____
        *          /    |      __-^     -----_____
        *       W /     |   __- V                 /
        *   <----/------|__-                     /
        *       /       *                       /
        *      -----_____                      /
        *                -----_____           /
        *                          -----_____/
        *

        */

        float defocusAngle;
        float focusDistance;
        geometry::Vec3 defocusDiskU;
        geometry::Vec3 defocusDiskV;
        /*
        Defocus blur (depth of field): is the aberration in
        which an image is out of focus.
        The distance between the camera center and the plane
        where everything is in perfect focus is called focus distance.

        Thin lens approximation: a lens with a thickness that is
        negligible compared to the radii of curvature (radius of an imaginary
        circle at a specific point that best matches the curve's
        shape at that point).
        *
        *                            |
        *             Lens           |
        *   |      _  * __           |
        *   |    _-  | |  --__       |
        *   |  _-   |   |     --__   |
        *   | -_    |   |       __-- |
        *   |   -_  |   |   __--     |
        *   |     -_ | |__--         |
        *   |         *              |
        *   Film      ^              |
        *             |              |
        *             |              Focus plane
        *      <------|------>
        *      Inside | Outside
        *             |
        *
        *              __
        *             |  __
        *     Lens    |    __
        *     * __    |      __
        *    | |  --__|        __
        *   |   |     |-__       __
        *   |   |   __|-           __
        *    | |__--  |__            |
        *     *           __         | Virtual film
        *                    __      | plane at the
        *                      __    | focus plane
        *                        __  |
        *                          __|
        *

        */

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
            const Camera &camera = Camera())
            : height(height),
              camera(camera)
        {
            float theta = geometry::degreesToRadians(camera.verticalFieldOfView);
            float h = tanf(theta / 2.0f);
            height = 2 * h * camera.focusDistance;
            width = height * (float(image.width()) / image.height());

            u = width * camera.u;
            pixelDeltaU = u / image.width();

            v = height * -camera.v;
            pixelDeltaV = v / image.height();

            upperLeft = camera.center() - (camera.focusDistance * camera.w) - (u / 2) - (v / 2);

            pixel00Location = upperLeft + 0.5f * (pixelDeltaU + pixelDeltaV);
        }

        // ----------------------------------------------------------------
        // --- Public attributes
        float height;
        float width;
        Camera camera;
        Vec3 u;
        Vec3 pixelDeltaU;
        Vec3 v;
        Vec3 pixelDeltaV;
        Vec3 upperLeft;
        Vec3 pixel00Location;

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
} // namespace graphics

#endif // VIEWPORT_H