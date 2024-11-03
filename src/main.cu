#include "Renderer.cuh"
#include "Material.cuh"

int main(void)
{
    auto materialGround = new Lambertian(Color(0.8, 0.8, 0.0));
    auto materialCenter = new Lambertian(Color(0.1, 0.2, 0.5));
    auto materialLeft = new Dielectric(1.5);
    auto materialBubble = new Dielectric(1.0 / 1.5);
    auto materialRight = new Metal(Color(0.8, 0.6, 0.2), 1.0);

    Camera camera = Camera();
    camera.verticalFieldOfView = 20;
    camera.lookFrom = Point(-2, 2, 1);
    camera.lookAt = Point(0, 0, -1);
    camera.viewUp = Vec3(0, 1, 0);
    camera.defocusAngle = 10;
    camera.focusDistance = 3.4;
    camera.computeAttributes();

    Renderer renderer = Renderer(camera, false);
    renderer.generateSampleSquareDistribution();

    renderer.addToWorld(Sphere(Point(0.0, -100.5, -1.0), 100, materialGround));
    renderer.addToWorld(Sphere(Point(0.0, 0.0, -1.2), 0.5, materialCenter));
    renderer.addToWorld(Sphere(Point(-1.0, 0.0, -1.0), 0.5, materialLeft));
    renderer.addToWorld(Sphere(Point(-1.0, 0.0, -1.0), 0.4, materialBubble));
    renderer.addToWorld(Sphere(Point(1.0, 0.0, -1.0), 0.5, materialRight));

    renderer.renderCPU();
    renderer.saveRenderedImage();

    return EXIT_SUCCESS;
}