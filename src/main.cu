#include "Renderer.cuh"
#include "Material.cuh"

int main(void)
{
    auto materialGround = new Lambertian(Color(0.8, 0.8, 0.0));
    auto materialCenter = new Lambertian(Color(0.1, 0.2, 0.5));
    auto materialLeft = new Metal(Color(0.8, 0.8, 0.8), 0.3);
    auto materialRight = new Metal(Color(0.8, 0.6, 0.2), 1.0);

    Renderer renderer = Renderer(false);
    renderer.generateSampleSquareDistribution();

    renderer.addToWorld(Sphere(Point(0.0, -100.5, -1.0), 100, materialGround));
    renderer.addToWorld(Sphere(Point(0.0, 0.0, -1.2), 0.5, materialCenter));
    renderer.addToWorld(Sphere(Point(-1.0, 0.0, -1.0), 0.5, materialLeft));
    renderer.addToWorld(Sphere(Point(1.0, 0.0, -1.0), 0.5, materialRight));

    renderer.renderCPU();
    renderer.saveRenderedImage();
    return EXIT_SUCCESS;
}

/*
| **Category**           | **Convention**                 | **Example**              |
|------------------------|--------------------------------|--------------------------|
| Local Variable         | `camelCase`                    | `counter`                |
| Member Variable        | `_` or `m_` prefix             | `_id`, `m_name`          |
| Global Variable        | `g_` prefix                    | `g_maxValue`             |
| Constant               | `UPPER_CASE_SNAKE_CASE`        | `MAX_BUFFER_SIZE`        |
| Function / Method      | `camelCase`                    | `calculateSum()`         |
| Class / Struct         | `PascalCase`                   | `Person`                 |
| Enum Type              | `PascalCase`                   | `Color`                  |
| Enum Value             | `UPPER_CASE_SNAKE_CASE`        | `COLOR_RED`              |
| Namespace              | `lower_case` / `camelCase`     | `physics`                |
| Template Parameter     | `PascalCase`                   | `template <typename T>`  |
| Macro                  | `UPPER_CASE_SNAKE_CASE`        | `#define MAX_SIZE 100`   |
| File Name              | `snake_case` or `PascalCase`   | `person.hpp`             |
| Exception Class        | `PascalCase` + `Exception`     | `FileNotFoundException`  |
*/

/*
public:
// ----------------------------------------------------------------
// --- Public methods

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
*/

/*
a ray tracer sends rays through pixels and computes the color seen in the direction of those rays.


viewport es un ectangulo 2d sobre la que se proyecta una escena 3d
*/