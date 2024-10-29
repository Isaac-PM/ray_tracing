#include "Renderer.cuh"

int main(void)
{
    Renderer renderer = Renderer(false);
    renderer.addToWorld(Sphere(Vec3(0.0f, 0.0f, -1.0f), 0.5f));
    renderer.addToWorld(Sphere(Vec3(0.0f, -100.5f, -1.0f), 100.0f));
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