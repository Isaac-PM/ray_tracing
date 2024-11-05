#include "Renderer.cuh"
#include "Material.cuh"

int main(void)
{
    Renderer *renderer = Renderer::loadBenchmark();
    renderer->renderCPU();
    renderer->saveRenderedImage();
    // TODO: Free memory
    return EXIT_SUCCESS;
}