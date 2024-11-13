#include <iostream>
#include <string>
#include <cstdlib>
#include "Renderer.cuh"
#include "Timer.cuh"

void printUsage(const std::string &programName)
{
    std::cout << "Usage: " << programName << " <benchmark_quality> <render_mode>\n";
    std::cout << "benchmark_quality:\n";
    std::cout << "  1 -> LOW\n";
    std::cout << "  2 -> MEDIUM\n";
    std::cout << "  3 -> HIGH\n";
    std::cout << "render_mode:\n";
    std::cout << "  cpu -> Render on CPU\n";
    std::cout << "  gpu -> Render on GPU\n";
}

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        printUsage(argv[0]);
        return 1;
    }

    int quality = std::atoi(argv[1]);
    if (quality < 1 || quality > 3)
    {
        printUsage(argv[0]);
        return 1;
    }

    std::string renderMode = argv[2];
    if (renderMode != "cpu" && renderMode != "gpu")
    {
        printUsage(argv[0]);
        return 1;
    }

    Renderer *renderer = Renderer::loadBenchmark(static_cast<BenchmarkQuality>(quality));
    Timer renderTimer;
    renderTimer.resume();

    if (renderMode == "cpu")
    {
        renderer->renderCPU();
    }
    else if (renderMode == "gpu")
    {
        renderer->renderGPU();
    }

    renderTimer.pause();
    renderTimer.print("Render time", TimeUnit::SECONDS);

    delete renderer;
    std::cout << "Done!\n";
    return EXIT_SUCCESS;
}