#include <iostream>
#include <string>
#include <cstdlib>
#include "Renderer.cuh"

void printUsage(const std::string &programName)
{
    std::cout << "Usage: " << programName << " <benchmark_quality>\n";
    std::cout << "benchmark_quality:\n";
    std::cout << "  1 -> LOW\n";
    std::cout << "  2 -> MEDIUM\n";
    std::cout << "  3 -> HIGH\n";
}

int main(int argc, char *argv[])
{
    if (argc != 2)
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

    Renderer *renderer = Renderer::loadBenchmark(static_cast<BenchmarkQuality>(quality));
    renderer->renderCPU();
    renderer->saveRenderedImage();

    // TODO: Free memory
    return EXIT_SUCCESS;

    return 0;
}

#include "Renderer.cuh"
#include "Material.cuh"