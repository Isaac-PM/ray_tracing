#include "Renderer.cuh"

#define VALUE_SEPARATOR ' '
#define REGISTRY_SEPARATOR '\n'

const std::string Renderer::M_DEFAULT_IMAGE_PATH = "output.ppm";
const std::string Renderer::M_BENCHMARK_PATH = "benchmark.txt";

std::string parseQuality(BenchmarkQuality quality)
{
    switch (quality)
    {
    case LOW:
        return "low";
    case MEDIUM:
        return "medium";
    case HIGH:
        return "high";
    default:
        return "low";
    }
}

uint getSamplesPerPixel(BenchmarkQuality quality)
{
    switch (quality)
    {
    case LOW:
        return 50;
    case MEDIUM:
        return 100;
    case HIGH:
        return 200;
    default:
        return 50;
    }
}

__host__ void Renderer::generateBenchmark()
{
    // Generates a benchmark containing three big spheres and several small ones.

    std::string path = M_BENCHMARK_PATH;
    if (std::filesystem::exists(path))
    {
        std::cout << "Warning: File " << path << " already exists. Cannot generate another benchmark.\n";
        return;
    }
    std::ofstream file(path, std::ios::out);
    if (!file)
    {
        std::cout << "Error: Unable to create the file " << path << ".\n";
        return;
    }

    file << 1200 << REGISTRY_SEPARATOR;                                               // Image width (1200px * 675px).
    file << 50 << REGISTRY_SEPARATOR;                                                 // Maximum number of bounces.
    file << 20 << REGISTRY_SEPARATOR;                                                 // Camera vertical field of view.
    file << 13 << VALUE_SEPARATOR << 2 << VALUE_SEPARATOR << 3 << REGISTRY_SEPARATOR; // Camera look from.
    file << 0 << VALUE_SEPARATOR << 0 << VALUE_SEPARATOR << 0 << REGISTRY_SEPARATOR;  // Camera look at.
    file << 0 << VALUE_SEPARATOR << 1 << VALUE_SEPARATOR << 0 << REGISTRY_SEPARATOR;  // Camera view up.
    file << 0.6 << REGISTRY_SEPARATOR;                                                // Camera defocus angle.
    file << 10.0 << REGISTRY_SEPARATOR;                                               // Camera focus distance.

    auto saveSphereInRegistry = [](std::ofstream &file, const Sphere &sphere, bool lastSeparator = true)
    {
        file << (int)sphere.material->type << VALUE_SEPARATOR;
        file << sphere.center << VALUE_SEPARATOR;
        file << sphere.radius << VALUE_SEPARATOR;
        file << sphere.material->albedo;
        if (lastSeparator)
        {
            file << VALUE_SEPARATOR;
        }
    };

    auto saveLambertianSphere = [&saveSphereInRegistry](std::ofstream &file, const Sphere &sphere)
    {
        saveSphereInRegistry(file, sphere, false);
        file << REGISTRY_SEPARATOR;
    };

    auto saveMetalSphere = [&saveSphereInRegistry](std::ofstream &file, const Sphere &sphere)
    {
        saveSphereInRegistry(file, sphere);
        file << sphere.material->metal.fuzz << REGISTRY_SEPARATOR;
    };

    auto saveDielectricSphere = [&saveSphereInRegistry](std::ofstream &file, const Sphere &sphere)
    {
        saveSphereInRegistry(file, sphere);
        file << sphere.material->dielectric.refractionIndex << REGISTRY_SEPARATOR;
    };

    size_t numberOfSpheres = 22 * 22 + 1;
    file << numberOfSpheres << REGISTRY_SEPARATOR;

    auto ground = Sphere(Point(0.0, -1000.0, 0.0), 1000.0, new Material(Color(0.5, 0.5, 0.5)));
    saveLambertianSphere(file, ground);

    LinearCongruentialGenerator lcg = LinearCongruentialGenerator(1);
    for (int a = -11; a < 11; a++)
    {
        for (int b = -11; b < 11; b++)
        {
            auto chooseMaterial = randomFloat();
            Point center(a + 0.9 * randomFloat(), 0.2, b + 0.9 * randomFloat());
            if ((center - Point(4.0, 0.2, 0.0)).length() > 0.9)
            {
                if (chooseMaterial < 0.8)
                {
                    auto sphereMaterial = new Material(Color::random(lcg) * Color::random(lcg));
                    Sphere sphere(center, 0.2, sphereMaterial);
                    saveLambertianSphere(file, sphere);
                }
                else if (chooseMaterial < 0.95)
                {
                    auto sphereMaterial = new Material(Color::random(lcg, 0.5, 1.0), randomFloat(0.0, 0.5));
                    Sphere sphere(center, 0.2, sphereMaterial);
                    saveMetalSphere(file, sphere);
                }
                else
                {
                    auto sphereMaterial = new Material(1.5);
                    Sphere sphere(center, 0.2, sphereMaterial);
                    saveDielectricSphere(file, sphere);
                }
            }
        }
    }

    auto material1 = new Material(1.5);
    Sphere sphere1(Point(0.0, 1.0, 0.0), 1.0, material1);
    saveDielectricSphere(file, sphere1);

    auto material2 = new Material(Color(0.4, 0.2, 0.1));
    Sphere sphere2(Point(-4.0, 1.0, 0.0), 1.0, material2);
    saveLambertianSphere(file, sphere2);

    auto material3 = new Material(Color(0.7, 0.6, 0.5), 0.0);
    Sphere sphere3(Point(4.0, 1.0, 0.0), 1.0, material3);
    saveMetalSphere(file, sphere3);

    file.close();
    std::cout << "Benchmark generated successfully.\n";
}

__host__ Renderer *Renderer::loadBenchmark(BenchmarkQuality quality)
{
    Renderer *renderer = nullptr;
    std::string path = M_BENCHMARK_PATH;
    if (!std::filesystem::exists(path))
    {
        std::cout << "Warning: File " << path << " does not exist. Generating a new benchmark.\n";
        generateBenchmark();
    }
    std::ifstream file(path, std::ios::in);
    if (!file)
    {
        std::cerr << "Error: Unable to open the file " << path << ".\n";
        return renderer;
    }

    int imageWidth;
    int samplesPerPixel;
    int maxNumberOfBounces;
    float verticalFieldOfView;
    Vec3 lookFrom;
    Vec3 lookAt;
    Vec3 viewUp;
    float defocusAngle;
    float focusDistance;
    int numberOfSpheres;

    file >> imageWidth;
    samplesPerPixel = getSamplesPerPixel(quality);
    file >> maxNumberOfBounces;
    file >> verticalFieldOfView;
    file >> lookFrom.x() >> lookFrom.y() >> lookFrom.z();
    file >> lookAt.x() >> lookAt.y() >> lookAt.z();
    file >> viewUp.x() >> viewUp.y() >> viewUp.z();
    file >> defocusAngle;
    file >> focusDistance;
    file >> numberOfSpheres;

    Camera camera = Camera(verticalFieldOfView, lookFrom, lookAt, viewUp, defocusAngle, focusDistance);
    renderer = new Renderer(imageWidth, samplesPerPixel, maxNumberOfBounces, numberOfSpheres, camera, false);

    auto loadVec3FromRegistry = [&file]()
    {
        Vec3 vec;
        file >> vec.x() >> vec.y() >> vec.z();
        return vec;
    };

    auto loadLambertianSphere = [&loadVec3FromRegistry, &renderer, &file]()
    {
        Vec3 center = loadVec3FromRegistry();
        float radius;
        file >> radius;
        Vec3 albedo = loadVec3FromRegistry();
        Sphere sphere(center, radius, new Material(albedo));
        renderer->addSphereToWorld(sphere);
    };

    auto loadMetalSphere = [&loadVec3FromRegistry, &renderer, &file]()
    {
        Vec3 center = loadVec3FromRegistry();
        float radius;
        file >> radius;
        Vec3 albedo = loadVec3FromRegistry();
        float fuzz;
        file >> fuzz;
        Sphere sphere(center, radius, new Material(albedo, fuzz));
        renderer->addSphereToWorld(sphere);
    };

    auto loadDielectricSphere = [&loadVec3FromRegistry, &renderer, &file]()
    {
        Vec3 center = loadVec3FromRegistry();
        float radius;
        file >> radius;
        Vec3 albedo = loadVec3FromRegistry();
        float refractionIndex;
        file >> refractionIndex;
        Sphere sphere(center, radius, new Material(refractionIndex));
        renderer->addSphereToWorld(sphere);
    };

    auto loadSphere = [&file, &loadLambertianSphere, &loadMetalSphere, &loadDielectricSphere]()
    {
        int materialType;
        file >> materialType;
        MaterialType type = static_cast<MaterialType>(materialType);
        switch (type)
        {
        case MaterialType::LAMBERTIAN:
            loadLambertianSphere();
            break;
        case MaterialType::METAL:
            loadMetalSphere();
            break;
        case MaterialType::DIELECTRIC:
            loadDielectricSphere();
            break;
        default:
            break;
        }
    };

    auto printSphere = [](const Sphere &sphere)
    {
        std::cout << "\tSphere: " << sphere.center << ", " << sphere.radius << ", " << (int)sphere.material->type << '\n';
        switch (sphere.material->type)
        {
        case MaterialType::LAMBERTIAN:
            std::cout << "\t\tAlbedo: " << sphere.material->albedo << '\n';
            break;
        case MaterialType::METAL:
            std::cout << "\t\tAlbedo: " << sphere.material->albedo << ", Fuzz: " << sphere.material->metal.fuzz << '\n';
            break;
        case MaterialType::DIELECTRIC:
            std::cout << "\t\tRefraction index: " << sphere.material->dielectric.refractionIndex << '\n';
            break;
        default:
            break;
        }
    };

    for (int i = 0; i < numberOfSpheres; i++)
    {
        loadSphere();
    }

    std::cout << "Benchmark loaded successfully.\n";
    std::cout << "Benchmark specifications:\n";
    std::cout << "\tImage width: " << imageWidth << '\n';
    std::cout << "\tSamples per pixel: " << samplesPerPixel << '\n';
    std::cout << "\tMaximum number of bounces: " << maxNumberOfBounces << '\n';
    std::cout << "\tCamera vertical field of view: " << verticalFieldOfView << '\n';
    std::cout << "\tCamera look from: " << lookFrom << '\n';
    std::cout << "\tCamera look at: " << lookAt << '\n';
    std::cout << "\tCamera view up: " << viewUp << '\n';
    std::cout << "\tCamera defocus angle: " << defocusAngle << '\n';
    std::cout << "\tCamera focus distance: " << focusDistance << '\n';
    std::cout << "\tNumber of spheres: " << numberOfSpheres << '\n';

    std::cout << "First five spheres:\n";
    for (size_t i = 0; i < 5; i++)
    {
        printSphere(renderer->world().getSphere(i));
    }

    file.close();
    return renderer;
}