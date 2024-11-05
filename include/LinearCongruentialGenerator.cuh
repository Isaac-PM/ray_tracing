#ifndef LINEAR_CONGRUENTIAL_GENERATOR_H
#define LINEAR_CONGRUENTIAL_GENERATOR_H

using Seed = unsigned long long;

class LinearCongruentialGenerator
{
public:
    // ----------------------------------------------------------------
    // --- Public methods
    __host__ __device__ LinearCongruentialGenerator(Seed initialSeed = 0)
        : m_seed(initialSeed)
    {
    }

    __host__ __device__ uint next()
    {
        m_seed = (M_MULTIPLIER * m_seed + M_INCREMENT) % M_MODULUS;
        return static_cast<uint>(m_seed);
    }

    __host__ __device__ uint next(uint min, uint max)
    {
        return min + next() % (max - min);
    }

    __host__ __device__ float nextFloat(float min = 0.0f, float max = 1.0f)
    {
        uint randomValue = next();
        float normalizedValue = static_cast<float>(randomValue) / static_cast<float>(M_MODULUS);
        return min + normalizedValue * (max - min);
    }

    __host__ __device__ void setSeed(Seed seed)
    {
        m_seed = seed;
    }

    // ----------------------------------------------------------------
    // --- Public attributes

    // ----------------------------------------------------------------
    // --- Public class constants

private:
    // ----------------------------------------------------------------
    // --- Private methods
    Seed m_seed;

    // ----------------------------------------------------------------
    // --- Private attributes

    // ----------------------------------------------------------------
    // --- Private class constants
    static constexpr Seed M_MULTIPLIER = 1664525;
    static constexpr Seed M_INCREMENT = 1013904223;
    static constexpr Seed M_MODULUS = 4294967296;
};

#endif // LINEAR_CONGRUENTIAL_GENERATOR_H