#ifndef TIMER_H
#define TIMER_H

#include <iostream>

enum TimeUnit
{
    SECONDS,
    MILLISECONDS,
    CLOCK_TICKS
};

class Timer
{
public:
    // ----------------------------------------------------------------
    // --- Public methods
    __host__ Timer()
    {
        m_start = 0;
        m_end = 0;
    }

    __host__ ~Timer() = default;

    __host__ void resume()
    {
        m_start = clock();
    }

    __host__ void pause()
    {
        m_end = clock();
    }

    __host__ void reset()
    {
        m_start = 0;
        m_end = 0;
    }

    __host__ double elapsed(TimeUnit timeUnit = TimeUnit::SECONDS)
    {
        double elapsed = static_cast<double>(m_end - m_start);
        switch (timeUnit)
        {
        case TimeUnit::SECONDS:
            elapsed /= CLOCKS_PER_SEC;
            break;
        case TimeUnit::MILLISECONDS:
            elapsed /= (CLOCKS_PER_SEC / 1000);
            break;
        case TimeUnit::CLOCK_TICKS:
            break;
        default:
            break;
        }
        return elapsed;
    }

    __host__ double print(std::string task, TimeUnit timeUnit = TimeUnit::MILLISECONDS)
    {
        double elapsed = this->elapsed(timeUnit);
        std::cout << task << " took " << elapsed << " ";
        switch (timeUnit)
        {
        case TimeUnit::SECONDS:
            std::cout << "seconds\n";
            break;
        case TimeUnit::MILLISECONDS:
            std::cout << "milliseconds\n";
            break;
        case TimeUnit::CLOCK_TICKS:
            std::cout << "clock ticks\n";
            break;
        default:
            break;
        }
        return elapsed;
    }

    // ----------------------------------------------------------------
    // --- Public attributes

    // ----------------------------------------------------------------
    // --- Public class constants

private:
    // ----------------------------------------------------------------
    // --- Private methods
    clock_t m_start;
    clock_t m_end;

    // ----------------------------------------------------------------
    // --- Private attributes

    // ----------------------------------------------------------------
    // --- Private class constants
};

#endif // TIMER_H