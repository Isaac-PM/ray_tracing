#include "Interval.cuh"

namespace geometry
{
    const Interval Interval::EMPTY = Interval(+INFINITY_VALUE, -INFINITY_VALUE);
    const Interval Interval::UNIVERSE = Interval(-INFINITY_VALUE, +INFINITY_VALUE);
} // namespace geometry