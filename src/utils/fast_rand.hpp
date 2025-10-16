#pragma once

#include <random>

#include "random/pcg_random.hpp"

namespace rng
{
    // one generator per thread, zero contention
    inline pcg32_fast &tl_engine()
    {
        thread_local pcg32_fast eng{
            pcg_extras::seed_seq_from<std::random_device>() // good seed once
        };
        return eng;
    }

    // inclusive [min,max]
    inline int irange(int min, int max)
    {
        std::uniform_int_distribution<int> dist(min, max);
        return dist(tl_engine());
    }

    // floating point [0,1)
    inline float urand()
    {
        std::uniform_real_distribution<float> dist(0.f, 1.f);
        return dist(tl_engine());
    }

    // 0-99 (% style)
    inline bool chance(int percent) // chance(30) = 30 % true
    {
        std::uniform_int_distribution<int> dist(0, 99);
        return dist(tl_engine()) < percent;
    }
} // namespace rng
