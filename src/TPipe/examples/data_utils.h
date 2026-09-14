//===- data_utils.h - host-side helpers for the examples ------------------===//
#ifndef MINI_ASCENDC_DATA_UTILS_H
#define MINI_ASCENDC_DATA_UTILS_H

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

/// Stands in for aclrtMalloc: "device" memory is just host memory here.
inline std::vector<float> MakeInput(uint32_t n, uint32_t seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> v(n);
    for (uint32_t i = 0; i < n; ++i) { v[i] = dist(rng); }
    return v;
}

inline bool Verify(const char *name, const std::vector<float> &got, const std::vector<float> &want)
{
    for (size_t i = 0; i < want.size(); ++i) {
        if (std::fabs(got[i] - want[i]) > 1e-6f) {
            std::printf("  FAIL %s: element %zu is %g, expected %g\n", name, i, got[i], want[i]);
            return false;
        }
    }
    std::printf("  PASS %s: %zu elements match\n", name, want.size());
    return true;
}

inline std::vector<float> AddReference(const std::vector<float> &x, const std::vector<float> &y)
{
    std::vector<float> z(x.size());
    for (size_t i = 0; i < x.size(); ++i) { z[i] = x[i] + y[i]; }
    return z;
}

#endif  // MINI_ASCENDC_DATA_UTILS_H
