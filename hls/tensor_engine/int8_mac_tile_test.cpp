#include "int8_mac_tile.hpp"

#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>

namespace {

using diffusion_accel::int8_mac_tile_kernel;
using diffusion_accel::kTensorTileKLanes;
using diffusion_accel::kTensorTileMLanes;
using diffusion_accel::kTensorTileNLanes;
using diffusion_accel::kTensorTileOutputs;

void fill_tile(std::array<std::int8_t, kTensorTileMLanes * kTensorTileKLanes>& a,
               std::array<std::int8_t, kTensorTileNLanes * kTensorTileKLanes>& w,
               std::int32_t seed) {
    for (std::uint32_t index = 0; index < a.size(); ++index) {
        a[index] = static_cast<std::int8_t>((index * 7 + seed) % 31 - 15);
    }
    for (std::uint32_t index = 0; index < w.size(); ++index) {
        w[index] = static_cast<std::int8_t>((index * 11 + seed * 3) % 27 - 13);
    }
}

void accumulate_reference(
    const std::array<std::int8_t, kTensorTileMLanes * kTensorTileKLanes>& a,
    const std::array<std::int8_t, kTensorTileNLanes * kTensorTileKLanes>& w,
    bool clear,
    std::array<std::int32_t, kTensorTileOutputs>& accumulators) {
    for (std::uint32_t m = 0; m < kTensorTileMLanes; ++m) {
        for (std::uint32_t n = 0; n < kTensorTileNLanes; ++n) {
            std::int32_t sum = 0;
            for (std::uint32_t k = 0; k < kTensorTileKLanes; ++k) {
                sum += std::int32_t{a[m * kTensorTileKLanes + k]}
                    * std::int32_t{w[n * kTensorTileKLanes + k]};
            }
            const std::uint32_t output = m * kTensorTileNLanes + n;
            accumulators[output] = clear ? sum : accumulators[output] + sum;
        }
    }
}

}  // namespace

int main() {
    std::array<std::int8_t, kTensorTileMLanes * kTensorTileKLanes> activations{};
    std::array<std::int8_t, kTensorTileNLanes * kTensorTileKLanes> weights{};
    std::array<std::int32_t, kTensorTileOutputs> actual{};
    std::array<std::int32_t, kTensorTileOutputs> expected{};

    fill_tile(activations, weights, 3);
    int8_mac_tile_kernel(activations.data(), weights.data(), true, actual.data());
    accumulate_reference(activations, weights, true, expected);
    assert(actual == expected);

    fill_tile(activations, weights, 19);
    int8_mac_tile_kernel(activations.data(), weights.data(), false, actual.data());
    accumulate_reference(activations, weights, false, expected);
    assert(actual == expected);

    std::cout << "int8_mac_tile_test: all checks passed\n";
    return 0;
}
