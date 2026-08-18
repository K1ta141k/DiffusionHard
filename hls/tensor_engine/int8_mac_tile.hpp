#pragma once

#include <cstdint>

namespace diffusion_accel {

constexpr std::uint32_t kTensorTileMLanes = 4;
constexpr std::uint32_t kTensorTileNLanes = 8;
constexpr std::uint32_t kTensorTileKLanes = 32;
constexpr std::uint32_t kTensorTileOutputs =
    kTensorTileMLanes * kTensorTileNLanes;

extern "C" void int8_mac_tile_kernel(
    const std::int8_t activations[kTensorTileMLanes * kTensorTileKLanes],
    const std::int8_t weights[kTensorTileNLanes * kTensorTileKLanes],
    bool clear_accumulators,
    std::int32_t accumulators[kTensorTileOutputs]);

}  // namespace diffusion_accel
