#include "int8_mac_tile.hpp"

#include <cstdint>

namespace diffusion_accel {

extern "C" void int8_mac_tile_kernel(
    const std::int8_t activations[kTensorTileMLanes * kTensorTileKLanes],
    const std::int8_t weights[kTensorTileNLanes * kTensorTileKLanes],
    bool clear_accumulators,
    std::int32_t accumulators[kTensorTileOutputs]) {
#ifdef __SYNTHESIS__
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS ARRAY_PARTITION variable=activations complete dim=1
#pragma HLS ARRAY_PARTITION variable=weights complete dim=1
#pragma HLS ARRAY_PARTITION variable=accumulators complete dim=1
#pragma HLS PIPELINE II=1
#endif

    for (std::uint32_t m = 0; m < kTensorTileMLanes; ++m) {
#ifdef __SYNTHESIS__
#pragma HLS UNROLL
#endif
        for (std::uint32_t n = 0; n < kTensorTileNLanes; ++n) {
#ifdef __SYNTHESIS__
#pragma HLS UNROLL
#endif
            std::int32_t sum = 0;
            for (std::uint32_t k = 0; k < kTensorTileKLanes; ++k) {
#ifdef __SYNTHESIS__
#pragma HLS UNROLL
#endif
                sum += std::int32_t{activations[m * kTensorTileKLanes + k]}
                    * std::int32_t{weights[n * kTensorTileKLanes + k]};
            }
            const std::uint32_t output = m * kTensorTileNLanes + n;
            accumulators[output] =
                clear_accumulators ? sum : accumulators[output] + sum;
        }
    }
}

}  // namespace diffusion_accel
