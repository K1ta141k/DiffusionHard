#include "candidate_sampler.hpp"

namespace diffusion_accel {

extern "C" std::uint64_t candidate_reveal_kernel(
    std::uint16_t token_ids[kCandidateMaxPositions],
    const std::uint16_t candidate_ids[kCandidateMaxPositions],
    std::uint64_t active_bitmap[kCandidateBitmapWords],
    std::uint64_t valid_bitmap[kCandidateBitmapWords],
    const std::uint32_t random_words[kCandidateMaxPositions],
    std::uint64_t reveal_threshold_q32,
    std::uint32_t position_count) {
#ifdef __SYNTHESIS__
#pragma HLS INTERFACE m_axi port=token_ids offset=slave bundle=canvas
#pragma HLS INTERFACE m_axi port=candidate_ids offset=slave bundle=candidate
#pragma HLS INTERFACE m_axi port=random_words offset=slave bundle=random
#pragma HLS INTERFACE m_axi port=active_bitmap offset=slave bundle=bitmap
#pragma HLS INTERFACE m_axi port=valid_bitmap offset=slave bundle=bitmap
#pragma HLS INTERFACE s_axilite port=token_ids bundle=control
#pragma HLS INTERFACE s_axilite port=candidate_ids bundle=control
#pragma HLS INTERFACE s_axilite port=random_words bundle=control
#pragma HLS INTERFACE s_axilite port=active_bitmap bundle=control
#pragma HLS INTERFACE s_axilite port=valid_bitmap bundle=control
#pragma HLS INTERFACE s_axilite port=reveal_threshold_q32 bundle=control
#pragma HLS INTERFACE s_axilite port=position_count bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
#endif

    if (position_count == 0 || position_count > kCandidateMaxPositions) {
        return pack_candidate_result(kCandidateKernelInvalidPositionCount, 0);
    }
    if (reveal_threshold_q32 > kRevealProbabilityOneQ32) {
        return pack_candidate_result(kCandidateKernelInvalidThreshold, 0);
    }

    std::uint64_t changed_bitmap[kCandidateBitmapWords] = {};
    std::uint32_t changed_count = 0;

#ifdef __SYNTHESIS__
candidate_loop:
#endif
    for (std::uint32_t position = 0; position < kCandidateMaxPositions;
         ++position) {
#ifdef __SYNTHESIS__
#pragma HLS PIPELINE II=1
#endif
        if (position >= position_count) {
            continue;
        }
        const std::uint32_t word = position >> 6;
        const std::uint32_t bit = position & 63;
        const std::uint64_t bit_mask = std::uint64_t{1} << bit;
        const bool active = (active_bitmap[word] & bit_mask) != 0;
        const bool valid = (valid_bitmap[word] & bit_mask) != 0;
        const bool reveal =
            std::uint64_t{random_words[position]} < reveal_threshold_q32;

        if (active && valid && reveal) {
            token_ids[position] = candidate_ids[position];
            active_bitmap[word] &= ~bit_mask;
            changed_bitmap[word] |= bit_mask;
            ++changed_count;
        }
    }

    if (changed_count != 0) {
#ifdef __SYNTHESIS__
invalidate_loop:
#endif
        for (std::uint32_t word = 0; word < kCandidateBitmapWords; ++word) {
#ifdef __SYNTHESIS__
#pragma HLS UNROLL
#endif
            valid_bitmap[word] = 0;
        }
    }

    return pack_candidate_result(kCandidateKernelOk, changed_count);
}

}  // namespace diffusion_accel
