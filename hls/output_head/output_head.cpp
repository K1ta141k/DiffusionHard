#include "output_head.hpp"

#include <cstdint>
#include <limits>

namespace diffusion_accel {

namespace {

std::int32_t requantize_q20(std::int32_t accumulator,
                            std::int32_t multiplier_q20) {
    const std::int64_t product =
        std::int64_t{accumulator} * multiplier_q20;
    constexpr std::int64_t half = std::int64_t{1} << 19;
    if (product >= 0) {
        return static_cast<std::int32_t>((product + half) >> 20);
    }
    return -static_cast<std::int32_t>((-product + half) >> 20);
}

}  // namespace

extern "C" std::uint32_t output_head_int8_kernel(
    const std::int8_t* hidden,
    const std::int8_t* weights,
    const std::int32_t* requant_multiplier_q20,
    const std::int32_t* bias,
    OutputHeadNoiseStream& race_noise,
    std::uint16_t candidate_ids[kOutputHeadMaxPositions],
    std::uint32_t position_count,
    std::uint32_t hidden_size,
    std::uint32_t vocabulary_size,
    std::uint32_t mask_token_id) {
#ifdef __SYNTHESIS__
#pragma HLS INTERFACE m_axi port=hidden offset=slave bundle=activation
#pragma HLS INTERFACE m_axi port=weights offset=slave bundle=weight
#pragma HLS INTERFACE m_axi port=requant_multiplier_q20 offset=slave bundle=weight
#pragma HLS INTERFACE m_axi port=bias offset=slave bundle=weight
#pragma HLS INTERFACE axis port=race_noise
#pragma HLS INTERFACE m_axi port=candidate_ids offset=slave bundle=candidate
#pragma HLS INTERFACE s_axilite port=hidden bundle=control
#pragma HLS INTERFACE s_axilite port=weights bundle=control
#pragma HLS INTERFACE s_axilite port=requant_multiplier_q20 bundle=control
#pragma HLS INTERFACE s_axilite port=bias bundle=control
#pragma HLS INTERFACE s_axilite port=candidate_ids bundle=control
#pragma HLS INTERFACE s_axilite port=position_count bundle=control
#pragma HLS INTERFACE s_axilite port=hidden_size bundle=control
#pragma HLS INTERFACE s_axilite port=vocabulary_size bundle=control
#pragma HLS INTERFACE s_axilite port=mask_token_id bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
#endif

    if (position_count == 0 || position_count > kOutputHeadMaxPositions) {
        return kOutputHeadKernelInvalidPositionCount;
    }
    if (hidden_size == 0 || hidden_size > kOutputHeadMaxHidden) {
        return kOutputHeadKernelInvalidHiddenSize;
    }
    if (vocabulary_size == 0 ||
        vocabulary_size > kOutputHeadMaxVocabulary) {
        return kOutputHeadKernelInvalidVocabularySize;
    }
    if (mask_token_id >= vocabulary_size) {
        return kOutputHeadKernelInvalidMaskToken;
    }

    std::int8_t hidden_buffer[kOutputHeadMaxPositions][kOutputHeadMaxHidden];
    std::int32_t accumulators[kOutputHeadMaxPositions]
                             [kOutputHeadVocabularyTile];
    std::int64_t best_scores[kOutputHeadMaxPositions];
#ifdef __SYNTHESIS__
#pragma HLS BIND_STORAGE variable=hidden_buffer type=ram_2p impl=bram
#pragma HLS ARRAY_PARTITION variable=accumulators complete dim=2
#pragma HLS ARRAY_PARTITION variable=best_scores complete dim=1
#endif

    for (std::uint32_t position = 0; position < position_count; ++position) {
        best_scores[position] = std::numeric_limits<std::int64_t>::min();
        candidate_ids[position] = 0;
        for (std::uint32_t hidden_index = 0; hidden_index < hidden_size;
             ++hidden_index) {
#ifdef __SYNTHESIS__
#pragma HLS PIPELINE II=1
#endif
            hidden_buffer[position][hidden_index] =
                hidden[position * hidden_size + hidden_index];
        }
    }

    for (std::uint32_t vocabulary_base = 0;
         vocabulary_base < vocabulary_size;
         vocabulary_base += kOutputHeadVocabularyTile) {
        for (std::uint32_t position = 0; position < position_count;
             ++position) {
#ifdef __SYNTHESIS__
#pragma HLS PIPELINE II=1
#endif
            for (std::uint32_t lane = 0; lane < kOutputHeadVocabularyTile;
                 ++lane) {
                accumulators[position][lane] = 0;
            }
        }

        for (std::uint32_t lane = 0; lane < kOutputHeadVocabularyTile;
             ++lane) {
            const std::uint32_t token = vocabulary_base + lane;
            if (token >= vocabulary_size) {
                continue;
            }
            for (std::uint32_t hidden_index = 0; hidden_index < hidden_size;
                 ++hidden_index) {
                const std::int32_t weight =
                    weights[token * hidden_size + hidden_index];
                for (std::uint32_t position = 0; position < position_count;
                     ++position) {
#ifdef __SYNTHESIS__
#pragma HLS PIPELINE II=1
#endif
                    accumulators[position][lane] +=
                        std::int32_t{hidden_buffer[position][hidden_index]} *
                        weight;
                }
            }
        }

        for (std::uint32_t position = 0; position < position_count;
             ++position) {
            for (std::uint32_t lane = 0; lane < kOutputHeadVocabularyTile;
                 ++lane) {
#ifdef __SYNTHESIS__
#pragma HLS PIPELINE II=1
#endif
                const std::uint32_t token = vocabulary_base + lane;
                if (token >= vocabulary_size) {
                    continue;
                }
                const std::int32_t noise = race_noise.read();
                if (token == mask_token_id) {
                    continue;
                }
                const std::int32_t logit_q10 = requantize_q20(
                    accumulators[position][lane],
                    requant_multiplier_q20[token]);
                const std::int64_t score =
                    std::int64_t{logit_q10} +
                    std::int64_t{bias[token]} + std::int64_t{noise};
                if (score > best_scores[position]) {
                    best_scores[position] = score;
                    candidate_ids[position] =
                        static_cast<std::uint16_t>(token);
                }
            }
        }
    }

    return kOutputHeadKernelOk;
}

}  // namespace diffusion_accel
