#include "philox_rng.hpp"
#include "gumbel_lut.hpp"

#include <cstdint>

namespace diffusion_accel {

namespace {

std::uint32_t high_word(std::uint64_t value) {
    return static_cast<std::uint32_t>(value >> 32);
}

std::uint32_t low_word(std::uint64_t value) {
    return static_cast<std::uint32_t>(value);
}

std::uint32_t count_leading_zeros(std::uint32_t value) {
    std::uint32_t count = 32;
    for (std::uint32_t bit = 0; bit < 32; ++bit) {
#ifdef __SYNTHESIS__
#pragma HLS UNROLL
#endif
        if ((value & (std::uint32_t{1} << bit)) != 0) {
            count = 31U - bit;
        }
    }
    return count;
}

}  // namespace

PhiloxCounter philox4x32_10(PhiloxCounter counter, PhiloxKey key) {
    for (std::uint32_t round = 0; round < 10; ++round) {
#ifdef __SYNTHESIS__
#pragma HLS UNROLL
#endif
        const std::uint64_t product0 =
            std::uint64_t{kPhiloxM0} * counter[0];
        const std::uint64_t product1 =
            std::uint64_t{kPhiloxM1} * counter[2];
        counter = {
            high_word(product1) ^ counter[1] ^ key[0],
            low_word(product1),
            high_word(product0) ^ counter[3] ^ key[1],
            low_word(product0),
        };
        if (round != 9) {
            key[0] += kPhiloxW0;
            key[1] += kPhiloxW1;
        }
    }
    return counter;
}

std::uint32_t gumbel_lut_address(std::uint32_t random_word) {
    const std::uint32_t distance_from_one = UINT32_MAX - random_word;
    if (distance_from_one == 0) {
        return 32U * kGumbelMantissaEntries;
    }
    const std::uint32_t exponent = count_leading_zeros(distance_from_one);
    const std::uint32_t floor_log2 = 31U - exponent;
    std::uint32_t mantissa = 0;
    if (floor_log2 >= kGumbelMantissaBits) {
        const std::uint32_t normalized =
            distance_from_one >> (floor_log2 - kGumbelMantissaBits);
        mantissa = normalized - kGumbelMantissaEntries;
    } else {
        mantissa = distance_from_one;
    }
    return exponent * kGumbelMantissaEntries + mantissa;
}

std::int16_t factorized_gumbel_score_q10(std::uint32_t random_word) {
    const std::uint32_t distance_from_one = UINT32_MAX - random_word;
    if (distance_from_one == 0) {
        return kGumbelTopWordQ;
    }
    const std::uint32_t exponent = count_leading_zeros(distance_from_one);
    const std::uint32_t floor_log2 = 31U - exponent;
    const std::uint32_t normalized =
        floor_log2 >= kGumbelMantissaBits
            ? distance_from_one >> (floor_log2 - kGumbelMantissaBits)
            : distance_from_one << (kGumbelMantissaBits - floor_log2);
    const std::uint32_t mantissa = normalized - kGumbelMantissaEntries;
    if (exponent < kGumbelCorrectionExponents) {
        return kGumbelCorrectionQ[exponent][mantissa];
    }
    const std::int32_t score =
        static_cast<std::int32_t>(exponent + 1) * kGumbelLogTwoQ +
        kGumbelBaseQ[mantissa];
    return static_cast<std::int16_t>(score);
}

extern "C" std::uint32_t philox_word_stream_kernel(
    PhiloxWordStream& output,
    std::uint32_t position_count,
    std::uint32_t vocabulary_size,
    std::uint32_t evaluation_id,
    std::uint32_t stream_id,
    std::uint32_t seed_low,
    std::uint32_t seed_high) {
#ifdef __SYNTHESIS__
#pragma HLS INTERFACE axis port=output
#pragma HLS INTERFACE s_axilite port=position_count bundle=control
#pragma HLS INTERFACE s_axilite port=vocabulary_size bundle=control
#pragma HLS INTERFACE s_axilite port=evaluation_id bundle=control
#pragma HLS INTERFACE s_axilite port=stream_id bundle=control
#pragma HLS INTERFACE s_axilite port=seed_low bundle=control
#pragma HLS INTERFACE s_axilite port=seed_high bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
#endif

    if (position_count == 0 || position_count > 64) {
        return kRngKernelInvalidPositionCount;
    }
    if (vocabulary_size == 0 || vocabulary_size > 65'535) {
        return kRngKernelInvalidVocabularySize;
    }

    const PhiloxKey key = {seed_low, seed_high};
    for (std::uint32_t vocabulary_base = 0;
         vocabulary_base < vocabulary_size;
         vocabulary_base += 16) {
        for (std::uint32_t position = 0; position < position_count;
             ++position) {
            for (std::uint32_t lane_block = 0; lane_block < 4;
                 ++lane_block) {
#ifdef __SYNTHESIS__
#pragma HLS PIPELINE II=1
#endif
                const std::uint32_t token_base =
                    vocabulary_base + lane_block * 4;
                if (token_base >= vocabulary_size) {
                    continue;
                }
                const PhiloxCounter random = philox4x32_10(
                    {token_base / 4, position, evaluation_id, stream_id},
                    key);
                for (std::uint32_t word = 0; word < 4; ++word) {
                    if (token_base + word < vocabulary_size) {
                        output.write(random[word]);
                    }
                }
            }
        }
    }
    return kRngKernelOk;
}

extern "C" std::uint32_t philox_gumbel_stream_kernel(
    GumbelScoreStream& output,
    std::uint32_t position_count,
    std::uint32_t vocabulary_size,
    std::uint32_t evaluation_id,
    std::uint32_t stream_id,
    std::uint32_t seed_low,
    std::uint32_t seed_high) {
#ifdef __SYNTHESIS__
#pragma HLS INTERFACE axis port=output
#pragma HLS INTERFACE s_axilite port=position_count bundle=control
#pragma HLS INTERFACE s_axilite port=vocabulary_size bundle=control
#pragma HLS INTERFACE s_axilite port=evaluation_id bundle=control
#pragma HLS INTERFACE s_axilite port=stream_id bundle=control
#pragma HLS INTERFACE s_axilite port=seed_low bundle=control
#pragma HLS INTERFACE s_axilite port=seed_high bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
#endif

    if (position_count == 0 || position_count > 64) {
        return kRngKernelInvalidPositionCount;
    }
    if (vocabulary_size == 0 || vocabulary_size > 65'535) {
        return kRngKernelInvalidVocabularySize;
    }

    const PhiloxKey key = {seed_low, seed_high};
    for (std::uint32_t vocabulary_base = 0;
         vocabulary_base < vocabulary_size;
         vocabulary_base += 16) {
        for (std::uint32_t position = 0; position < position_count;
             ++position) {
            for (std::uint32_t lane_block = 0; lane_block < 4;
                 ++lane_block) {
#ifdef __SYNTHESIS__
#pragma HLS PIPELINE II=1
#endif
                const std::uint32_t token_base =
                    vocabulary_base + lane_block * 4;
                if (token_base >= vocabulary_size) {
                    continue;
                }
                const PhiloxCounter random = philox4x32_10(
                    {token_base / 4, position, evaluation_id, stream_id},
                    key);
                for (std::uint32_t word = 0; word < 4; ++word) {
                    if (token_base + word < vocabulary_size) {
                        output.write(factorized_gumbel_score_q10(random[word]));
                    }
                }
            }
        }
    }
    return kRngKernelOk;
}

}  // namespace diffusion_accel
