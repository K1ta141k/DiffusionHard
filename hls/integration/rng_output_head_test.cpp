#include "../output_head/output_head.hpp"
#include "../rng/philox_rng.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

namespace {

std::int32_t requantize(std::int32_t accumulator, std::int32_t multiplier) {
    const std::int64_t product = std::int64_t{accumulator} * multiplier;
    constexpr std::int64_t half = std::int64_t{1} << 19;
    return product >= 0
               ? static_cast<std::int32_t>((product + half) >> 20)
               : -static_cast<std::int32_t>((-product + half) >> 20);
}

}  // namespace

int main() {
    using namespace diffusion_accel;
    constexpr std::uint32_t positions = 5;
    constexpr std::uint32_t hidden_size = 7;
    constexpr std::uint32_t vocabulary = 19;
    constexpr std::uint32_t mask_token = 18;

    std::vector<std::int8_t> hidden(positions * hidden_size);
    std::vector<std::int8_t> weights(vocabulary * hidden_size);
    std::vector<std::int32_t> multipliers(vocabulary);
    std::vector<std::int32_t> bias(vocabulary);
    for (std::size_t index = 0; index < hidden.size(); ++index) {
        hidden[index] = static_cast<std::int8_t>((index * 5 + 1) % 19 - 9);
    }
    for (std::size_t index = 0; index < weights.size(); ++index) {
        weights[index] = static_cast<std::int8_t>((index * 7 + 2) % 23 - 11);
    }
    for (std::uint32_t token = 0; token < vocabulary; ++token) {
        multipliers[token] = (1 << 19) + static_cast<std::int32_t>(token * 19001);
        bias[token] = static_cast<std::int32_t>(token * 17) - 90;
    }

    GumbelScoreStream generated_scores;
    assert(philox_gumbel_stream_kernel(
               generated_scores,
               positions,
               vocabulary,
               3,
               5,
               7,
               11) == kRngKernelOk);
    OutputHeadNoiseStream output_head_scores;
    std::vector<std::int32_t> noise(positions * vocabulary);
    for (std::uint32_t base = 0; base < vocabulary;
         base += kOutputHeadVocabularyTile) {
        for (std::uint32_t position = 0; position < positions; ++position) {
            for (std::uint32_t lane = 0; lane < kOutputHeadVocabularyTile;
                 ++lane) {
                const std::uint32_t token = base + lane;
                if (token < vocabulary) {
                    const std::int32_t score = generated_scores.read();
                    output_head_scores.write(score);
                    noise[position * vocabulary + token] = score;
                }
            }
        }
    }
    assert(generated_scores.remaining() == 0);

    std::array<std::uint16_t, kOutputHeadMaxPositions> candidates{};
    assert(output_head_int8_kernel(
               hidden.data(),
               weights.data(),
               multipliers.data(),
               bias.data(),
               output_head_scores,
               candidates.data(),
               positions,
               hidden_size,
               vocabulary,
               mask_token) == kOutputHeadKernelOk);
    assert(output_head_scores.remaining() == 0);

    std::array<std::uint16_t, kOutputHeadMaxPositions> expected{};
    for (std::uint32_t position = 0; position < positions; ++position) {
        std::int64_t best = std::numeric_limits<std::int64_t>::min();
        for (std::uint32_t token = 0; token < vocabulary; ++token) {
            if (token == mask_token) {
                continue;
            }
            std::int32_t accumulator = 0;
            for (std::uint32_t index = 0; index < hidden_size; ++index) {
                accumulator +=
                    std::int32_t{hidden[position * hidden_size + index]} *
                    std::int32_t{weights[token * hidden_size + index]};
            }
            const std::int64_t score =
                std::int64_t{requantize(accumulator, multipliers[token])} +
                bias[token] + noise[position * vocabulary + token];
            if (score > best) {
                best = score;
                expected[position] = static_cast<std::uint16_t>(token);
            }
        }
    }
    assert(std::equal(
        expected.begin(), expected.begin() + positions, candidates.begin()));
    std::cout << "rng_output_head_test: all checks passed\n";
    return 0;
}
