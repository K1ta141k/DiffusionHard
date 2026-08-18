#include "output_head.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

namespace {

using diffusion_accel::OutputHeadNoiseStream;
using diffusion_accel::kOutputHeadKernelInvalidHiddenSize;
using diffusion_accel::kOutputHeadKernelInvalidMaskToken;
using diffusion_accel::kOutputHeadKernelInvalidPositionCount;
using diffusion_accel::kOutputHeadKernelInvalidVocabularySize;
using diffusion_accel::kOutputHeadKernelOk;
using diffusion_accel::kOutputHeadMaxHidden;
using diffusion_accel::kOutputHeadMaxPositions;
using diffusion_accel::kOutputHeadMaxVocabulary;
using diffusion_accel::kOutputHeadVocabularyTile;
using diffusion_accel::output_head_int8_kernel;

void write_noise(OutputHeadNoiseStream& stream,
                 const std::vector<std::int32_t>& noise,
                 std::uint32_t positions,
                 std::uint32_t vocabulary) {
    for (std::uint32_t base = 0; base < vocabulary;
         base += kOutputHeadVocabularyTile) {
        for (std::uint32_t position = 0; position < positions; ++position) {
            for (std::uint32_t lane = 0; lane < kOutputHeadVocabularyTile;
                 ++lane) {
                const std::uint32_t token = base + lane;
                if (token < vocabulary) {
                    stream.write(noise[position * vocabulary + token]);
                }
            }
        }
    }
}

std::vector<std::uint16_t> reference(
    const std::vector<std::int8_t>& hidden,
    const std::vector<std::int8_t>& weights,
    const std::vector<std::int32_t>& multipliers,
    const std::vector<std::int32_t>& bias,
    const std::vector<std::int32_t>& noise,
    std::uint32_t positions,
    std::uint32_t hidden_size,
    std::uint32_t vocabulary,
    std::uint32_t mask_token) {
    std::vector<std::uint16_t> result(positions);
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
            const std::int64_t product =
                std::int64_t{accumulator} * multipliers[token];
            const std::int32_t logit_q10 =
                product >= 0
                    ? static_cast<std::int32_t>(
                          (product + (std::int64_t{1} << 19)) >> 20)
                    : -static_cast<std::int32_t>(
                          (-product + (std::int64_t{1} << 19)) >> 20);
            const std::int64_t score =
                std::int64_t{logit_q10} + bias[token] +
                noise[position * vocabulary + token];
            if (score > best) {
                best = score;
                result[position] = static_cast<std::uint16_t>(token);
            }
        }
    }
    return result;
}

void test_tiled_kernel_matches_integer_reference() {
    constexpr std::uint32_t positions = 5;
    constexpr std::uint32_t hidden_size = 7;
    constexpr std::uint32_t vocabulary = 19;
    constexpr std::uint32_t mask_token = 18;
    std::vector<std::int8_t> hidden(positions * hidden_size);
    std::vector<std::int8_t> weights(vocabulary * hidden_size);
    std::vector<std::int32_t> multipliers(vocabulary);
    std::vector<std::int32_t> bias(vocabulary);
    std::vector<std::int32_t> noise(positions * vocabulary);
    for (std::size_t index = 0; index < hidden.size(); ++index) {
        hidden[index] = static_cast<std::int8_t>((index * 7 + 3) % 17 - 8);
    }
    for (std::size_t index = 0; index < weights.size(); ++index) {
        weights[index] = static_cast<std::int8_t>((index * 11 + 5) % 23 - 11);
    }
    for (std::uint32_t token = 0; token < vocabulary; ++token) {
        multipliers[token] =
            static_cast<std::int32_t>((1U << 19) + token * 17003U);
        bias[token] = static_cast<std::int32_t>(token * 13) - 80;
    }
    for (std::size_t index = 0; index < noise.size(); ++index) {
        noise[index] = static_cast<std::int32_t>((index * 29 + 17) % 101) - 50;
    }

    OutputHeadNoiseStream stream;
    write_noise(stream, noise, positions, vocabulary);
    std::array<std::uint16_t, kOutputHeadMaxPositions> candidates{};
    const std::uint32_t status = output_head_int8_kernel(
        hidden.data(),
        weights.data(),
        multipliers.data(),
        bias.data(),
        stream,
        candidates.data(),
        positions,
        hidden_size,
        vocabulary,
        mask_token);
    const auto expected = reference(
        hidden,
        weights,
        multipliers,
        bias,
        noise,
        positions,
        hidden_size,
        vocabulary,
        mask_token);
    assert(status == kOutputHeadKernelOk);
    assert(stream.remaining() == 0);
    assert(std::equal(expected.begin(), expected.end(), candidates.begin()));
}

void test_ties_choose_the_first_unmasked_token() {
    constexpr std::uint32_t positions = 2;
    constexpr std::uint32_t hidden_size = 3;
    constexpr std::uint32_t vocabulary = 5;
    std::vector<std::int8_t> hidden(positions * hidden_size, 0);
    std::vector<std::int8_t> weights(vocabulary * hidden_size, 0);
    std::vector<std::int32_t> multipliers(vocabulary, 1 << 20);
    std::vector<std::int32_t> bias(vocabulary, 0);
    std::vector<std::int32_t> noise(positions * vocabulary, 0);
    OutputHeadNoiseStream stream;
    write_noise(stream, noise, positions, vocabulary);
    std::array<std::uint16_t, kOutputHeadMaxPositions> candidates{};
    const std::uint32_t status = output_head_int8_kernel(
        hidden.data(),
        weights.data(),
        multipliers.data(),
        bias.data(),
        stream,
        candidates.data(),
        positions,
        hidden_size,
        vocabulary,
        0);
    assert(status == kOutputHeadKernelOk);
    assert(candidates[0] == 1);
    assert(candidates[1] == 1);
}

void test_invalid_dimensions_are_rejected() {
    std::array<std::int8_t, 1> values{};
    std::array<std::int32_t, 1> bias{};
    std::array<std::int32_t, 1> multiplier{{1 << 20}};
    std::array<std::uint16_t, kOutputHeadMaxPositions> candidates{};
    OutputHeadNoiseStream stream;
    assert(output_head_int8_kernel(values.data(), values.data(), multiplier.data(), bias.data(),
                                   stream, candidates.data(), 0, 1, 1, 0) ==
           kOutputHeadKernelInvalidPositionCount);
    assert(output_head_int8_kernel(values.data(), values.data(), multiplier.data(), bias.data(),
                                   stream, candidates.data(), 1,
                                   kOutputHeadMaxHidden + 1, 1, 0) ==
           kOutputHeadKernelInvalidHiddenSize);
    assert(output_head_int8_kernel(values.data(), values.data(), multiplier.data(), bias.data(),
                                   stream, candidates.data(), 1, 1,
                                   kOutputHeadMaxVocabulary + 1, 0) ==
           kOutputHeadKernelInvalidVocabularySize);
    assert(output_head_int8_kernel(values.data(), values.data(), multiplier.data(), bias.data(),
                                   stream, candidates.data(), 1, 1, 1, 1) ==
           kOutputHeadKernelInvalidMaskToken);
}

}  // namespace

int main() {
    test_tiled_kernel_matches_integer_reference();
    test_ties_choose_the_first_unmasked_token();
    test_invalid_dimensions_are_rejected();
    std::cout << "output_head_test: all checks passed\n";
    return 0;
}
