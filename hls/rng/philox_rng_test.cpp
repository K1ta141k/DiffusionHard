#include "philox_rng.hpp"

#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>

namespace {

using diffusion_accel::PhiloxCounter;
using diffusion_accel::GumbelScoreStream;
using diffusion_accel::PhiloxKey;
using diffusion_accel::PhiloxWordStream;
using diffusion_accel::gumbel_lut_address;
using diffusion_accel::factorized_gumbel_score_q10;
using diffusion_accel::kGumbelLutEntries;
using diffusion_accel::kRngKernelInvalidPositionCount;
using diffusion_accel::kRngKernelInvalidVocabularySize;
using diffusion_accel::kRngKernelOk;
using diffusion_accel::philox4x32_10;
using diffusion_accel::philox_word_stream_kernel;
using diffusion_accel::philox_gumbel_stream_kernel;

void test_random123_known_answers() {
    // Random123 tests/kat_vectors, Philox4x32 10-round entries.
    assert((philox4x32_10({0, 0, 0, 0}, {0, 0}) ==
            PhiloxCounter{0x6627E8D5, 0xE169C58D, 0xBC57AC4C,
                          0x9B00DBD8}));
    assert((philox4x32_10(
                {UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX},
                {UINT32_MAX, UINT32_MAX}) ==
            PhiloxCounter{0x408F276D, 0x41C83B0E, 0xA20BC7C6,
                          0x6D5451FD}));
    assert((philox4x32_10(
                {0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344},
                {0xA4093822, 0x299F31D0}) ==
            PhiloxCounter{0xD16CFE09, 0x94FDCCEB, 0x5001E420,
                          0x24126EA1}));
}

void test_stream_order_matches_counter_mapping() {
    constexpr std::uint32_t positions = 3;
    constexpr std::uint32_t vocabulary = 19;
    constexpr std::uint32_t evaluation = 7;
    constexpr std::uint32_t stream_id = 11;
    const PhiloxKey key = {13, 17};
    PhiloxWordStream stream;
    const std::uint32_t status = philox_word_stream_kernel(
        stream, positions, vocabulary, evaluation, stream_id, key[0], key[1]);
    assert(status == kRngKernelOk);

    for (std::uint32_t base = 0; base < vocabulary; base += 16) {
        for (std::uint32_t position = 0; position < positions; ++position) {
            for (std::uint32_t lane_block = 0; lane_block < 4; ++lane_block) {
                const std::uint32_t token_base = base + lane_block * 4;
                if (token_base >= vocabulary) {
                    continue;
                }
                const PhiloxCounter expected = philox4x32_10(
                    {token_base / 4, position, evaluation, stream_id}, key);
                for (std::uint32_t word = 0; word < 4; ++word) {
                    if (token_base + word < vocabulary) {
                        assert(stream.read() == expected[word]);
                    }
                }
            }
        }
    }
    assert(stream.remaining() == 0);
}

void test_lut_addresses_are_bounded_and_tail_sensitive() {
    const std::uint32_t top = gumbel_lut_address(UINT32_MAX);
    const std::uint32_t next = gumbel_lut_address(UINT32_MAX - 1);
    const std::uint32_t lower = gumbel_lut_address(UINT32_MAX - 2);
    assert(top < kGumbelLutEntries);
    assert(next < kGumbelLutEntries);
    assert(lower < kGumbelLutEntries);
    assert(top != next);
    assert(next != lower);
    assert(gumbel_lut_address(0) < kGumbelLutEntries);
}

void test_factorized_gumbel_scores_match_python_reference() {
    assert(factorized_gumbel_score_q10(0) == -1983);
    assert(factorized_gumbel_score_q10(1) == -1983);
    assert(factorized_gumbel_score_q10(0x7FFFFFFF) == 372);
    assert(factorized_gumbel_score_q10(0x80000000) == 377);
    assert(factorized_gumbel_score_q10(0xFFFFFF00) == 17043);
    assert(factorized_gumbel_score_q10(0xFFFFFFFE) == 22718);
    assert(factorized_gumbel_score_q10(0xFFFFFFFF) == 23423);
}

void test_gumbel_stream_matches_word_transform() {
    constexpr std::uint32_t positions = 2;
    constexpr std::uint32_t vocabulary = 19;
    PhiloxWordStream words;
    GumbelScoreStream scores;
    assert(philox_word_stream_kernel(words, positions, vocabulary, 3, 5, 7, 11) ==
           kRngKernelOk);
    assert(philox_gumbel_stream_kernel(scores, positions, vocabulary, 3, 5, 7,
                                       11) == kRngKernelOk);
    for (std::uint32_t index = 0; index < positions * vocabulary; ++index) {
        assert(scores.read() == factorized_gumbel_score_q10(words.read()));
    }
    assert(words.remaining() == 0);
    assert(scores.remaining() == 0);
}

void test_invalid_dimensions_are_rejected() {
    PhiloxWordStream stream;
    assert(philox_word_stream_kernel(stream, 0, 1, 0, 0, 0, 0) ==
           kRngKernelInvalidPositionCount);
    assert(philox_word_stream_kernel(stream, 1, 0, 0, 0, 0, 0) ==
           kRngKernelInvalidVocabularySize);
}

}  // namespace

int main() {
    test_random123_known_answers();
    test_stream_order_matches_counter_mapping();
    test_lut_addresses_are_bounded_and_tail_sensitive();
    test_factorized_gumbel_scores_match_python_reference();
    test_gumbel_stream_matches_word_transform();
    test_invalid_dimensions_are_rejected();
    std::cout << "philox_rng_test: all checks passed\n";
    return 0;
}
