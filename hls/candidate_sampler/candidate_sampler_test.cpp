#include "candidate_sampler.hpp"

#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>

namespace {

using diffusion_accel::candidate_result_changed_count;
using diffusion_accel::candidate_result_status;
using diffusion_accel::candidate_reveal_kernel;
using diffusion_accel::kCandidateBitmapWords;
using diffusion_accel::kCandidateKernelInvalidPositionCount;
using diffusion_accel::kCandidateKernelInvalidThreshold;
using diffusion_accel::kCandidateKernelOk;
using diffusion_accel::kCandidateMaxPositions;
using diffusion_accel::kRevealProbabilityOneQ32;

struct Buffers {
    std::array<std::uint16_t, kCandidateMaxPositions> tokens{};
    std::array<std::uint16_t, kCandidateMaxPositions> candidates{};
    std::array<std::uint64_t, kCandidateBitmapWords> active{};
    std::array<std::uint64_t, kCandidateBitmapWords> valid{};
    std::array<std::uint32_t, kCandidateMaxPositions> random{};
};

Buffers initialized(std::uint32_t positions) {
    Buffers buffers;
    for (std::uint32_t position = 0; position < positions; ++position) {
        buffers.tokens[position] = 50'257;
        buffers.candidates[position] = static_cast<std::uint16_t>(100 + position);
        buffers.active[position >> 6] |= std::uint64_t{1} << (position & 63);
        buffers.valid[position >> 6] |= std::uint64_t{1} << (position & 63);
    }
    return buffers;
}

std::uint64_t run(Buffers& buffers,
                  std::uint64_t threshold,
                  std::uint32_t positions) {
    return candidate_reveal_kernel(
        buffers.tokens.data(),
        buffers.candidates.data(),
        buffers.active.data(),
        buffers.valid.data(),
        buffers.random.data(),
        threshold,
        positions);
}

void test_zero_probability_preserves_candidates() {
    Buffers buffers = initialized(64);
    const std::uint64_t result = run(buffers, 0, 64);
    assert(candidate_result_status(result) == kCandidateKernelOk);
    assert(candidate_result_changed_count(result) == 0);
    assert(buffers.active[0] == ~std::uint64_t{0});
    assert(buffers.valid[0] == ~std::uint64_t{0});
}

void test_probability_one_commits_every_position() {
    Buffers buffers = initialized(64);
    buffers.random.fill(UINT32_MAX);
    const std::uint64_t result =
        run(buffers, kRevealProbabilityOneQ32, 64);
    assert(candidate_result_status(result) == kCandidateKernelOk);
    assert(candidate_result_changed_count(result) == 64);
    assert(buffers.active[0] == 0);
    for (std::uint64_t word : buffers.valid) {
        assert(word == 0);
    }
    for (std::uint32_t position = 0; position < 64; ++position) {
        assert(buffers.tokens[position] == buffers.candidates[position]);
    }
}

void test_partial_commit_invalidates_all_remaining_candidates() {
    Buffers buffers = initialized(70);
    for (std::uint32_t position = 0; position < 70; ++position) {
        buffers.random[position] = position % 2 == 0 ? 0 : UINT32_MAX;
    }
    const std::uint64_t result =
        run(buffers, kRevealProbabilityOneQ32 / 2, 70);
    assert(candidate_result_status(result) == kCandidateKernelOk);
    assert(candidate_result_changed_count(result) == 35);
    for (std::uint64_t word : buffers.valid) {
        assert(word == 0);
    }
    for (std::uint32_t position = 0; position < 70; ++position) {
        const bool active =
            (buffers.active[position >> 6] >> (position & 63)) & 1;
        assert(active == (position % 2 == 1));
    }
}

void test_invalid_candidate_never_commits() {
    Buffers buffers = initialized(8);
    buffers.valid[0] &= ~(std::uint64_t{1} << 3);
    const std::uint64_t result =
        run(buffers, kRevealProbabilityOneQ32, 8);
    assert(candidate_result_changed_count(result) == 7);
    assert(buffers.tokens[3] == 50'257);
    assert(((buffers.active[0] >> 3) & 1) == 1);
}

void test_invalid_controls_are_rejected_without_mutation() {
    Buffers buffers = initialized(8);
    const std::uint64_t original_active = buffers.active[0];
    std::uint64_t result = run(buffers, 0, 0);
    assert(candidate_result_status(result) == kCandidateKernelInvalidPositionCount);
    assert(buffers.active[0] == original_active);

    result = run(buffers, kRevealProbabilityOneQ32 + 1, 8);
    assert(candidate_result_status(result) == kCandidateKernelInvalidThreshold);
    assert(buffers.active[0] == original_active);
}

}  // namespace

int main() {
    test_zero_probability_preserves_candidates();
    test_probability_one_commits_every_position();
    test_partial_commit_invalidates_all_remaining_candidates();
    test_invalid_candidate_never_commits();
    test_invalid_controls_are_rejected_without_mutation();
    std::cout << "candidate_sampler_test: all checks passed\n";
    return 0;
}
