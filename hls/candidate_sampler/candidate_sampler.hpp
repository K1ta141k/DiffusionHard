#ifndef DIFFUSION_ACCEL_CANDIDATE_SAMPLER_HPP
#define DIFFUSION_ACCEL_CANDIDATE_SAMPLER_HPP

#include <cstdint>

namespace diffusion_accel {

constexpr std::uint32_t kCandidateMaxPositions = 256;
constexpr std::uint32_t kCandidateBitmapWords = kCandidateMaxPositions / 64;
constexpr std::uint64_t kRevealProbabilityOneQ32 = std::uint64_t{1} << 32;

enum CandidateKernelStatus : std::uint32_t {
    kCandidateKernelOk = 0,
    kCandidateKernelInvalidPositionCount = 1,
    kCandidateKernelInvalidThreshold = 2,
};

constexpr std::uint64_t pack_candidate_result(std::uint32_t status,
                                              std::uint32_t changed_count) {
    return (std::uint64_t{status} << 32) | changed_count;
}

constexpr std::uint32_t candidate_result_status(std::uint64_t result) {
    return static_cast<std::uint32_t>(result >> 32);
}

constexpr std::uint32_t candidate_result_changed_count(std::uint64_t result) {
    return static_cast<std::uint32_t>(result);
}

extern "C" std::uint64_t candidate_reveal_kernel(
    std::uint16_t token_ids[kCandidateMaxPositions],
    const std::uint16_t candidate_ids[kCandidateMaxPositions],
    std::uint64_t active_bitmap[kCandidateBitmapWords],
    std::uint64_t valid_bitmap[kCandidateBitmapWords],
    const std::uint32_t random_words[kCandidateMaxPositions],
    std::uint64_t reveal_threshold_q32,
    std::uint32_t position_count);

}  // namespace diffusion_accel

#endif
