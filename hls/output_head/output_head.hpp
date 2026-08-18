#ifndef DIFFUSION_ACCEL_OUTPUT_HEAD_HPP
#define DIFFUSION_ACCEL_OUTPUT_HEAD_HPP

#include <cstdint>

#ifdef __SYNTHESIS__
#include <hls_stream.h>
#else
#include <cstddef>
#include <vector>
#endif

namespace diffusion_accel {

constexpr std::uint32_t kOutputHeadMaxPositions = 64;
constexpr std::uint32_t kOutputHeadMaxHidden = 768;
constexpr std::uint32_t kOutputHeadMaxVocabulary = 65'535;
constexpr std::uint32_t kOutputHeadVocabularyTile = 16;

enum OutputHeadKernelStatus : std::uint32_t {
    kOutputHeadKernelOk = 0,
    kOutputHeadKernelInvalidPositionCount = 1,
    kOutputHeadKernelInvalidHiddenSize = 2,
    kOutputHeadKernelInvalidVocabularySize = 3,
    kOutputHeadKernelInvalidMaskToken = 4,
};

#ifdef __SYNTHESIS__
using OutputHeadNoiseStream = hls::stream<std::int32_t>;
#else
class OutputHeadNoiseStream {
  public:
    void write(std::int32_t value) { values_.push_back(value); }

    std::int32_t read() {
        const std::int32_t value = values_.at(read_index_);
        ++read_index_;
        return value;
    }

    std::size_t remaining() const { return values_.size() - read_index_; }

  private:
    std::vector<std::int32_t> values_;
    std::size_t read_index_ = 0;
};
#endif

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
    std::uint32_t mask_token_id);

}  // namespace diffusion_accel

#endif
