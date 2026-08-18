#ifndef DIFFUSION_ACCEL_PHILOX_RNG_HPP
#define DIFFUSION_ACCEL_PHILOX_RNG_HPP

#include <array>
#include <cstdint>

#ifdef __SYNTHESIS__
#include <hls_stream.h>
#else
#include <cstddef>
#include <vector>
#endif

namespace diffusion_accel {

constexpr std::uint32_t kPhiloxM0 = 0xD2511F53U;
constexpr std::uint32_t kPhiloxM1 = 0xCD9E8D57U;
constexpr std::uint32_t kPhiloxW0 = 0x9E3779B9U;
constexpr std::uint32_t kPhiloxW1 = 0xBB67AE85U;
constexpr std::uint32_t kGumbelMantissaBits = 8;
constexpr std::uint32_t kGumbelMantissaEntries = 1U << kGumbelMantissaBits;
constexpr std::uint32_t kGumbelExponentEntries = 33;
constexpr std::uint32_t kGumbelLutEntries =
    kGumbelExponentEntries * kGumbelMantissaEntries;

using PhiloxCounter = std::array<std::uint32_t, 4>;
using PhiloxKey = std::array<std::uint32_t, 2>;

enum RngKernelStatus : std::uint32_t {
    kRngKernelOk = 0,
    kRngKernelInvalidPositionCount = 1,
    kRngKernelInvalidVocabularySize = 2,
};

#ifdef __SYNTHESIS__
using PhiloxWordStream = hls::stream<std::uint32_t>;
using GumbelScoreStream = hls::stream<std::int32_t>;
#else
class PhiloxWordStream {
  public:
    void write(std::uint32_t value) { values_.push_back(value); }

    std::uint32_t read() {
        const std::uint32_t value = values_.at(read_index_);
        ++read_index_;
        return value;
    }

    std::size_t remaining() const { return values_.size() - read_index_; }

  private:
    std::vector<std::uint32_t> values_;
    std::size_t read_index_ = 0;
};

class GumbelScoreStream {
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

PhiloxCounter philox4x32_10(PhiloxCounter counter, PhiloxKey key);

std::uint32_t gumbel_lut_address(std::uint32_t random_word);

std::int16_t factorized_gumbel_score_q10(std::uint32_t random_word);

extern "C" std::uint32_t philox_word_stream_kernel(
    PhiloxWordStream& output,
    std::uint32_t position_count,
    std::uint32_t vocabulary_size,
    std::uint32_t evaluation_id,
    std::uint32_t stream_id,
    std::uint32_t seed_low,
    std::uint32_t seed_high);

extern "C" std::uint32_t philox_gumbel_stream_kernel(
    GumbelScoreStream& output,
    std::uint32_t position_count,
    std::uint32_t vocabulary_size,
    std::uint32_t evaluation_id,
    std::uint32_t stream_id,
    std::uint32_t seed_low,
    std::uint32_t seed_high);

}  // namespace diffusion_accel

#endif
