"""Counter-based Philox RNG and fixed-point Gumbel approximation checks."""

from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

UINT32_MASK = (1 << 32) - 1
PHILOX_M0 = 0xD2511F53
PHILOX_M1 = 0xCD9E8D57
PHILOX_W0 = 0x9E3779B9
PHILOX_W1 = 0xBB67AE85

PhiloxCounter = Tuple[int, int, int, int]
PhiloxKey = Tuple[int, int]

PHILOX4X32_10_KNOWN_ANSWERS = (
    (
        (0x00000000, 0x00000000, 0x00000000, 0x00000000),
        (0x00000000, 0x00000000),
        (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8),
    ),
    (
        (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF),
        (0xFFFFFFFF, 0xFFFFFFFF),
        (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD),
    ),
    (
        (0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344),
        (0xA4093822, 0x299F31D0),
        (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1),
    ),
)


def philox4x32_10(counter: PhiloxCounter, key: PhiloxKey) -> PhiloxCounter:
    """Return the Random123 Philox4x32 result after ten rounds."""
    if len(counter) != 4 or len(key) != 2:
        raise ValueError("Philox4x32 requires four counter and two key words")
    if any(not 0 <= word <= UINT32_MASK for word in counter + key):
        raise ValueError("Philox words must be unsigned 32-bit integers")

    c0, c1, c2, c3 = counter
    k0, k1 = key
    for round_index in range(10):
        product0 = PHILOX_M0 * c0
        product1 = PHILOX_M1 * c2
        lo0 = product0 & UINT32_MASK
        hi0 = (product0 >> 32) & UINT32_MASK
        lo1 = product1 & UINT32_MASK
        hi1 = (product1 >> 32) & UINT32_MASK
        c0, c1, c2, c3 = (
            (hi1 ^ c1 ^ k0) & UINT32_MASK,
            lo1,
            (hi0 ^ c3 ^ k1) & UINT32_MASK,
            lo0,
        )
        if round_index != 9:
            k0 = (k0 + PHILOX_W0) & UINT32_MASK
            k1 = (k1 + PHILOX_W1) & UINT32_MASK
    return c0, c1, c2, c3


def approximate_gumbel_q_from_words(
    words: object,
    *,
    mantissa_bits: int = 8,
    fraction_bits: int = 10,
) -> object:
    """Approximate Gumbel scores with a leading-zero and mantissa LUT shape."""
    if not 1 <= mantissa_bits <= 16:
        raise ValueError("mantissa_bits must be between 1 and 16")
    if not 1 <= fraction_bits <= 12:
        raise ValueError("fraction_bits must be between 1 and 12")
    import torch

    if words.dtype != torch.int64:
        raise ValueError("words must use torch.int64 to hold unsigned values")
    if bool(((words < 0) | (words > UINT32_MASK)).any().item()):
        raise ValueError("words must contain unsigned 32-bit values")

    distance_from_one = UINT32_MASK - words
    safe_distance = distance_from_one.clamp(min=1)
    floor_log2 = torch.floor(torch.log2(safe_distance.to(torch.float64))).to(
        torch.int64
    )
    shift = (floor_log2 - mantissa_bits).clamp(min=0)
    bucket_width = torch.bitwise_left_shift(torch.ones_like(shift), shift)
    representative = (
        torch.div(distance_from_one, bucket_width, rounding_mode="floor")
        * bucket_width
    ).to(torch.float64) + (bucket_width.to(torch.float64) - 1.0) / 2.0
    representative = torch.where(
        distance_from_one == 0,
        torch.zeros_like(representative),
        representative,
    )
    uniform = 1.0 - (representative + 0.5) / float(1 << 32)
    gumbel = -torch.log(-torch.log(uniform))
    quantized = torch.round(gumbel * (1 << fraction_bits))
    if bool(
        (
            (quantized < -(1 << 15))
            | (quantized > (1 << 15) - 1)
        ).any().item()
    ):
        raise RuntimeError("Gumbel Q format exceeded signed 16-bit range")
    return quantized.to(torch.int16)


def approximate_factorized_gumbel_q_from_words(
    words: object,
    *,
    mantissa_bits: int = 8,
    correction_exponents: int = 3,
    fraction_bits: int = 10,
) -> object:
    """Use a normalized mantissa LUT plus exact low-exponent corrections."""
    if not 1 <= mantissa_bits <= 12:
        raise ValueError("mantissa_bits must be between 1 and 12")
    if not 0 <= correction_exponents <= 16:
        raise ValueError("correction_exponents must be between 0 and 16")
    if not 1 <= fraction_bits <= 12:
        raise ValueError("fraction_bits must be between 1 and 12")
    import torch

    if words.dtype != torch.int64:
        raise ValueError("words must use torch.int64 to hold unsigned values")
    if bool(((words < 0) | (words > UINT32_MASK)).any().item()):
        raise ValueError("words must contain unsigned 32-bit values")

    distance_from_one = UINT32_MASK - words
    safe_distance = distance_from_one.clamp(min=1)
    floor_log2 = torch.floor(torch.log2(safe_distance.to(torch.float64))).to(
        torch.int64
    )
    exponent = 31 - floor_log2
    right_shift = (floor_log2 - mantissa_bits).clamp(min=0)
    left_shift = (mantissa_bits - floor_log2).clamp(min=0)
    normalized = torch.where(
        floor_log2 >= mantissa_bits,
        torch.bitwise_right_shift(distance_from_one, right_shift),
        torch.bitwise_left_shift(distance_from_one, left_shift),
    )
    mantissa = (normalized - (1 << mantissa_bits)).clamp(
        min=0,
        max=(1 << mantissa_bits) - 1,
    )

    scale = 1 << fraction_bits
    table_index = torch.arange(1 << mantissa_bits, dtype=torch.float64)
    normalized_midpoint = (
        (1 << mantissa_bits) + table_index + 0.5
    ) / (1 << mantissa_bits)
    base_table = torch.round(-torch.log(normalized_midpoint) * scale).to(
        torch.int64
    )
    log_two_q = round(math.log(2.0) * scale)
    quantized = (exponent + 1) * log_two_q + base_table[mantissa]

    if correction_exponents:
        exact_tables = []
        for correction_exponent in range(correction_exponents):
            delta = normalized_midpoint * 2.0 ** (-(correction_exponent + 1))
            exact_tables.append(
                torch.round(-torch.log(-torch.log1p(-delta)) * scale).to(
                    torch.int64
                )
            )
        exact_table = torch.stack(exact_tables)
        corrected_exponent = exponent.clamp(max=correction_exponents - 1)
        corrected = exact_table[corrected_exponent, mantissa]
        quantized = torch.where(
            exponent < correction_exponents,
            corrected,
            quantized,
        )

    top_uniform = 1.0 - 0.5 / float(1 << 32)
    top_q = round(-math.log(-math.log(top_uniform)) * scale)
    quantized = torch.where(
        distance_from_one == 0,
        torch.full_like(quantized, top_q),
        quantized,
    )
    if bool(
        (
            (quantized < -(1 << 15))
            | (quantized > (1 << 15) - 1)
        ).any().item()
    ):
        raise RuntimeError("factorized Gumbel Q format exceeded signed 16-bit range")
    return quantized.to(torch.int16)


def validate_rng_and_gumbel(
    *,
    distribution_trials: int = 200_000,
    distribution_vocabulary_size: int = 16,
    full_vocabulary_size: int = 50_258,
    full_vocabulary_batches: int = 16,
    full_vocabulary_batch_size: int = 64,
    mantissa_bits: Sequence[int] = (4, 6, 8),
    fraction_bits: int = 10,
    seed: int = 0,
    maximum_distribution_tv: float = 0.01,
    minimum_full_vocabulary_agreement: float = 0.995,
) -> Dict[str, object]:
    """Validate Philox vectors and sweep the fixed-point Gumbel LUT shape."""
    integer_values = {
        "distribution_trials": distribution_trials,
        "distribution_vocabulary_size": distribution_vocabulary_size,
        "full_vocabulary_size": full_vocabulary_size,
        "full_vocabulary_batches": full_vocabulary_batches,
        "full_vocabulary_batch_size": full_vocabulary_batch_size,
        "fraction_bits": fraction_bits,
    }
    if min(integer_values.values()) <= 0 or not mantissa_bits:
        raise ValueError("trial counts, dimensions, and bit widths must be positive")
    if not 0 < maximum_distribution_tv < 1:
        raise ValueError("maximum_distribution_tv must be between zero and one")
    if not 0 < minimum_full_vocabulary_agreement <= 1:
        raise ValueError(
            "minimum_full_vocabulary_agreement must be in (0, 1]"
        )

    known_answers = []
    for counter, key, expected in PHILOX4X32_10_KNOWN_ANSWERS:
        actual = philox4x32_10(counter, key)
        known_answers.append(
            {
                "counter": ["%08x" % word for word in counter],
                "key": ["%08x" % word for word in key],
                "expected": ["%08x" % word for word in expected],
                "actual": ["%08x" % word for word in actual],
                "passed": actual == expected,
            }
        )

    import torch

    generator = torch.Generator(device="cpu").manual_seed(seed)
    logits = torch.randn(
        distribution_vocabulary_size,
        generator=generator,
        dtype=torch.float64,
    )
    target_distribution = torch.softmax(logits, dim=-1)
    distribution_words = torch.randint(
        0,
        1 << 32,
        (distribution_trials, distribution_vocabulary_size),
        generator=generator,
        dtype=torch.int64,
    )
    distribution_noise = approximate_factorized_gumbel_q_from_words(
        distribution_words,
        mantissa_bits=max(mantissa_bits),
        correction_exponents=3,
        fraction_bits=fraction_bits,
    ).to(torch.int64)
    logits_q = torch.round(logits * (1 << fraction_bits)).to(torch.int64)
    distribution_candidates = (logits_q + distribution_noise).argmax(dim=-1)
    empirical = torch.bincount(
        distribution_candidates,
        minlength=distribution_vocabulary_size,
    ).to(torch.float64) / distribution_trials
    distribution_tv = float(
        0.5 * (empirical - target_distribution).abs().sum().item()
    )

    agreements = {bits: 0 for bits in mantissa_bits}
    factorized_matches = 0
    total_full_vocabulary_samples = (
        full_vocabulary_batches * full_vocabulary_batch_size
    )
    for _ in range(full_vocabulary_batches):
        batch_logits = torch.randn(
            (full_vocabulary_batch_size, full_vocabulary_size),
            generator=generator,
            dtype=torch.float32,
        ).to(torch.float64)
        words = torch.randint(
            0,
            1 << 32,
            (full_vocabulary_batch_size, full_vocabulary_size),
            generator=generator,
            dtype=torch.int64,
        )
        uniforms = (words.to(torch.float64) + 0.5) / float(1 << 32)
        exact_noise = -torch.log(-torch.log(uniforms))
        exact_candidates = (batch_logits + exact_noise).argmax(dim=-1)
        batch_logits_q = torch.round(
            batch_logits * (1 << fraction_bits)
        ).to(torch.int64)
        for bits in mantissa_bits:
            approximate_noise = approximate_gumbel_q_from_words(
                words,
                mantissa_bits=bits,
                fraction_bits=fraction_bits,
            ).to(torch.int64)
            approximate_candidates = (
                batch_logits_q + approximate_noise
            ).argmax(dim=-1)
            agreements[bits] += int(
                approximate_candidates.eq(exact_candidates).sum().item()
            )
        factorized_noise = approximate_factorized_gumbel_q_from_words(
            words,
            mantissa_bits=max(mantissa_bits),
            correction_exponents=3,
            fraction_bits=fraction_bits,
        ).to(torch.int64)
        factorized_candidates = (batch_logits_q + factorized_noise).argmax(
            dim=-1
        )
        factorized_matches += int(
            factorized_candidates.eq(exact_candidates).sum().item()
        )

    sweep = []
    for bits in mantissa_bits:
        agreement = agreements[bits] / total_full_vocabulary_samples
        sweep.append(
            {
                "mantissa_bits": bits,
                "conservative_lut_entries": 33 * (1 << bits),
                "int16_lut_bytes": 33 * (1 << bits) * 2,
                "matching_candidates": agreements[bits],
                "samples": total_full_vocabulary_samples,
                "pathwise_agreement": agreement,
                "passed": agreement >= minimum_full_vocabulary_agreement,
            }
        )
    selected = next(item for item in sweep if item["mantissa_bits"] == max(mantissa_bits))
    factorized_agreement = factorized_matches / total_full_vocabulary_samples
    factorized_lut_bytes = (1 + 3) * (1 << max(mantissa_bits)) * 2
    factorized = {
        "mantissa_bits": max(mantissa_bits),
        "correction_exponents": 3,
        "int16_lut_bytes": factorized_lut_bytes,
        "matching_candidates": factorized_matches,
        "samples": total_full_vocabulary_samples,
        "pathwise_agreement": factorized_agreement,
        "passed": factorized_agreement >= minimum_full_vocabulary_agreement,
    }
    passed = (
        all(item["passed"] for item in known_answers)
        and distribution_tv <= maximum_distribution_tv
        and bool(factorized["passed"])
    )
    return {
        "philox4x32_10": {
            "known_answer_source": (
                "Random123 tests/kat_vectors, Philox4x32 10-round entries"
            ),
            "known_answer_source_url": (
                "https://github.com/DEShawResearch/random123/blob/"
                "9545ff6413f258be2f04c1d319d99aaef7521150/tests/kat_vectors#L27-L29"
            ),
            "random123_revision": (
                "9545ff6413f258be2f04c1d319d99aaef7521150"
            ),
            "known_answers": known_answers,
            "passed": all(item["passed"] for item in known_answers),
        },
        "distribution_validation": {
            "trials": distribution_trials,
            "vocabulary_size": distribution_vocabulary_size,
            "mantissa_bits": max(mantissa_bits),
            "fraction_bits": fraction_bits,
            "total_variation_distance": distribution_tv,
            "maximum_total_variation": maximum_distribution_tv,
            "passed": distribution_tv <= maximum_distribution_tv,
        },
        "full_vocabulary_stress": {
            "vocabulary_size": full_vocabulary_size,
            "batches": full_vocabulary_batches,
            "batch_size": full_vocabulary_batch_size,
            "samples": total_full_vocabulary_samples,
            "fraction_bits": fraction_bits,
            "minimum_pathwise_agreement": minimum_full_vocabulary_agreement,
            "sweep": sweep,
            "full_table_reference": selected,
            "selected_design": "factorized-mantissa-plus-three-correction-tables",
            "factorized_design": factorized,
            "selected_passed": bool(factorized["passed"]),
        },
        "scope": (
            "philox-known-answer-and-software-gumbel-lut-shape-validation; "
            "not-synthesized-and-not-a-complete-rng-statistical-suite"
        ),
        "passed": passed,
    }
