import torch

from diffusion_accel.fixed_norm import fixed_layer_norm_q12


def test_fixed_layer_norm_q12_tracks_float_reference() -> None:
    generator = torch.Generator().manual_seed(17)
    activation = torch.randn(4, 768, generator=generator) * 3.0 + 0.75

    output, output_q, metrics = fixed_layer_norm_q12(activation)
    expected = torch.nn.functional.layer_norm(activation, [768])

    assert output_q.dtype == torch.int64
    assert metrics["input_saturation_fraction"] == 0.0
    assert metrics["inverse_std_bits_required"] <= 20
    torch.testing.assert_close(output, expected, rtol=0.01, atol=0.003)
