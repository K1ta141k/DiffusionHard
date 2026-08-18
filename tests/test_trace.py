from pathlib import Path

from diffusion_accel.trace import read_trace, synthetic_trace, write_trace


def test_synthetic_trace_round_trip(tmp_path: Path) -> None:
    trace = synthetic_trace(steps=3, layers=2, canvas_tokens=16)
    path = tmp_path / "trace.jsonl"
    write_trace(trace, path)
    restored = read_trace(path)

    assert restored == trace
    assert len(restored.steps) == 3
    assert restored.steps[-1].active_tokens < restored.steps[0].active_tokens


def test_synthetic_trace_rejects_invalid_dimensions() -> None:
    try:
        synthetic_trace(steps=0)
    except ValueError as error:
        assert "steps" in str(error)
    else:
        raise AssertionError("expected invalid dimensions to fail")
