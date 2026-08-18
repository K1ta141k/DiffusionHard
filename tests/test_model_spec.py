from pathlib import Path

from diffusion_accel.model_spec import load_model_spec


ROOT = Path(__file__).resolve().parents[1]


def test_mdlm_hardware_manifest_matches_checkpoint_architecture() -> None:
    spec = load_model_spec(ROOT / "configs/models/mdlm_owt_169m.yaml")
    manifest = spec.hardware_manifest()

    assert manifest["model"]["parameter_count"] == 169_627_218
    assert sum(manifest["parameter_breakdown"].values()) == 169_627_218
    assert manifest["weight_storage_bytes"]["fp32"] == 678_508_872
    assert manifest["weight_storage_bytes"]["fp16"] == 339_254_436
    assert manifest["fp16_working_sets"]["activation_canvas"] == 98_304
    assert manifest["fp16_working_sets"]["qkv"] == 294_912
    assert manifest["fp16_working_sets"]["mlp_intermediate"] == 393_216
    assert manifest["forward_macs"]["total"] == 7_988_920_320
