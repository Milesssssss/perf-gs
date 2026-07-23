"""Executable architecture boundaries introduced by the refactor."""

from __future__ import annotations

import ast
from pathlib import Path

from trainer import buffers, config
from trainer.config_types import DensificationMode
from trainer.simple_densifier import SimpleDensifier


ROOT = Path(__file__).resolve().parent.parent


def test_densify_module_does_not_reach_into_pipeline_privates() -> None:
    path = ROOT / "src/trainer/densify.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    private_uses = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute) or not node.attr.startswith("_"):
            continue
        if isinstance(node.value, ast.Name) and node.value.id == "pipeline":
            private_uses.append((node.lineno, node.attr))
    assert private_uses == []


def test_simple_densifier_has_narrow_constructor() -> None:
    parameter_names = tuple(SimpleDensifier.__init__.__annotations__)
    assert "pipeline" not in parameter_names
    assert "buffers" not in parameter_names
    assert {"primitive", "adam", "resources", "eta", "scan", "kernels", "commands"} \
        <= set(parameter_names)


def test_capability_resources_are_real_dataclasses() -> None:
    # These are allocation owners, not Protocols or ad-hoc dict wrappers.
    for capability in (
        buffers.PrimitiveStorage,
        buffers.AdamResources,
        buffers.ScanResources,
        buffers.DensificationResources,
        buffers.EtaResources,
    ):
        assert hasattr(capability, "__dataclass_fields__")


def test_pipeline_configuration_is_responsibility_scoped() -> None:
    assert not hasattr(config, "RenderConfig")
    assert tuple(config.RasterConfig.__dataclass_fields__) == (
        "z_near_cull", "cluster_cull", "mip_enabled", "kernel_size")
    assert tuple(config.CompositorConfig.__dataclass_fields__) == (
        "extinction_hardness",)
    assert tuple(config.LossConfig.__dataclass_fields__) == (
        "alpha_weight", "ssim_weight", "l2_weight")
    assert tuple(config.RegularizationConfig.__dataclass_fields__) == (
        "opacity_weight", "scale_weight")


def test_fastgs_importance_path_is_absent() -> None:
    forbidden_symbols = {
        "importance_score", "importance_score_accum", "FastGS",
        "densify_score", "growth_budget", "prune_budget",
        "error_mask", "l1_map", "l1_minmax", "loss_thresh",
    }
    production = []
    for root in (ROOT / "src" / "trainer", ROOT / "shaders"):
        for path in root.rglob("*"):
            if path.suffix in {".py", ".slang"}:
                production.append(path.read_text(encoding="utf-8"))
    text = "\n".join(production)
    assert all(symbol not in text for symbol in forbidden_symbols)
    assert tuple(mode.value for mode in DensificationMode) == (
        "none", "simple", "structure_aware")
    for field in (
        "importance_score_accum", "growth_budget", "prune_budget",
        "error_mask", "l1_map", "l1_minmax", "densify_score",
    ):
        assert field not in buffers.GaussianBuffers.__dataclass_fields__
    for shader in (
        "clear_importance.slang", "compute_error_mask.slang",
        "decide_actions.slang", "score_forward.slang",
    ):
        assert not (ROOT / "shaders" / "densify" / shader).exists()
