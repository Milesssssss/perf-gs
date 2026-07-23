"""Executable architecture boundaries introduced by the refactor."""

from __future__ import annotations

import ast
from pathlib import Path

from trainer import buffers
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
