"""The vendored SAM loader rejects missing weights on every consumed branch."""

import ast
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest


def loader(checkpoint=None):
    # Extract the pure validation boundary without importing the GPU model stack.
    source = (
        Path(__file__).resolve().parents[1] / "third_party/sam3/sam3/model_builder.py"
    )
    tree = ast.parse(source.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_load_checkpoint"
    )
    scope = {
        "g_pathmgr": SimpleNamespace(open=lambda *args: BytesIO()),
        "torch": SimpleNamespace(
            load=lambda *args, **kwargs: checkpoint or {"detector.head.weight": 1}
        ),
    }
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"),
        scope,
    )
    return scope["_load_checkpoint"]


def model(missing=(), unexpected=(), scalp=1):
    return SimpleNamespace(
        backbone=SimpleNamespace(scalp=scalp),
        inst_interactive_predictor=None,
        load_state_dict=lambda weights, strict: (list(missing), list(unexpected)),
    )


def test_sam31_allows_only_the_permanently_discarded_fourth_level():
    missing = {
        f"backbone.vision_backbone.convs.3.{layer}.{parameter}"
        for layer in ("conv_1x1", "conv_3x3")
        for parameter in ("weight", "bias")
    }
    loader()(model(missing), "sam3.1_multiplex.pt")
    with pytest.raises(RuntimeError, match="checkpoint mismatch"):
        loader()(model(missing, scalp=0), "sam3.1_multiplex.pt")


@pytest.mark.parametrize(
    "missing,unexpected",
    [
        (["backbone.vision_backbone.convs.2.conv_1x1.weight"], []),
        (["segmentation_head.weight"], []),
        ([], ["unknown_detector.weight"]),
    ],
)
def test_sam3_rejects_every_required_mismatch(missing, unexpected):
    with pytest.raises(RuntimeError, match="checkpoint mismatch"):
        loader()(model(missing, unexpected), "wrong.pt")


def test_sam31_excludes_only_declared_video_neck_parameters():
    image = model()
    image.load_state_dict = lambda weights, strict: (
        [],
        sorted(set(weights) - {"head.weight"}),
    )
    checkpoint = {
        "detector.head.weight": 1,
        "detector.backbone.vision_backbone.interactive_convs.0.conv_1x1.weight": 2,
        "detector.backbone.vision_backbone.propagation_convs.2.conv_3x3.bias": 3,
    }
    loader(checkpoint)(image, "sam3.1_multiplex.pt")
    checkpoint[
        "detector.backbone.vision_backbone.interactive_convs.0.unknown.weight"
    ] = 4
    with pytest.raises(RuntimeError, match="unexpected"):
        loader(checkpoint)(image, "wrong.pt")
