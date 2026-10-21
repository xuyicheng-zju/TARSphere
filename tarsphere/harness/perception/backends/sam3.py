"""SAM3-backed semantic perception adapter.

SAM3 is optional and heavy. Imports and GPU loading happen at explicit readiness
before an evaluated rollout; individual images are segmented only on request.
Selecting this backend without installing SAM3
fails loudly with installation guidance instead of silently returning empty
detections.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import sys
import types
from typing import Any

import numpy as np

from tarsphere.harness.perception.contracts import (
    Detection2D,
    PerceptionInput,
    PerceptionOutput,
)


SAM3_INSTALL_HINT = (
    "SAM3 backend selected but facebookresearch/sam3 is not importable. "
    "Install SAM3 from https://github.com/facebookresearch/sam3 in the active "
    "environment, ensure PyTorch/CUDA requirements are satisfied, and provide "
    "access to the SAM3 checkpoints required by build_sam3_image_model()."
)


@dataclass
class SAM3PerceptionBackend:
    """Run SAM3 image prompting for current-frame object perception."""

    model_kwargs: dict[str, Any] | None = None

    def __post_init__(self):
        self.model_kwargs = dict(self.model_kwargs or {})
        self._model = None
        self._processor = None

    @property
    def name(self) -> str:
        return "sam3"

    def reset(self) -> None:
        """Keep the loaded model; no per-episode state is required."""

    def require_ready(self) -> None:
        """Load the actual GPU model before policy calls, without segmenting an image."""
        self._ensure_processor()

    def process(self, data: PerceptionInput) -> PerceptionOutput:
        observation = data.observation
        ego = observation.views.get("ego")
        rgb = ego.rgb if ego is not None else None
        if rgb is None:
            raise ValueError("SAM3 perception requires observation.views['ego'].rgb.")

        prompts = list(data.text_prompts or [])
        if not prompts:
            # An empty result and "you never said what to look for" mean
            # opposite things to a searching agent; never conflate them.
            raise ValueError(
                "SAM3 perception requires at least one text prompt; the caller "
                "must name what to detect."
            )

        processor = self._ensure_processor()
        image = self._to_pil_image(rgb)
        with self._torch_inference_context():
            inference_state = processor.set_image(image)

        detections: list[Detection2D] = []
        trace: list[dict[str, Any]] = []
        for prompt in prompts:
            with self._torch_inference_context():
                output = processor.set_text_prompt(
                    state=inference_state,
                    prompt=str(prompt),
                )
            prompt_detections = self._detections_from_output(output, str(prompt))
            detections.extend(prompt_detections)
            trace.append(
                {
                    "backend": self.name,
                    "prompt": str(prompt),
                    "detections": len(prompt_detections),
                }
            )

        # Release the per-inference activation fragments so the next SAM3
        # prompt batch finds contiguous VRAM.
        # Without this, repeated SAM3 calls leave pytorch's caching allocator
        # holding fragmented blocks that eventually trigger an OOM on the path
        # that freezes the box. Best-effort + gated by env so it can be
        # disabled when profiling (the cuda.synchronize it implies costs ~ms).
        import os as _os

        if _os.environ.get("TARSPHERE_GPU_NO_EMPTY_CACHE", "").strip().lower() not in (
            "1",
            "true",
            "yes",
            "on",
        ):
            try:
                import torch as _torch

                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()
            except Exception:
                pass

        return PerceptionOutput(
            actor_id=observation.actor_id,
            detections=detections,
            semantic_facts=[
                f"{det.label}: score={det.score:.3f}, box={list(det.box_xyxy)}"
                for det in detections
            ],
            tool_trace=trace,
            metadata={"backend": self.name},
        )

    def _ensure_processor(self):
        if self._processor is not None:
            return self._processor
        try:
            from PIL import Image  # noqa: F401

            with self._torchvision_dynamo_import_guard():
                from sam3.model_builder import build_sam3_image_model
                from sam3.model.sam3_image_processor import Sam3Processor
        except Exception as exc:
            raise RuntimeError(SAM3_INSTALL_HINT) from exc

        # Stock SAM3 stages the whole checkpoint in host RAM before moving the
        # model to the GPU. That costs ~4.9 GB extra and surfaces as a
        # mysterious stall or an OOM kill beside Isaac rather than an error.
        # third_party/sam3 is vendored with that fix already applied, so this
        # only fires if someone restored the file from upstream.
        import inspect
        from sam3.model_builder import _load_checkpoint

        if "map_location" not in inspect.signature(_load_checkpoint).parameters:
            raise RuntimeError(
                "third_party/sam3 has been reverted to stock upstream, which "
                "loads the checkpoint through host RAM. Restore the vendored "
                "copy, or reapply third_party/patches/"
                "0001-sam3-load-checkpoint-on-target-device.patch"
            )

        self._model = build_sam3_image_model(**self.model_kwargs)
        self._processor = Sam3Processor(self._model)
        return self._processor

    @staticmethod
    def _to_pil_image(rgb: np.ndarray):
        from PIL import Image

        array = np.asarray(rgb)
        if array.ndim != 3 or array.shape[2] not in (3, 4):
            raise ValueError(
                f"Expected RGB/RGBA image with shape HxWx3/4, got {array.shape}"
            )
        if array.shape[2] == 4:
            array = array[..., :3]
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        return Image.fromarray(array)

    @staticmethod
    @contextmanager
    def _torch_inference_context():
        try:
            import torch
        except Exception:
            yield
            return

        with torch.inference_mode():
            if torch.cuda.is_available():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    yield
            else:
                yield

    @staticmethod
    @contextmanager
    def _torchvision_dynamo_import_guard():
        """Prevent torchvision RoIAlign import from pulling broken dynamo in Isaac.

        Isaac 5.1 bundles a PyTorch/torchvision pair where importing
        ``torchvision.ops.roi_align`` imports ``torch._dynamo.utils``. In the
        current environment that transitive import fails inside
        ``torch._inductor`` because of a typing_extensions compatibility issue.
        SAM3 only needs eager RoIAlign, so report compile unsupported while
        torchvision imports instead of importing full dynamo.
        """
        try:
            import torch
        except Exception:
            yield
            return

        previous_dynamo = sys.modules.get("torch._dynamo")
        previous_utils = sys.modules.get("torch._dynamo.utils")
        had_torch_attr = hasattr(torch, "_dynamo")
        previous_torch_attr = getattr(torch, "_dynamo", None)

        config = types.SimpleNamespace(
            cache_size_limit=None,
            accumulated_cache_size_limit=None,
            capture_scalar_outputs=False,
            suppress_errors=False,
            optimize_ddp=True,
        )

        def _identity_decorator(fn=None, *_args, **_kwargs):
            if fn is None:
                return lambda wrapped: wrapped
            return fn

        dynamo_stub = types.ModuleType("torch._dynamo")
        dynamo_stub.__path__ = []
        dynamo_stub.config = config
        dynamo_stub.disable = _identity_decorator
        dynamo_stub.allow_in_graph = _identity_decorator
        dynamo_stub.assume_constant_result = _identity_decorator
        dynamo_stub.is_compiling = lambda: False
        utils_stub = types.ModuleType("torch._dynamo.utils")
        utils_stub.is_compile_supported = lambda *_args, **_kwargs: False
        dynamo_stub.utils = utils_stub

        sys.modules["torch._dynamo"] = dynamo_stub
        sys.modules["torch._dynamo.utils"] = utils_stub
        setattr(torch, "_dynamo", dynamo_stub)
        try:
            yield
        finally:
            if previous_dynamo is None:
                sys.modules.pop("torch._dynamo", None)
            else:
                sys.modules["torch._dynamo"] = previous_dynamo
            if previous_utils is None:
                sys.modules.pop("torch._dynamo.utils", None)
            else:
                sys.modules["torch._dynamo.utils"] = previous_utils
            if had_torch_attr:
                setattr(torch, "_dynamo", previous_torch_attr)
            else:
                try:
                    delattr(torch, "_dynamo")
                except AttributeError:
                    pass

    @staticmethod
    def _detections_from_output(output: Any, prompt: str) -> list[Detection2D]:
        masks = SAM3PerceptionBackend._to_numpy_list(output.get("masks", []))
        boxes = SAM3PerceptionBackend._to_numpy_list(output.get("boxes", []))
        scores = SAM3PerceptionBackend._to_numpy_list(output.get("scores", []))

        detections: list[Detection2D] = []
        count = max(len(boxes), len(masks), len(scores))
        for index in range(count):
            box = boxes[index] if index < len(boxes) else None
            if box is None:
                continue
            box_values = np.asarray(box, dtype=float).reshape(-1)
            if box_values.size < 4:
                continue
            score = (
                float(np.asarray(scores[index]).reshape(-1)[0])
                if index < len(scores)
                else 0.0
            )
            mask = (
                SAM3PerceptionBackend._normalize_instance_mask(masks[index])
                if index < len(masks)
                else None
            )
            detections.append(
                Detection2D(
                    label=prompt,
                    score=score,
                    box_xyxy=(
                        float(box_values[0]),
                        float(box_values[1]),
                        float(box_values[2]),
                        float(box_values[3]),
                    ),
                    mask=mask,
                    source="sam3",
                )
            )
        return detections

    @staticmethod
    def _normalize_instance_mask(value: Any) -> np.ndarray:
        """Return one SAM3 instance mask as a spatial ``H x W`` raster.

        SAM3 releases have emitted both ``N x H x W`` and ``N x 1 x H x W``.
        ``_to_numpy_list`` removes ``N``; this boundary removes only an
        explicitly singleton channel/instance axis. A genuinely multi-plane
        mask stays malformed so synchronized depth lifting rejects it loudly.
        """
        mask = np.asarray(value)
        while mask.ndim > 2 and mask.shape[0] == 1:
            mask = mask[0]
        while mask.ndim > 2 and mask.shape[-1] == 1:
            mask = mask[..., 0]
        return mask

    @staticmethod
    def _to_numpy_list(value: Any) -> list[Any]:
        if value is None:
            return []
        value = SAM3PerceptionBackend._tensor_to_numpy(value)
        if isinstance(value, (list, tuple)):
            return [SAM3PerceptionBackend._tensor_to_numpy(item) for item in value]
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                return [value]
            return [item for item in value]
        return [value]

    @staticmethod
    def _tensor_to_numpy(value: Any) -> Any:
        if hasattr(value, "detach"):
            value = value.detach()
            if str(getattr(value, "dtype", "")) == "torch.bfloat16":
                value = value.float()
            value = value.cpu().numpy()
        elif hasattr(value, "cpu") and hasattr(value, "numpy"):
            if str(getattr(value, "dtype", "")) == "torch.bfloat16":
                value = value.float()
            value = value.cpu().numpy()
        return value
