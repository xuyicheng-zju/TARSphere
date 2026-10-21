"""FrontierNet inference for OpenFrontier-style visual frontiers.

The network is the published RGB-D, ResNet-34 U-Net with two output heads:
one regresses the normalized distance-to-frontier field and one classifies
per-pixel information gain into eleven fixed bins. Heavy dependencies are
loaded only when :class:`FrontierNetPredictor` is constructed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np

from tarsphere.infra.navigation.open_frontier.contracts import (
    CameraIntrinsics,
    FrontierNetPrediction,
)
from tarsphere.infra.navigation.open_frontier.preprocessing import (
    PreprocessedFrontierFrame,
    preprocess_frontier_rgbd,
)


_GAIN_BIN_MAXIMA = (
    1.0,
    50.0,
    150.0,
    300.0,
    450.0,
    750.0,
    1000.0,
    1500.0,
    2000.0,
    2500.0,
    3500.0,
)


@dataclass(frozen=True, slots=True)
class FrontierNetConfig:
    checkpoint_path: str
    input_size: tuple[int, int] = (320, 320)
    num_gain_classes: int = 11
    distance_normalizer_px: float = 10.0
    frontier_distance_threshold_px: float = 0.3
    resize_margin: float = 0.02
    # OpenFrontier's released navigation configuration builds its RGB-D
    # volume with a 3.5 m sensor range. Isaac depth uses a much larger far
    # plane, so clamp no-return/far pixels to the same range before the
    # checkpoint's per-image normalization. Otherwise a handful of 100 m+
    # returns collapse nearly all useful indoor depth contrast.
    maximum_input_depth_m: float = 3.5
    device: str = "cuda:0"
    expected_checkpoint_sha256: str | None = None

    def __post_init__(self) -> None:
        if not str(self.device).startswith("cuda:"):
            raise ValueError(
                "FrontierNet inference requires an explicit CUDA device; "
                f"got {self.device!r}."
            )
        if self.num_gain_classes != 11:
            raise ValueError(
                "The released RGB-D FrontierNet checkpoint requires 11 gain classes."
            )
        if tuple(self.input_size) != (320, 320):
            raise ValueError(
                "The released RGB-D FrontierNet checkpoint requires a 320x320 crop."
            )
        if float(self.distance_normalizer_px) <= 0.0:
            raise ValueError("distance_normalizer_px must be positive.")
        if float(self.frontier_distance_threshold_px) <= 0.0:
            raise ValueError("frontier_distance_threshold_px must be positive.")
        if (
            not math.isfinite(float(self.maximum_input_depth_m))
            or self.maximum_input_depth_m <= 0.0
        ):
            raise ValueError("maximum_input_depth_m must be finite and positive.")
        if self.expected_checkpoint_sha256 is not None:
            digest = self.expected_checkpoint_sha256.strip().lower()
            if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
                raise ValueError(
                    "expected_checkpoint_sha256 must be 64 lowercase hex digits."
                )


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_two_head_unet(torch: Any, *, classes: int) -> Any:
    """Construct the released network without the legacy SMP dependency.

    OpenFrontier's checkpoint was trained with SMP 0.3.3. That package pins an
    obsolete ``timm`` release which conflicts with the repository's SAM3
    runtime. Isaac Sim also bundles its own Torchvision extension, which need
    not match the process' Torch build. The modules below retain the
    checkpoint's exact parameter names and tensor shapes while using only
    ``torch.nn``. Strict checkpoint loading verifies this claim at runtime.
    """

    nn = torch.nn
    functional = torch.nn.functional

    class Conv2dReLU(nn.Sequential):
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

    class ResNet34Encoder(nn.Module):
        """SMP-compatible six-stage ResNet-34 feature encoder."""

        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(
                4,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self._in_channels = 64
            self.layer1 = self._make_layer(64, blocks=3, stride=1)
            self.layer2 = self._make_layer(128, blocks=4, stride=2)
            self.layer3 = self._make_layer(256, blocks=6, stride=2)
            self.layer4 = self._make_layer(512, blocks=3, stride=2)

        def _make_layer(
            self,
            out_channels: int,
            *,
            blocks: int,
            stride: int,
        ) -> Any:
            downsample = None
            if stride != 1 or self._in_channels != out_channels:
                downsample = nn.Sequential(
                    nn.Conv2d(
                        self._in_channels,
                        out_channels,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                )
            layers = [
                BasicBlock(
                    self._in_channels,
                    out_channels,
                    stride=stride,
                    downsample=downsample,
                )
            ]
            self._in_channels = out_channels
            layers.extend(
                BasicBlock(out_channels, out_channels) for _ in range(1, blocks)
            )
            return nn.Sequential(*layers)

        def forward(self, tensor: Any) -> list[Any]:
            features = [tensor]
            tensor = self.conv1(tensor)
            tensor = self.bn1(tensor)
            tensor = self.relu(tensor)
            features.append(tensor)
            tensor = self.maxpool(tensor)
            tensor = self.layer1(tensor)
            features.append(tensor)
            tensor = self.layer2(tensor)
            features.append(tensor)
            tensor = self.layer3(tensor)
            features.append(tensor)
            tensor = self.layer4(tensor)
            features.append(tensor)
            return features

    class BasicBlock(nn.Module):
        """Torchvision-compatible ResNet basic block without Torchvision."""

        expansion = 1

        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            *,
            stride: int = 1,
            downsample: Any = None,
        ) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            )
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            )
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.downsample = downsample
            self.stride = stride

        def forward(self, tensor: Any) -> Any:
            identity = tensor
            tensor = self.conv1(tensor)
            tensor = self.bn1(tensor)
            tensor = self.relu(tensor)
            tensor = self.conv2(tensor)
            tensor = self.bn2(tensor)
            if self.downsample is not None:
                identity = self.downsample(identity)
            tensor += identity
            return self.relu(tensor)

    class DecoderBlock(nn.Module):
        def __init__(
            self,
            in_channels: int,
            skip_channels: int,
            out_channels: int,
        ) -> None:
            super().__init__()
            self.conv1 = Conv2dReLU(
                in_channels + skip_channels,
                out_channels,
            )
            self.attention1 = nn.Identity()
            self.conv2 = Conv2dReLU(
                out_channels,
                out_channels,
            )
            self.attention2 = nn.Identity()

        def forward(self, tensor: Any, skip: Any = None) -> Any:
            tensor = functional.interpolate(tensor, scale_factor=2, mode="nearest")
            if skip is not None:
                tensor = torch.cat([tensor, skip], dim=1)
                tensor = self.attention1(tensor)
            tensor = self.conv1(tensor)
            tensor = self.conv2(tensor)
            return self.attention2(tensor)

    class Decoder(nn.Module):
        def __init__(self, encoder_channels: list[int]) -> None:
            super().__init__()
            channels = list(encoder_channels[1:])[::-1]
            decoder_channels = [256, 128, 64, 32, 16]
            inputs = [channels[0], *decoder_channels[:-1]]
            skips = [*channels[1:], 0]
            self.center = nn.Identity()
            self.blocks = nn.ModuleList(
                [
                    DecoderBlock(in_channels, skip_channels, out_channels)
                    for in_channels, skip_channels, out_channels in zip(
                        inputs, skips, decoder_channels
                    )
                ]
            )

        def forward(self, *features: Any) -> Any:
            features = list(features[1:])[::-1]
            tensor = self.center(features[0])
            skips = features[1:]
            for index, block in enumerate(self.blocks):
                tensor = block(tensor, skips[index] if index < len(skips) else None)
            return tensor

    def output_head(out_channels: int, *, final_relu: bool) -> Any:
        layers: list[Any] = [
            nn.Conv2d(16, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, out_channels, kernel_size=1),
        ]
        layers.extend((nn.Identity(), nn.ReLU() if final_relu else nn.Identity()))
        return nn.Sequential(*layers)

    class TwoHeadFrontierNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = ResNet34Encoder()
            self.decoder = Decoder([4, 64, 64, 128, 256, 512])
            self.segmentation_head = output_head(classes, final_relu=False)
            self.df_regression_head = output_head(1, final_relu=True)

        def forward(self, tensor: Any) -> tuple[Any, Any]:
            height, width = tensor.shape[-2:]
            if height % 32 or width % 32:
                raise ValueError(
                    "FrontierNet input height and width must be divisible by 32."
                )
            features = self.encoder(tensor)
            decoded = self.decoder(*features)
            return self.df_regression_head(decoded), self.segmentation_head(decoded)

    return TwoHeadFrontierNet()


def _gain_values(num_classes: int) -> np.ndarray:
    if num_classes != len(_GAIN_BIN_MAXIMA):
        raise ValueError("FrontierNet gain bins do not match the model classes.")
    log_edges = np.log1p(np.asarray(_GAIN_BIN_MAXIMA, dtype=np.float64))
    values = np.zeros(num_classes, dtype=np.float32)
    values[1:] = np.expm1(0.5 * (log_edges[:-1] + log_edges[1:]))
    return values


class FrontierNetPredictor:
    """Load one released FrontierNet checkpoint and run deterministic inference."""

    def __init__(self, config: FrontierNetConfig) -> None:
        self.config = config
        checkpoint = Path(config.checkpoint_path).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"FrontierNet checkpoint not found: {checkpoint}")
        observed_sha = _checkpoint_sha256(checkpoint)
        expected_sha = config.expected_checkpoint_sha256
        if expected_sha is not None and observed_sha != expected_sha:
            raise ValueError(
                f"FrontierNet checkpoint SHA mismatch: {observed_sha} != {expected_sha}."
            )
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "FrontierNet requires torch; install the project open-frontier extra."
            ) from exc

        requested_device = config.device
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"FrontierNet requested {requested_device!r}, but CUDA is unavailable."
            )
        self._torch = torch
        self.device = torch.device(requested_device)
        self.checkpoint_path = checkpoint
        self.checkpoint_sha256 = observed_sha
        self.model = _build_two_head_unet(
            torch,
            classes=config.num_gain_classes,
        ).to(self.device)
        state = torch.load(
            checkpoint,
            map_location=self.device,
            weights_only=True,
        )
        if not isinstance(state, dict) or not state:
            raise ValueError(
                "FrontierNet checkpoint must be a non-empty state dictionary."
            )
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        self._gain_by_class = _gain_values(config.num_gain_classes)

    def predict(
        self,
        *,
        rgb: Any,
        depth_m: Any,
        intrinsics: CameraIntrinsics,
    ) -> tuple[PreprocessedFrontierFrame, FrontierNetPrediction]:
        """Return the aligned crop and decoded dense FrontierNet outputs."""

        depth = np.asarray(depth_m, dtype=np.float32)
        maximum_depth = float(self.config.maximum_input_depth_m)
        # Habitat-style depth presents no-return pixels at the configured far
        # plane. Reproduce that observable contract for Isaac's metric depth.
        bounded_depth = np.where(
            np.isfinite(depth) & (depth > 0.0),
            np.minimum(depth, maximum_depth),
            maximum_depth,
        ).astype(np.float32, copy=False)
        frame = preprocess_frontier_rgbd(
            rgb,
            bounded_depth,
            intrinsics,
            output_size=self.config.input_size,
            resize_margin=self.config.resize_margin,
        )
        rgb_chw = np.asarray(frame.rgb, dtype=np.float32).transpose(2, 0, 1) / 255.0
        depth_chw = frame.normalized_depth[None, :, :]
        network_input = np.concatenate([rgb_chw, depth_chw], axis=0)[None, ...]
        tensor = self._torch.from_numpy(network_input).to(
            self.device,
            dtype=self._torch.float32,
        )
        with self._torch.inference_mode():
            distance_prediction, class_logits = self.model(tensor)
        normalized = (
            distance_prediction[0, 0].detach().to("cpu").numpy().astype(np.float32)
        )
        classes = (
            class_logits.argmax(dim=1)[0].detach().to("cpu").numpy().astype(np.int16)
        )
        distance_px = (
            np.exp(-normalized.astype(np.float64))
            * float(self.config.distance_normalizer_px)
        ).astype(np.float32)
        frontier_mask = distance_px < float(self.config.frontier_distance_threshold_px)
        gain = self._gain_by_class[classes]
        information_gain = np.where(frontier_mask, gain, 0.0).astype(np.float32)
        if not np.all(np.isfinite(distance_px)):
            raise RuntimeError("FrontierNet decoded a non-finite distance field.")
        return frame, FrontierNetPrediction(
            normalized_distance_field=normalized,
            distance_field_px=distance_px,
            class_indices=classes,
            frontier_mask=frontier_mask,
            information_gain=information_gain,
        )


__all__ = ["FrontierNetConfig", "FrontierNetPredictor"]
