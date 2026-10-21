"""Call live benchmark-certification operations on one Isaac Scene Server."""

from __future__ import annotations

import json
from typing import Any, Literal, Mapping, Optional, Tuple
import urllib.error
import urllib.request
import uuid

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tarsphere.bench.certification.placement import (
    PlacementSurfacePatch,
    PlacementSurfaceSearch,
)
from tarsphere.bench.certification.source_support import (
    SourceSupportEvidence,
    SourceSupportSearch,
)
from tarsphere.bench.schema import TARSphereEpisode
from tarsphere.bench.execution_metrics import episode_content_sha256


class StrictModel(BaseModel):
    """Immutable current-release scene-probe response base."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class SceneServerIdentity(StrictModel):
    """Identity and loaded scene of one live single-flight Isaac process."""

    scene_server_id: str = Field(
        min_length=1,
        description="Unique identity of the currently running Scene Server.",
    )
    scene_server_git_sha: str = Field(
        min_length=1,
        description="Repository revision loaded by that process.",
    )
    scene: str = Field(
        min_length=1,
        description="Scene asset currently loaded by Isaac.",
    )
    active_request_id: Optional[str] = Field(
        default=None,
        description="Request currently occupying the server, when any.",
    )


class SourceSupportProbeResult(StrictModel):
    """Live source-support evidence returned by the loaded Isaac scene."""

    request_id: str = Field(
        min_length=1,
        description=(
            "Single-flight Scene Server request identifier used for "
            "cancellation and log attribution."
        ),
    )
    scene_id: str = Field(
        min_length=1,
        description="Loaded grounded scene in which the probe ran.",
    )
    search: SourceSupportSearch = Field(
        description="Exact source-support search contract executed by Isaac.",
    )
    evidence: SourceSupportEvidence = Field(
        description="Live target bounds and PhysX support-ray evidence.",
    )
    scene_server_id: str = Field(
        min_length=1,
        description=(
            "Unique process identity of the baked Scene Server that executed "
            "this probe."
        ),
    )
    scene_server_git_sha: str = Field(
        min_length=1,
        description=("Repository revision loaded by the Scene Server process."),
    )


class SourceSupportProbeRejection(StrictModel):
    """Typed live geometry rejection for one target support probe."""

    request_id: str = Field(
        min_length=1,
        description="Single-flight request that produced this rejection.",
    )
    scene_id: str = Field(
        min_length=1,
        description="Loaded grounded scene in which the probe ran.",
    )
    target_object_id: str = Field(
        min_length=1,
        description="Grounded target rejected by the live support contract.",
    )
    search: SourceSupportSearch = Field(
        description="Exact source-support search executed by Isaac.",
    )
    rejection: Mapping[str, str] = Field(
        description="Typed candidate-specific rejection and direct message.",
    )
    scene_server_id: str = Field(
        min_length=1,
        description="Unique Scene Server that executed the rejected probe.",
    )
    scene_server_git_sha: str = Field(
        min_length=1,
        description="Repository revision loaded by that Scene Server.",
    )

    @model_validator(mode="after")
    def validate_rejection(self) -> "SourceSupportProbeRejection":
        if (
            not str(self.rejection.get("type") or "").strip()
            or not str(self.rejection.get("message") or "").strip()
        ):
            raise ValueError("source-support rejection requires type and message")
        return self


class PlacementSurfaceProbeResult(StrictModel):
    """Live destination patches returned by the loaded Isaac scene."""

    request_id: str = Field(
        min_length=1,
        description=(
            "Single-flight Scene Server request identifier used for "
            "cancellation and log attribution."
        ),
    )
    scene_id: str = Field(
        min_length=1,
        description="Loaded grounded scene in which the probe ran.",
    )
    target_object_id: str = Field(
        min_length=1,
        description="Grounded pickable whose footprint conditioned the probe.",
    )
    destination_object_id: str = Field(
        min_length=1,
        description="Grounded destination whose collision surface was sampled.",
    )
    search: PlacementSurfaceSearch = Field(
        description="Exact placement-surface search contract executed by Isaac.",
    )
    patches: Tuple[PlacementSurfacePatch, ...] = Field(
        description=(
            "Deterministically ordered live patches; an empty tuple rejects "
            "this target–destination pair at the surface gate."
        ),
    )
    scene_server_id: str = Field(
        min_length=1,
        description=(
            "Unique process identity of the baked Scene Server that executed "
            "this probe."
        ),
    )
    scene_server_git_sha: str = Field(
        min_length=1,
        description=("Repository revision loaded by the Scene Server process."),
    )


class SceneAbortResult(StrictModel):
    """Outcome of cancelling one exact Scene Server request."""

    aborted: bool = Field(
        description="Whether the active Scene Server request was cancelled.",
    )
    active_request_id: Optional[str] = Field(
        default=None,
        description=("Request occupying the single-flight slot when one was present."),
    )
    reason: Optional[str] = Field(
        default=None,
        description="Why no request was cancelled when aborted is false.",
    )
    simulator_active: Optional[bool] = Field(
        default=None,
        description=(
            "Whether the in-simulator rollout had started when cancellation "
            "was signalled."
        ),
    )

    @model_validator(mode="after")
    def validate_outcome(self) -> "SceneAbortResult":
        if self.aborted and self.active_request_id is None:
            raise ValueError("a successful abort must identify the active request")
        if not self.aborted and not (self.reason or "").strip():
            raise ValueError("an unsuccessful abort must provide a reason")
        return self


class SceneServerError(RuntimeError):
    """Raised when the Scene Server cannot return valid live evidence."""


class SourceSupportProbeRejected(ValueError):
    """Raised when live Isaac evidence rejects one target candidate."""

    def __init__(self, result: SourceSupportProbeRejection) -> None:
        self.result = result
        super().__init__(str(result.rejection["message"]))


class SceneServerClient:
    """Strict certification client for one loaded Isaac Scene Server.

    The client never starts Isaac, retries a failed request, switches ports,
    or fabricates evidence. The caller owns Scene Server lifecycle and decides
    whether a failed candidate should be rejected or the whole authoring run
    should stop.
    """

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
        timeout_s: Optional[float] = 120.0,
    ) -> None:
        normalized_host = str(host).strip()
        if not normalized_host:
            raise ValueError("Scene Server Client host must not be empty")
        if int(port) < 1 or int(port) > 65535:
            raise ValueError("Scene Server Client port must be within [1, 65535]")
        if timeout_s is not None and float(timeout_s) <= 0.0:
            raise ValueError("Scene Server Client timeout must be positive or None")
        self._base_url = f"http://{normalized_host}:{int(port)}"
        self._timeout_s = None if timeout_s is None else float(timeout_s)

    def identity(self) -> SceneServerIdentity:
        """Return the exact live process and loaded scene before certification."""
        response = self._get_json("health")
        if response.get("status") != "ok":
            raise SceneServerError(
                "Scene Server health response did not report status=ok"
            )
        try:
            return SceneServerIdentity.model_validate(
                {
                    "scene_server_id": response["scene_server_id"],
                    "scene_server_git_sha": response["scene_server_git_sha"],
                    "scene": response["scene"],
                    "active_request_id": response.get("active_request_id"),
                }
            )
        except (KeyError, ValueError) as exc:
            raise SceneServerError(
                f"Scene Server returned an invalid identity: {exc}"
            ) from exc

    def probe_source_support(
        self,
        *,
        scene_id: str,
        target_object_id: str,
        search: Optional[SourceSupportSearch] = None,
        request_id: Optional[str] = None,
    ) -> SourceSupportProbeResult:
        """Measure the live object initially supporting one target."""
        payload: dict[str, Any] = {
            "scene_id": _required_text(scene_id, "scene_id"),
            "target_object_id": _required_text(
                target_object_id,
                "target_object_id",
            ),
            "request_id": _request_id(request_id),
        }
        if search is not None:
            payload["search"] = search.model_dump(mode="json")
        response = self._post_json("probe_source_support", payload)
        if "rejection" in response:
            try:
                rejection = SourceSupportProbeRejection.model_validate(response)
            except ValueError as exc:
                raise SceneServerError(
                    f"Scene Server returned an invalid source-support rejection: {exc}"
                ) from exc
            if (
                rejection.request_id != payload["request_id"]
                or rejection.scene_id != payload["scene_id"]
                or rejection.target_object_id != payload["target_object_id"]
            ):
                raise SceneServerError(
                    "Source-support rejection does not match the request"
                )
            raise SourceSupportProbeRejected(rejection)
        try:
            result = SourceSupportProbeResult.model_validate(response)
        except ValueError as exc:
            raise SceneServerError(
                f"Scene Server returned an invalid source-support result: {exc}"
            ) from exc
        if (
            result.request_id != payload["request_id"]
            or result.scene_id != payload["scene_id"]
            or result.evidence.target_object_id != payload["target_object_id"]
        ):
            raise SceneServerError(
                "Source-support result does not match the requested scene and target"
            )
        return result

    def probe_placement_surfaces(
        self,
        *,
        scene_id: str,
        target_object_id: str,
        destination_object_id: str,
        search: Optional[PlacementSurfaceSearch] = None,
        request_id: Optional[str] = None,
    ) -> PlacementSurfaceProbeResult:
        """Measure live supported patches for one target–destination pair."""
        payload: dict[str, Any] = {
            "scene_id": _required_text(scene_id, "scene_id"),
            "target_object_id": _required_text(
                target_object_id,
                "target_object_id",
            ),
            "destination_object_id": _required_text(
                destination_object_id,
                "destination_object_id",
            ),
            "request_id": _request_id(request_id),
        }
        if search is not None:
            payload["search"] = search.model_dump(mode="json")
        response = self._post_json("probe_placement_surfaces", payload)
        try:
            result = PlacementSurfaceProbeResult.model_validate(response)
        except ValueError as exc:
            raise SceneServerError(
                f"Scene Server returned an invalid placement-surface result: {exc}"
            ) from exc
        if (
            result.request_id != payload["request_id"]
            or result.scene_id != payload["scene_id"]
            or result.target_object_id != payload["target_object_id"]
            or result.destination_object_id != payload["destination_object_id"]
            or any(
                patch.target_object_id != payload["target_object_id"]
                or patch.destination_object_id != payload["destination_object_id"]
                for patch in result.patches
            )
        ):
            raise SceneServerError(
                "Placement-surface result does not match the requested "
                "scene, target, and destination"
            )
        return result

    def run_certification_episode(
        self,
        episode: TARSphereEpisode,
        *,
        certification_stage: Literal["A", "B", "C"],
        request_id: Optional[str] = None,
        record_video: bool = False,
        max_steps: Optional[int] = None,
    ) -> Mapping[str, Any]:
        """Send one Golden certification episode through the rollout API."""
        if certification_stage not in {"A", "B", "C"}:
            raise ValueError("certification_stage must be A, B, or C")
        if max_steps is not None and int(max_steps) <= 0:
            raise ValueError("episode max_steps must be positive when specified")

        resolved_request_id = _request_id(request_id)
        payload: dict[str, Any] = {
            "episode": episode.model_dump(
                mode="json",
                exclude_none=True,
            ),
            "request_id": resolved_request_id,
            "certification_stage": certification_stage,
            "record_video": bool(record_video),
            "episode_sha256": episode_content_sha256(episode),
        }
        if max_steps is not None:
            payload["max_steps"] = int(max_steps)
        response = self._post_json(
            "run_episode",
            payload,
            timeout_s=self._timeout_s,
        )
        if response.get("request_id") != resolved_request_id:
            raise SceneServerError(
                "Scene Server episode result has a different request id"
            )
        return response

    def abort(self, request_id: str) -> SceneAbortResult:
        """Request cancellation of the exact active Scene Server operation."""
        expected_request_id = _required_text(request_id, "request_id")
        response = self._post_json(
            "abort",
            {"request_id": expected_request_id},
            timeout_s=10.0,
        )
        try:
            result = SceneAbortResult.model_validate(response)
        except ValueError as exc:
            raise SceneServerError(
                f"Scene Server returned an invalid abort result: {exc}"
            ) from exc
        if result.aborted and result.active_request_id != expected_request_id:
            raise SceneServerError("Scene Server aborted a different active request")
        return result

    def _post_json(
        self,
        route: str,
        payload: Mapping[str, Any],
        *,
        timeout_s: Optional[float] = None,
    ) -> Mapping[str, Any]:
        request = urllib.request.Request(
            f"{self._base_url}/{route}",
            data=json.dumps(
                dict(payload),
                ensure_ascii=False,
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=(self._timeout_s if timeout_s is None else timeout_s),
            ) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise SceneServerError(
                f"Scene Server {route!r} failed with HTTP {exc.code}: {body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise SceneServerError(
                f"Scene Server {route!r} is unavailable: {exc.reason}"
            ) from exc
        try:
            decoded = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SceneServerError(
                f"Scene Server {route!r} returned invalid JSON"
            ) from exc
        if not isinstance(decoded, dict):
            raise SceneServerError(
                f"Scene Server {route!r} response must be a JSON object"
            )
        return decoded

    def _get_json(self, route: str) -> Mapping[str, Any]:
        request = urllib.request.Request(
            f"{self._base_url}/{route}",
            method="GET",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self._timeout_s,
            ) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise SceneServerError(
                f"Scene Server {route!r} failed with HTTP {exc.code}: {body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise SceneServerError(
                f"Scene Server {route!r} is unavailable: {exc.reason}"
            ) from exc
        try:
            decoded = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SceneServerError(
                f"Scene Server {route!r} returned invalid JSON"
            ) from exc
        if not isinstance(decoded, dict):
            raise SceneServerError(
                f"Scene Server {route!r} response must be a JSON object"
            )
        return decoded


def _required_text(value: str, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    return normalized


def _request_id(value: Optional[str]) -> str:
    return str(uuid.uuid4()) if value is None else _required_text(value, "request_id")


__all__ = [
    "PlacementSurfaceProbeResult",
    "SceneAbortResult",
    "SceneServerClient",
    "SceneServerError",
    "SceneServerIdentity",
    "SourceSupportProbeResult",
    "SourceSupportProbeRejected",
    "SourceSupportProbeRejection",
]
