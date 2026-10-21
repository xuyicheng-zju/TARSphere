"""Identity contract for one persistent Scene Server episode request."""

from __future__ import annotations

from typing import Any

from tarsphere.bench.execution_metrics import episode_content_sha256


class SceneRequestIdentityMismatch(ValueError):
    """The submitted episode content does not match its declared identity."""

    def __init__(self, identity: str, expected: str, actual: str) -> None:
        self.identity = identity
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"{identity} mismatch: caller expected {expected}, server has {actual}"
        )


def _sha256(value: Any, field: str) -> str:
    digest = str(value or "")
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return digest


def validate_scene_run_identity(
    *,
    expected_episode_sha256: Any,
    episode: Any,
) -> None:
    """Reject changed authored episode content.

    The persistent server records its own source identity in every result. A
    caller-side source hash is deliberately not a request gate: evaluation
    hotfixes are handled by restarting and re-queuing workers, not by turning
    every request to an older resident server into HTTP 409.
    """

    expected_episode = _sha256(
        expected_episode_sha256,
        "episode_sha256",
    )
    actual_episode = episode_content_sha256(episode)
    if expected_episode != actual_episode:
        raise SceneRequestIdentityMismatch(
            "episode_sha256",
            expected_episode,
            actual_episode,
        )


__all__ = [
    "SceneRequestIdentityMismatch",
    "validate_scene_run_identity",
]
