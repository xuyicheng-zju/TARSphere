"""Single source of truth for authorable Human runtime capabilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Tuple

from pydantic import BaseModel, ConfigDict, model_validator


HUMAN_RUNTIME_CAPABILITIES_PATH = Path(__file__).with_name("runtime_capabilities.json")
HUMAN_RUNTIME_CAPABILITIES_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class IsaacPeopleRuntimeAssets:
    """Physical assets required before an Isaac People actor can run.

    Attributes:
        biped_setup_usd: Animation-graph setup consumed by Isaac People.
        biped_rig_usd: Skeleton and mesh payload referenced by the setup.
        animation_directory: Directory containing the declared motion clips.
    """

    biped_setup_usd: Path
    biped_rig_usd: Path
    animation_directory: Path


class HumanRuntimeCommand(BaseModel):
    """One command exposed to episode authors and the Human policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    description: str
    required_parameters: Tuple[str, ...]
    optional_parameters: Tuple[str, ...]
    prerequisites: Tuple[str, ...]
    backend_command: str
    animation_clips: Tuple[str, ...]
    llm_example: Mapping[str, Any]

    @model_validator(mode="after")
    def _validate_command(self) -> "HumanRuntimeCommand":
        if not self.name or not self.description:
            raise ValueError("Human runtime commands require name and description.")
        if not self.name.isidentifier() or self.name != self.name.lower():
            raise ValueError(
                f"Human runtime command must be lower snake_case: {self.name!r}"
            )
        required = set(self.required_parameters)
        optional = set(self.optional_parameters)
        if len(required) != len(self.required_parameters):
            raise ValueError(f"{self.name}: required_parameters contains duplicates.")
        if len(optional) != len(self.optional_parameters):
            raise ValueError(f"{self.name}: optional_parameters contains duplicates.")
        if required & optional:
            raise ValueError(
                f"{self.name}: required and optional parameters must be disjoint."
            )
        if len(set(self.prerequisites)) != len(self.prerequisites):
            raise ValueError(f"{self.name}: prerequisites contains duplicates.")
        if len(set(self.animation_clips)) != len(self.animation_clips):
            raise ValueError(f"{self.name}: animation_clips contains duplicates.")
        if self.llm_example.get("action") != self.name:
            raise ValueError(
                f"{self.name}: llm_example.action must equal the command name."
            )
        missing = [
            parameter
            for parameter in self.required_parameters
            if parameter not in self.llm_example
        ]
        if missing:
            raise ValueError(
                f"{self.name}: llm_example is missing required parameters {missing}."
            )
        return self


class HumanRuntimeCapabilityCatalog(BaseModel):
    """Validated command vocabulary and physical animation-clip contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str
    commands: Tuple[HumanRuntimeCommand, ...]
    required_animation_clips: Tuple[str, ...]

    @model_validator(mode="after")
    def _validate_catalog(self) -> "HumanRuntimeCapabilityCatalog":
        if self.schema_version != HUMAN_RUNTIME_CAPABILITIES_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported Human runtime capability schema "
                f"{self.schema_version!r}; expected "
                f"{HUMAN_RUNTIME_CAPABILITIES_SCHEMA_VERSION!r}."
            )
        names = self.command_names
        if len(set(names)) != len(names):
            raise ValueError("Human runtime command names must be unique.")
        if not names:
            raise ValueError("Human runtime capability catalog cannot be empty.")
        if len(set(self.required_animation_clips)) != len(
            self.required_animation_clips
        ):
            raise ValueError("required_animation_clips contains duplicates.")
        if tuple(sorted(self.required_animation_clips)) != (
            self.required_animation_clips
        ):
            raise ValueError("required_animation_clips must be sorted.")
        declared_clips = set(self.required_animation_clips)
        for command in self.commands:
            unknown = set(command.animation_clips) - declared_clips
            if unknown:
                raise ValueError(
                    f"{command.name}: animation_clips are absent from the global "
                    f"clip contract: {sorted(unknown)}"
                )
        return self

    @property
    def command_names(self) -> Tuple[str, ...]:
        return tuple(command.name for command in self.commands)

    def require_command(self, name: str) -> HumanRuntimeCommand:
        for command in self.commands:
            if command.name == name:
                return command
        raise KeyError(f"Unsupported Human runtime command: {name!r}")

    @property
    def custom_animation_commands(self) -> dict[str, str]:
        return {
            command.name: command.backend_command
            for command in self.commands
            if command.name in {"type_keyboard", "push_button"}
        }

    def prompt_lines(self) -> Tuple[str, ...]:
        return tuple(
            f"- {command.name}: {command.description} "
            f"Format: {json.dumps(dict(command.llm_example), ensure_ascii=False)}"
            for command in self.commands
        )

    def validate_action(self, action: Mapping[str, Any]) -> HumanRuntimeCommand:
        name = str(action.get("command") or "").strip()
        command = self.require_command(name)
        missing = [
            parameter
            for parameter in command.required_parameters
            if action.get(parameter) in (None, "")
        ]
        if missing:
            raise ValueError(
                f"Human command {name!r} is missing required parameters {missing}."
            )
        return command


def load_human_runtime_capabilities(
    path: Path | str = HUMAN_RUNTIME_CAPABILITIES_PATH,
) -> HumanRuntimeCapabilityCatalog:
    """Load the current Human runtime capability contract."""
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Cannot load Human runtime capability catalog: {source}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("Human runtime capability catalog root must be an object.")
    return HumanRuntimeCapabilityCatalog.model_validate(payload)


def resolve_isaac_people_runtime_assets(
    character_root: Path | str,
) -> IsaacPeopleRuntimeAssets:
    """Resolve the non-authorable assets required by Isaac People.

    ``biped_demo`` is the rig payload behind ``Biped_Setup.usd``.  It is not an
    episode Human, but deleting it leaves the visible character parent at its
    authored pose while the animation runtime reports an invalid origin pose.
    The resulting origin-synchronised collision proxy can then trip the Robot.
    """
    root = Path(character_root).expanduser().resolve()
    resolved = IsaacPeopleRuntimeAssets(
        biped_setup_usd=root / "Biped_Setup.usd",
        biped_rig_usd=root / "biped_demo" / "biped_demo_meters.usd",
        # Biped_Setup.usd owns ``../Animations/*.skelanim.usd`` payloads.
        # Preserve that authored USD relationship instead of inventing a
        # second runtime directory or repairing it with search fallbacks.
        animation_directory=root.parent / "Animations",
    )
    required_files = (
        resolved.biped_setup_usd,
        resolved.biped_rig_usd,
    )
    missing_files = [path for path in required_files if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(
            "Isaac People runtime asset is missing: "
            + ", ".join(str(path) for path in missing_files)
        )
    if not resolved.animation_directory.is_dir():
        raise FileNotFoundError(
            "Isaac People animation directory is missing: "
            f"{resolved.animation_directory}"
        )
    return resolved


HUMAN_RUNTIME_CAPABILITIES = load_human_runtime_capabilities()


__all__ = [
    "HUMAN_RUNTIME_CAPABILITIES",
    "HUMAN_RUNTIME_CAPABILITIES_PATH",
    "IsaacPeopleRuntimeAssets",
    "HumanRuntimeCapabilityCatalog",
    "HumanRuntimeCommand",
    "load_human_runtime_capabilities",
    "resolve_isaac_people_runtime_assets",
]
