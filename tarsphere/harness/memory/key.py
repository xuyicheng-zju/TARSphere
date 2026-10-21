"""Agent-authored key memories that outlive bounded step history.

The ledger stores only concise, episode-local knowledge the Agent explicitly
chooses to retain.  Runtime state with an existing authority (holding state,
Human identity/dialogue, object grounding, and the active plan) does not belong
here.  Each mutation is versioned and tied to the policy step that produced
it.

The model never manages identifiers: it writes ``{op, note}`` entries, and the
kernel resolves ``revise``/``forget`` targets by content matching against the
current ledger and assigns ids for ``remember`` (``m1``, ``m2``, ...).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import re
from typing import Any, Iterable


MAX_KEY_MEMORIES = 24
MAX_KEY_MEMORY_CONTENT_CHARS = 1024
#: Minimum normalized word overlap for a remember to count as a
#: restatement of an existing fact (skipped as a duplicate).
_MEMORY_DUPLICATE_THRESHOLD = 0.7
#: Minimum normalized word overlap for a revise/forget note to bind to an
#: existing memory. Content matching is deliberately simple: the model's note
#: names the fact it wants to change, not an identifier.
_MEMORY_MATCH_THRESHOLD = 0.34
_MEMORY_OPERATIONS = frozenset({"remember", "revise", "forget"})


def _normalized_words(text: str) -> frozenset[str]:
    return frozenset(re.findall(r"[a-z0-9]+", text.casefold()))


def _content_overlap(a: str, b: str) -> float:
    """Word-overlap similarity between two memory contents."""

    wa, wb = _normalized_words(a), _normalized_words(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / min(len(wa), len(wb))


@dataclass(frozen=True)
class KeyMemoryRecord:
    """One self-contained episode fact retained by the Agent.

    ``memory_id`` is a harness-assigned plain number (``"1"``, ``"2"``, ...)
    and stable for the lifetime of the fact; ids are never reused after a
    forget, so trace history stays unambiguous. ``version`` starts at one
    and increases only on a content revision; the raw step trace retains
    superseded versions.
    """

    memory_id: str
    content: str
    version: int = 1

    def describe(self) -> str:
        return f"`{self.memory_id}` v{self.version}: {self.content}"


class KeyMemoryLedger:
    """Bounded current-state store for Agent-authored key memories."""

    def __init__(self) -> None:
        self._records: dict[str, KeyMemoryRecord] = {}
        self._next_id = 1

    @property
    def records(self) -> list[KeyMemoryRecord]:
        return list(self._records.values())

    def resolve_target(self, note: str) -> KeyMemoryRecord | None:
        """Return the best content match for a revise/forget note, if any."""

        best: KeyMemoryRecord | None = None
        best_score = 0.0
        for record in self._records.values():
            score = _content_overlap(note, record.content)
            if score > best_score:
                best, best_score = record, score
        if best is not None and best_score >= _MEMORY_MATCH_THRESHOLD:
            return best
        return None

    def _assign_id(self) -> str:
        memory_id = str(self._next_id)
        self._next_id += 1
        return memory_id

    def preview(
        self,
        updates: Iterable[dict[str, Any]],
        *,
        action_step: int,
    ) -> list[KeyMemoryRecord]:
        """Validate updates atomically and return the resulting records.

        Updates carry kernel-resolved fields: ``op``, ``note``, and — for
        revise/forget — the bound ``memory_id``. ``action_step`` is retained
        in the signature because memory batches replay with their authoring
        step in episodic commits. Ids in the preview are provisional: only
        the committed ledger assigns canonical ids.
        """

        candidate = dict(self._records)
        counter = {"next": self._next_id}

        def provisional_id() -> str:
            value = counter["next"]
            counter["next"] += 1
            return str(value)

        self._apply_to(candidate, updates, id_allocator=provisional_id)
        return list(candidate.values())

    def apply(
        self,
        updates: Iterable[dict[str, Any]],
        *,
        action_step: int,
    ) -> None:
        """Commit one already validated decision's memory mutations."""

        self.apply_batches((updates,), action_step=action_step)

    def apply_batches(
        self,
        batches: Iterable[Iterable[dict[str, Any]]],
        *,
        action_step: int,
    ) -> None:
        """Atomically commit ordered model-call batches from one policy turn."""

        candidate = dict(self._records)
        for updates in batches:
            self._apply_to(
                candidate,
                updates,
                id_allocator=self._assign_id,
            )
        self._records = candidate

    @staticmethod
    def validate_updates(
        records: Iterable[KeyMemoryRecord],
        updates: Iterable[dict[str, Any]],
        *,
        action_step: int,
    ) -> tuple[list[KeyMemoryRecord], list[dict[str, Any]]]:
        """Validate against a retrieved snapshot without mutating memory.

        Returns ``(resulting_records, effective_updates)``. Restatements that
        leave the ledger unchanged, including an exact no-change revision,
        are valid but excluded from ``effective_updates``; callers must only
        credit mutations that actually change the ledger.
        """

        ledger = KeyMemoryLedger()
        ledger._records = {record.memory_id: record for record in records}
        ledger._sync_next_id()
        update_list = list(updates)
        resulting = ledger.preview(update_list, action_step=action_step)
        candidate = dict(ledger._records)
        counter = {"next": ledger._next_id}

        def provisional_id() -> str:
            value = counter["next"]
            counter["next"] += 1
            return str(value)

        effective: list[dict[str, Any]] = []
        for update in update_list:
            before = dict(candidate)
            ledger._apply_to(
                candidate,
                [update],
                id_allocator=provisional_id,
            )
            if candidate != before:
                effective.append(dict(update))
        return resulting, effective

    def _sync_next_id(self) -> None:
        used = [
            int(memory_id)
            for memory_id in self._records
            if re.fullmatch(r"\d+", memory_id)
        ]
        self._next_id = max(used, default=0) + 1

    @staticmethod
    def _apply_to(
        records: dict[str, KeyMemoryRecord],
        updates: Iterable[dict[str, Any]],
        *,
        id_allocator: Any,
    ) -> None:
        if isinstance(updates, (str, bytes, dict)):
            raise TypeError("memory updates must be a list of objects")

        seen_notes: set[str] = set()
        for index, raw in enumerate(updates):
            if not isinstance(raw, dict):
                raise TypeError(f"memory[{index}] must be an object")
            extras = set(raw) - {"op", "note", "memory_id"}
            if extras:
                raise ValueError(
                    f"memory[{index}] has unknown fields: {sorted(extras)}"
                )
            op = str(raw.get("op") or "").strip()
            if op not in _MEMORY_OPERATIONS:
                raise ValueError(
                    f"memory[{index}].op must be one of remember, revise, forget"
                )
            note = raw.get("note")
            if not isinstance(note, str) or not note.strip():
                raise ValueError(f"memory[{index}].note must be a non-empty string")
            content = " ".join(note.split())
            if len(content) > MAX_KEY_MEMORY_CONTENT_CHARS:
                raise ValueError(
                    f"memory[{index}].note exceeds "
                    f"{MAX_KEY_MEMORY_CONTENT_CHARS} characters"
                )
            note_key = content.casefold()
            if note_key in seen_notes:
                raise ValueError(
                    f"memory[{index}] repeats the same note within this update set"
                )
            seen_notes.add(note_key)

            memory_id = raw.get("memory_id")
            duplicate_id = next(
                (
                    mid
                    for mid, record in records.items()
                    if record.content.casefold() == note_key
                    or _content_overlap(record.content, content)
                    >= _MEMORY_DUPLICATE_THRESHOLD
                ),
                None,
            )
            if op == "forget":
                if not isinstance(memory_id, str) or memory_id not in records:
                    raise ValueError(
                        f"memory[{index}] forget found no matching memory "
                        f"for note: {content[:80]}"
                    )
                del records[memory_id]
                continue

            if op == "revise":
                if not isinstance(memory_id, str) or memory_id not in records:
                    raise ValueError(
                        f"memory[{index}] revise found no matching memory "
                        f"for note: {content[:80]}"
                    )
                existing = records[memory_id]
                if existing.content == content:
                    # The requested durable fact already has this exact
                    # content. Treat the revision as an observable no-op so a
                    # valid physical action is not blocked by bookkeeping.
                    continue
                records[memory_id] = replace(
                    existing,
                    content=content,
                    version=existing.version + 1,
                )
                continue

            # Exact repetition inside one submission set stays a hard
            # rejection (caught above by seen_notes); the check below only
            # fires on the first occurrence.
            if duplicate_id is not None:
                # Semantic dedup against earlier decisions: models restate
                # the same fact in slightly different words on later steps
                # (observed from every small open-weight policy). A
                # near-duplicate remember is a no-op on the existing record,
                # not a rejection — duplication control is harness
                # bookkeeping, and the model's intent was "keep this fact",
                # which the ledger already satisfies.
                continue
            new_id = id_allocator()
            records[new_id] = KeyMemoryRecord(
                memory_id=new_id,
                content=content,
            )

        if len(records) > MAX_KEY_MEMORIES:
            raise ValueError(
                f"key memory capacity is {MAX_KEY_MEMORIES}; consolidate or "
                "forget an existing memory in the same update set"
            )

    def reset(self) -> None:
        self._records.clear()
        self._next_id = 1


__all__ = [
    "KeyMemoryLedger",
    "KeyMemoryRecord",
    "MAX_KEY_MEMORIES",
    "MAX_KEY_MEMORY_CONTENT_CHARS",
]
