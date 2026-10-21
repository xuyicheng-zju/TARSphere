"""Plan local staging for the simulator, source and asset trees.

Kit opens many USD and MDL dependencies during scene loading. Network
filesystems can add substantial metadata latency to those small reads.
This module classifies filesystems and resolves staging destinations;
``scripts/stage.sh`` copies the trees, and ``scripts/activate.sh`` loads
the resulting runtime paths."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Filesystems where per-open metadata latency makes Kit's material phase
# pathological. fuse covers sshfs/goofys style mounts; overlay is deliberately
# absent because a container overlay sits on local disk.
NETWORK_FSTYPES = frozenset(
    {
        "nfs",
        "nfs4",
        "cifs",
        "smb3",
        "fuse",
        "fuseblk",
        "fuse.sshfs",
        "lustre",
        "gpfs",
        "ceph",
        "glusterfs",
        "beegfs",
        "afs",
        "9p",
        "virtiofs",
    }
)

_MOUNTS = Path("/proc/mounts")

# A depth- and fanout-bounded scan: the point is to catch asset trees that are
# themselves symlinks onto a remote mount, without paying a full walk of the
# 103k files underneath. GRScenes-100 relocates subtrees at two depths relative
# to the asset root -- home_scenes/{models,Materials} at 4 and the per-scene
# home_scenes/scenes/<scene>/{models,Materials} at 6 -- so the default reaches
# both. Symlinks are never descended into, which is what keeps the cost bounded;
# the entry cap covers the case where a real directory has a huge fanout.
_PROBE_MAX_DEPTH = 7
_PROBE_MAX_ENTRIES = 2048


def _unescape_mount_field(field: str) -> str:
    """Decode the octal escapes ``/proc/mounts`` uses for space, tab and newline."""
    for escape, literal in (
        ("\\040", " "),
        ("\\011", "\t"),
        ("\\012", "\n"),
        ("\\134", "\\"),
    ):
        field = field.replace(escape, literal)
    return field


def read_mount_table(mounts: Path = _MOUNTS) -> list[tuple[str, str]]:
    """Return ``(mountpoint, fstype)`` pairs, longest mountpoint first.

    Sorting by descending mountpoint length lets the first prefix match win, so
    a bind mount nested inside another mount resolves to its own filesystem.
    """
    try:
        raw = mounts.read_text()
    except OSError:
        return []

    table: list[tuple[str, str]] = []
    for line in raw.splitlines():
        fields = line.split()
        if len(fields) < 3:
            continue
        table.append((_unescape_mount_field(fields[1]), fields[2]))
    table.sort(key=lambda item: len(item[0]), reverse=True)
    return table


def filesystem_type(
    path: str | os.PathLike[str],
    table: list[tuple[str, str]] | None = None,
) -> str | None:
    """Return the filesystem type backing ``path``, or None when unknown.

    ``path`` is fully resolved first: a symlink pointing at a remote mount has
    to report that mount, not the local directory holding the link.
    """
    if table is None:
        table = read_mount_table()
    if not table:
        return None

    try:
        resolved = Path(path).resolve()
    except OSError:
        resolved = Path(os.path.abspath(os.fspath(path)))

    candidate = str(resolved)
    for mountpoint, fstype in table:
        if candidate == mountpoint or candidate.startswith(
            mountpoint.rstrip("/") + "/"
        ):
            return fstype
    return None


def is_network_filesystem(fstype: str | None) -> bool:
    """True when ``fstype`` pays a network round trip per ``open()``."""
    if not fstype:
        return False
    # Match the family too, so nfs4/fuse.foo variants are covered without
    # enumerating every suffix the kernel might report.
    base = fstype.split(".", 1)[0]
    return fstype in NETWORK_FSTYPES or base in NETWORK_FSTYPES


def iter_probe_targets(
    root: str | os.PathLike[str],
    max_depth: int = _PROBE_MAX_DEPTH,
    max_entries: int = _PROBE_MAX_ENTRIES,
) -> list[Path]:
    """Yield ``root`` plus the resolved targets of symlinks below it.

    Bounded breadth-first so a remote root costs a few hundred metadata calls
    rather than a full traversal. Only directory symlinks are followed to a new
    filesystem; that is the shape that matters (an asset subtree relocated onto
    a share), and it keeps the probe cheap.
    """
    start = Path(root)
    targets: list[Path] = [start]
    seen: set[Path] = {start}
    visited = 0

    queue: list[tuple[Path, int]] = [(start, 0)]
    while queue and visited < max_entries:
        current, depth = queue.pop(0)
        if depth >= max_depth:
            continue
        try:
            entries = list(os.scandir(current))
        except OSError:
            continue
        for entry in entries:
            visited += 1
            if visited >= max_entries:
                break
            try:
                if entry.is_symlink():
                    resolved = Path(entry.path).resolve()
                    if resolved not in seen:
                        seen.add(resolved)
                        targets.append(resolved)
                    continue
                if entry.is_dir(follow_symlinks=False):
                    queue.append((Path(entry.path), depth + 1))
            except OSError:
                continue
    return targets


@dataclass(frozen=True)
class RuntimeRoot:
    """One directory the Isaac runtime reads during scene load."""

    name: str
    path: Path
    fstype: str | None
    #: Resolved symlink targets under ``path`` that live on a network mount.
    remote_subpaths: tuple[Path, ...] = ()

    @property
    def networked(self) -> bool:
        return is_network_filesystem(self.fstype) or bool(self.remote_subpaths)


#: Environment variables naming a runtime root, in scene-load cost order.
#: ``probe`` marks the roots worth scanning for relocated subtrees.
RUNTIME_ROOT_VARS: tuple[tuple[str, bool], ...] = (
    ("TARSPHERE_ASSETS_PATH", True),
    ("ISAAC_SIM_PATH", False),
    ("XDG_CACHE_HOME", False),
    ("OMNI_USER_CACHE", False),
)


def describe_runtime_roots(
    env: dict[str, str] | None = None,
    table: list[tuple[str, str]] | None = None,
) -> list[RuntimeRoot]:
    """Classify each configured runtime root by backing filesystem."""
    environ = os.environ if env is None else env
    if table is None:
        table = read_mount_table()

    roots: list[RuntimeRoot] = []
    seen: set[Path] = set()
    for name, probe in RUNTIME_ROOT_VARS:
        raw = (environ.get(name) or "").strip()
        if not raw:
            continue
        path = Path(raw)
        if not path.is_dir():
            continue
        # Shared asset and cache roots are reported once.
        try:
            key = path.resolve()
        except OSError:
            key = path
        if key in seen:
            continue
        seen.add(key)

        fstype = filesystem_type(path, table)
        remote: list[Path] = []
        if probe:
            for target in iter_probe_targets(path):
                if target == path:
                    continue
                if is_network_filesystem(filesystem_type(target, table)):
                    remote.append(target)
        roots.append(
            RuntimeRoot(
                name=name,
                path=path,
                fstype=fstype,
                remote_subpaths=tuple(remote),
            )
        )
    return roots


def format_runtime_root_report(roots: list[RuntimeRoot]) -> str:
    """Render a human-readable filesystem report for the configured roots."""
    if not roots:
        return "No runtime roots configured."

    width = max(len(root.name) for root in roots)
    lines: list[str] = []
    for root in roots:
        marker = "NETWORK" if root.networked else "local"
        lines.append(
            f"  {root.name:<{width}}  {marker:<7}  "
            f"{root.fstype or 'unknown'}  {root.path}"
        )
        for remote in root.remote_subpaths:
            lines.append(f"  {'':<{width}}  {'':<7}  -> relocated onto {remote}")
    return "\n".join(lines)


def networked_roots(roots: list[RuntimeRoot]) -> list[RuntimeRoot]:
    return [root for root in roots if root.networked]


@dataclass(frozen=True)
class StageEntry:
    """One tree to mirror onto local disk."""

    name: str
    source: Path
    destination: Path
    #: Environment variable to repoint at ``destination`` once staged.
    env_var: str | None = None


@dataclass(frozen=True)
class StagePlan:
    scratch_root: Path
    entries: tuple[StageEntry, ...]
    #: Cache variables to export unconditionally; Kit writes these constantly
    #: and a remote cache directory is as costly as a remote asset tree.
    cache_env: tuple[tuple[str, Path], ...]

    def as_shell_lines(self) -> list[str]:
        """Emit the plan as ``KIND\\tNAME\\tSOURCE\\tDEST`` for the staging script."""
        lines = [
            f"STAGE\t{entry.name}\t{entry.source}\t{entry.destination}"
            f"\t{entry.env_var or '-'}"
            for entry in self.entries
        ]
        lines += [f"CACHE\t{name}\t-\t{path}\t{name}" for name, path in self.cache_env]
        return lines


def plan_local_stage(
    scratch_root: str | os.PathLike[str],
    env: dict[str, str] | None = None,
    table: list[tuple[str, str]] | None = None,
) -> StagePlan:
    """Build the staging plan for whichever runtime roots are still remote.

    Only networked roots are staged: re-copying a root that already lives on
    local disk would waste tens of gigabytes and, worse, invalidate the caches
    keyed on its absolute path.
    """
    environ = os.environ if env is None else env
    scratch = Path(scratch_root)
    roots = describe_runtime_roots(environ, table)

    # Asset trees are staged whole rather than per relocated subtree: the
    # symlinks inside them are relative, so mirroring the resolved tree in one
    # place keeps every reference intact.
    destinations = {
        "ISAAC_SIM_PATH": scratch / "isaacsim",
        "TARSPHERE_ASSETS_PATH": scratch / "assets",
    }

    entries: list[StageEntry] = []
    for root in roots:
        if root.name not in destinations or not root.networked:
            continue
        entries.append(
            StageEntry(
                name=root.name,
                source=root.path,
                destination=destinations[root.name],
                env_var=root.name,
            )
        )

    cache_env = (
        ("XDG_CACHE_HOME", scratch / "cache" / "xdg"),
        ("OMNI_USER_CACHE", scratch / "cache" / "ov"),
    )
    return StagePlan(
        scratch_root=scratch,
        entries=tuple(entries),
        cache_env=cache_env,
    )


def warn_on_networked_roots(env: dict[str, str] | None = None) -> list[RuntimeRoot]:
    """Report remote runtime roots before scene loading and return them."""
    roots = describe_runtime_roots(env)
    remote = networked_roots(roots)
    if not remote:
        return []

    names = ", ".join(root.name for root in remote)
    print(
        "[WARN] Isaac runtime roots on a network filesystem: "
        f"{names}.\n"
        f"{format_runtime_root_report(remote)}\n"
        "[WARN] Network metadata latency can slow MDL material loading.\n"
        "[WARN] Mirror them to local disk first: "
        "scripts/stage.sh",
        flush=True,
    )
    return remote


# --------------------------------------------------------------------------
# Copying
#
# Done here rather than by shelling out to rsync: rsync is absent from the
# cluster image, and neither of its symlink modes is correct for this tree
# anyway. The asset root is a symlink farm -- absolute links onto the shared
# mount for the big trees, relative in-tree links (``<scene>/models ->
# ../../models``) for the per-scene views. ``--safe-links`` drops the former,
# plain ``--links`` keeps them pointing at NFS so nothing is gained, and
# dereferencing everything (``tar -h``) expands the relative links into 71
# private copies of an 86 GiB tree. The rule that works is: follow links that
# leave the tree once, then point all aliases at their local staged targets.
# --------------------------------------------------------------------------

#: Threads used for the copy. The work is metadata-latency bound, not bandwidth
#: bound, so oversubscribing cores is the point: each thread spends its time
#: blocked in an NFS round trip with the GIL released.
DEFAULT_STAGE_WORKERS = 32

_COPY_BUFFER = 4 * 1024 * 1024

# Kit ships user-facing documentation below this path. It is not imported or
# opened by the runtime, and some cluster installations deliberately make the
# screenshot directory unreadable. Mirroring runtime roots must not fail on a
# non-runtime documentation payload.
_NON_RUNTIME_STAGE_PREFIXES = (Path("kit/data/documents"),)


@dataclass
class StageStats:
    """Outcome of staging one tree."""

    files_copied: int = 0
    files_skipped: int = 0
    bytes_copied: int = 0
    directories: int = 0
    symlinks: int = 0
    errors: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []

    def merge(self, other: StageStats) -> None:
        self.files_copied += other.files_copied
        self.files_skipped += other.files_skipped
        self.bytes_copied += other.bytes_copied
        self.directories += other.directories
        self.symlinks += other.symlinks
        self.errors.extend(other.errors)


def plan_tree_copy(
    source: str | os.PathLike[str],
    destination: str | os.PathLike[str],
) -> list[tuple[str, Path, Path]]:
    """Return ``(kind, source, destination)`` work items for one tree.

    ``kind`` is ``dir``, ``file``, ``symlink``, ``staged_symlink`` or ``error``.
    External referents are expanded once. Internal links and repeated external
    aliases become relative links to the staged copy; for ``staged_symlink``
    the second tuple member is that staged target, rather than a source path.
    Dangling external links retain their original target as ``symlink`` items.
    """
    try:
        src_root = Path(source).resolve()
    except OSError:
        src_root = Path(source).absolute()
    dst_root = Path(destination).absolute()

    items: list[tuple[str, Path, Path]] = [("dir", src_root, dst_root)]
    # Map expanded source roots to their staged locations. Looking up ancestor
    # roots also handles links into a shared subtree and back-edges within it.
    staged_roots: dict[Path, Path] = {src_root: dst_root}

    def staged_target(resolved: Path) -> Path | None:
        for ancestor in (resolved, *resolved.parents):
            if ancestor in staged_roots:
                return staged_roots[ancestor] / resolved.relative_to(ancestor)
        return None

    stack: list[tuple[Path, Path]] = [(src_root, dst_root)]

    while stack:
        src_dir, dst_dir = stack.pop()
        try:
            entries = sorted(os.scandir(src_dir), key=lambda entry: entry.name)
        except OSError as error:
            items.append(("error", src_dir, Path(str(error))))
            continue

        for entry in entries:
            src_path = Path(entry.path)
            dst_path = dst_dir / entry.name
            try:
                try:
                    relative = src_path.relative_to(src_root)
                except ValueError:
                    relative = None
                if relative is not None and any(
                    relative == prefix or prefix in relative.parents
                    for prefix in _NON_RUNTIME_STAGE_PREFIXES
                ):
                    continue
                if entry.is_symlink():
                    resolved = src_path.resolve()
                    mapped = staged_target(resolved)
                    if mapped is not None:
                        items.append(("staged_symlink", mapped, dst_path))
                        continue
                    if not resolved.exists():
                        # A dangling link is reproduced as a link rather than
                        # failing the whole stage.
                        items.append(("symlink", src_path, dst_path))
                        continue
                    if resolved.is_dir():
                        staged_roots[resolved] = dst_path
                        items.append(("dir", resolved, dst_path))
                        stack.append((resolved, dst_path))
                    else:
                        staged_roots[resolved] = dst_path
                        items.append(("file", resolved, dst_path))
                    continue

                if entry.is_dir(follow_symlinks=False):
                    items.append(("dir", src_path, dst_path))
                    stack.append((src_path, dst_path))
                else:
                    items.append(("file", src_path, dst_path))
            except OSError as error:
                items.append(("error", src_path, Path(str(error))))
    return items


def _needs_copy(source: Path, destination: Path) -> bool:
    """True unless destination already matches on size and whole-second mtime.

    The same heuristic rsync uses by default: cheap enough to run over 100k
    files and strong enough to make a re-run resume instead of restart.
    """
    try:
        src_stat = source.stat()
        dst_stat = destination.stat()
    except OSError:
        return True
    return src_stat.st_size != dst_stat.st_size or int(src_stat.st_mtime) != int(
        dst_stat.st_mtime
    )


def _drop_page_cache(fileno: int, length: int) -> None:
    """Ask the kernel to evict this file's pages; best effort, never fatal.

    Staging moves ~105 GiB, and every byte of it lands in the page cache. Inside
    a memory-capped cgroup that cache counts against ``memory.current``, so a
    plain copy leaves the cgroup pinned at its ceiling: measured here at
    117675 MB of 117760 MB, after which Isaac's own allocations fail. The
    kernel reports it as a SIGSEGV inside Kit rather than an OOM kill, because
    reclaim keeps returning just enough to avoid the killer while starving the
    workload. Dropping each file as we finish with it keeps the copy's
    footprint flat -- the data is on disk and Kit will fault back in only the
    fraction of the tree its scene actually references.
    """
    if not hasattr(os, "posix_fadvise"):  # non-Linux, e.g. a dev laptop
        return
    try:
        os.posix_fadvise(fileno, 0, length, os.POSIX_FADV_DONTNEED)
    except OSError:
        pass


def _copy_file(source: Path, destination: Path) -> int:
    """Copy one file and preserve mode and mtime; return bytes written."""
    tmp = destination.with_name(destination.name + ".hca-staging")
    try:
        with open(source, "rb") as src, open(tmp, "wb") as dst:
            written = 0
            while True:
                chunk = src.read(_COPY_BUFFER)
                if not chunk:
                    break
                dst.write(chunk)
                written += len(chunk)
            # Flush before advising: DONTNEED cannot evict a dirty page, so
            # without the fsync the write cache would survive the hint and the
            # cgroup would fill up exactly as before.
            dst.flush()
            os.fsync(dst.fileno())
            _drop_page_cache(dst.fileno(), written)
            _drop_page_cache(src.fileno(), written)
        src_stat = source.stat()
        os.chmod(tmp, src_stat.st_mode & 0o7777)
        os.utime(tmp, (src_stat.st_atime, src_stat.st_mtime))
        # Rename last so an interrupted copy never leaves a short file that the
        # size/mtime check would then accept.
        os.replace(tmp, destination)
        return written
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def stage_tree(
    source: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    workers: int = DEFAULT_STAGE_WORKERS,
    on_progress=None,
) -> StageStats:
    """Mirror ``source`` to ``destination`` on local disk, idempotently."""
    from concurrent.futures import ThreadPoolExecutor

    stats = StageStats()
    items = plan_tree_copy(source, destination)

    # Directories and symlinks first, serially: every file copy depends on its
    # parent existing, and there are few enough of them that threading adds
    # only ordering hazards.
    for kind, src, dst in items:
        if kind == "dir":
            try:
                dst.mkdir(parents=True, exist_ok=True)
                stats.directories += 1
            except OSError as error:
                stats.errors.append(f"mkdir {dst}: {error}")
        elif kind == "error":
            stats.errors.append(f"scan {src}: {dst}")

    for kind, src, dst in items:
        if kind not in {"symlink", "staged_symlink"}:
            continue
        try:
            target = (
                os.path.relpath(src, dst.parent)
                if kind == "staged_symlink"
                else os.readlink(src)
            )
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.is_symlink() or dst.exists():
                if dst.is_symlink() and os.readlink(dst) == target:
                    stats.symlinks += 1
                    continue
                dst.unlink()
            os.symlink(target, dst)
            stats.symlinks += 1
        except OSError as error:
            stats.errors.append(f"symlink {dst}: {error}")

    files = [(src, dst) for kind, src, dst in items if kind == "file"]
    total = len(files)
    done = 0

    def copy_one(pair: tuple[Path, Path]) -> tuple[int, int, str | None]:
        src, dst = pair
        if not _needs_copy(src, dst):
            return (0, 0, None)
        try:
            return (1, _copy_file(src, dst), None)
        except OSError as error:
            return (0, 0, f"copy {src}: {error}")

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for copied, written, error in pool.map(copy_one, files):
            done += 1
            if error:
                stats.errors.append(error)
            elif copied:
                stats.files_copied += 1
                stats.bytes_copied += written
            else:
                stats.files_skipped += 1
            if on_progress is not None:
                on_progress(done, total, stats)

    return stats


def _format_bytes(count: int) -> str:
    value = float(count)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}TiB"


def stage_plan(plan: StagePlan, workers: int = DEFAULT_STAGE_WORKERS) -> StageStats:
    """Execute ``plan``: mirror every entry and create the cache directories."""
    import time

    overall = StageStats()
    for entry in plan.entries:
        print(f"== staging {entry.name}", flush=True)
        print(f"   {entry.source}\n   -> {entry.destination}", flush=True)
        last = [0.0]

        def report(done: int, total: int, stats: StageStats) -> None:
            now = time.monotonic()
            if now - last[0] < 5.0 and done != total:
                return
            last[0] = now
            percent = 100.0 * done / total if total else 100.0
            print(
                f"   {done}/{total} files ({percent:.1f}%)  "
                f"copied={stats.files_copied} skipped={stats.files_skipped}  "
                f"{_format_bytes(stats.bytes_copied)}",
                flush=True,
            )

        started = time.monotonic()
        stats = stage_tree(entry.source, entry.destination, workers, report)
        elapsed = time.monotonic() - started
        print(
            f"   done in {elapsed:.0f}s: {stats.files_copied} copied, "
            f"{stats.files_skipped} unchanged, {stats.directories} dirs, "
            f"{stats.symlinks} symlinks, {_format_bytes(stats.bytes_copied)}, "
            f"{len(stats.errors)} errors",
            flush=True,
        )
        for message in stats.errors[:20]:
            print(f"   ERROR {message}", flush=True)
        overall.merge(stats)

    for _name, path in plan.cache_env:
        path.mkdir(parents=True, exist_ok=True)
    return overall


def drop_tree_page_cache(root: str | os.PathLike[str]) -> tuple[int, int]:
    """Evict a staged tree from the page cache; return (files, bytes advised).

    Needed for trees staged before the copy learned to drop as it went, and
    after any bulk read of the tree. Only clean pages are affected, so this
    cannot lose data -- the worst case is that Kit faults a page back in.
    """
    files = 0
    total = 0
    for dirpath, _dirnames, filenames in os.walk(root, followlinks=False):
        for name in filenames:
            path = os.path.join(dirpath, name)
            if os.path.islink(path):
                continue
            try:
                fileno = os.open(path, os.O_RDONLY)
            except OSError:
                continue
            try:
                size = os.fstat(fileno).st_size
                _drop_page_cache(fileno, size)
                files += 1
                total += size
            except OSError:
                pass
            finally:
                os.close(fileno)
    return files, total


def main(argv: list[str] | None = None) -> int:
    """``python -m tarsphere.infra.simulation.local_stage [--plan|--stage SCRATCH]``."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--plan",
        metavar="SCRATCH_ROOT",
        help="emit a tab-separated staging plan for this local scratch root",
    )
    group.add_argument(
        "--stage",
        metavar="SCRATCH_ROOT",
        help="mirror every remote runtime root into this local scratch root",
    )
    group.add_argument(
        "--drop-cache",
        metavar="DIR",
        help="release DIR from the page cache, freeing cgroup memory for Kit",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_STAGE_WORKERS,
        help=f"parallel copy threads (default {DEFAULT_STAGE_WORKERS})",
    )
    args = parser.parse_args(argv)

    if args.plan:
        for line in plan_local_stage(args.plan).as_shell_lines():
            print(line)
        return 0

    if args.stage:
        plan = plan_local_stage(args.stage)
        if not plan.entries:
            print("No runtime root reports a network filesystem; nothing to copy.")
        stats = stage_plan(plan, args.workers)
        print(
            f"\nStaged {stats.files_copied} files "
            f"({_format_bytes(stats.bytes_copied)}) with {len(stats.errors)} errors."
        )
        print("\nExport these so every run uses the local copies:")
        print(f"export TARSPHERE_LOCAL_SCRATCH={args.stage}")
        for entry in plan.entries:
            if entry.env_var:
                print(f"export {entry.env_var}={entry.destination}")
        for name, path in plan.cache_env:
            print(f"export {name}={path}")
        return 1 if stats.errors else 0

    if args.drop_cache:
        files, total = drop_tree_page_cache(args.drop_cache)
        print(f"Advised {files} files ({_format_bytes(total)}) out of the page cache.")
        return 0

    roots = describe_runtime_roots()
    print("Isaac runtime roots by filesystem:")
    print(format_runtime_root_report(roots))
    return 1 if networked_roots(roots) else 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
