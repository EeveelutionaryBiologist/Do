
"""The tier engine: a command string in, a verdict out."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import parse as parse_mod
from .rules import (CWD_TARGETS, DELETE_HEADS, ORDER, RULES, Context, Rule,
                    Scope, Tier, is_recursive)


MAX_ENTRIES = 10_000
BUDGET_S = 0.15


@dataclass(frozen=True)
class Verdict:
    tier: Tier
    reasons: tuple[str, ...] = ()
    rule_ids: tuple[str, ...] = ()
    blast_radius: dict | None = None
    command: str = ""

    @property
    def runnable(self) -> bool:
        """DENY needs --force; everything else runs on a single Enter."""
        return self.tier is not Tier.DENY

    def to_dict(self) -> dict:
        return {"tier": self.tier.value, "reasons": list(self.reasons),
                "rule_ids": list(self.rule_ids),
                "blast_radius": self.blast_radius}


def analyze(command: str, cwd=None, context: Context | None = None) -> Verdict:
    """Tier a command and explain why."""
    command = (command or "").strip()
    if not command:
        return Verdict(tier=Tier.OK, command=command)

    if context is None:
        context = Context(cwd=Path(cwd) if cwd else None)
    elif cwd is not None and context.cwd is None:
        context = Context(cwd=Path(cwd), path_lookup=context.path_lookup,
                          allow_fs=context.allow_fs)

    parsed = parse_mod.parse(command)
    matched: list[Rule] = []

    for rule in RULES:
        try:
            if rule.scope is Scope.COMMAND:
                hit = bool(rule.matcher(parsed, context))
            else:
                hit = any(rule.matcher(stage, context) for stage in parsed.stages)
        except Exception:
            matched.append(_ERROR_RULE)
            continue
        if hit:
            matched.append(rule)

    tier = Tier.OK
    for rule in matched:
        if ORDER.index(rule.tier) > ORDER.index(tier):
            tier = rule.tier

    blast = None
    if tier is not Tier.OK and context.allow_fs:
        blast = _blast_radius_for(parsed, context)

    # Ordered by severity so the CLI can print the worst reason first, then by
    # table order so the output is stable across runs.
    matched.sort(key=lambda r: (-ORDER.index(r.tier), r.id))

    return Verdict(
        tier=tier,
        reasons=tuple(rule.reason for rule in matched),
        rule_ids=tuple(rule.id for rule in matched),
        blast_radius=blast,
        command=command,
    )


_ERROR_RULE = Rule("internal.matcher_error", Tier.WARN, Scope.COMMAND,
                   lambda *_: False,
                   "a safety rule failed to evaluate; treating as unreviewed",
                   catches=(), allows=())


# -- blast radius -----------------------------------------------------------

def _deletion_targets(parsed, context) -> list[str]:
    """Paths a WARN/DENY-tier command would remove, in the order given."""
    targets: list[str] = []
    for stage in parsed.stages:
        if stage.head in DELETE_HEADS and is_recursive(stage):
            targets.extend(stage.args)
        elif stage.head == "find" and "-delete" in stage.raw:
            # find's first positional is the search root; the tokenizer shreds
            # its long single-dash predicates, so take args that look like paths.
            targets.extend(a for a in stage.args if not a.startswith(("+", "-")))
    return targets


def _blast_radius_for(parsed, context) -> dict | None:
    for target in _deletion_targets(parsed, context):
        cleaned = target.strip().strip("'\"")
        if cleaned in CWD_TARGETS and context.cwd is not None:
            # `.` and `*` both mean "everything here". Measuring the cwd
            # overcounts slightly for `*`, which the shell expands without
            # dotfiles -- and overcounting is the safe direction for a number
            # whose only job is to make someone pause before pressing Enter.
            measured = blast_radius(str(context.cwd))
        elif not cleaned or any(ch in cleaned for ch in "$*?["):
            continue                # unresolvable: refuse to guess at a number
        else:
            measured = blast_radius(cleaned, cwd=context.cwd)
        if measured is not None:
            return measured
    return None


def blast_radius(target: str, cwd: Path | None = None,
                 max_entries: int = MAX_ENTRIES,
                 budget_s: float = BUDGET_S) -> dict | None:
    """
    Count what `target` covers, read-only and time-boxed.
    Turns an abstract warning into a concrete fact.
    """
    path = Path(os.path.expanduser(target))
    if not path.is_absolute() and cwd is not None:
        path = Path(cwd) / path

    try:
        stat = path.lstat()
    except OSError:
        return None

    if not path.is_dir() or path.is_symlink():
        # A symlink is not followed: deleting it costs the link, not the tree.
        return {"target": str(path), "files": 1, "dirs": 0,
                "bytes": stat.st_size, "truncated": False}

    files = dirs = total_bytes = 0
    seen = 0
    truncated = False
    deadline = time.monotonic() + budget_s
    stack = [str(path)]

    while stack:
        if seen >= max_entries or time.monotonic() > deadline:
            truncated = True
            break
        current = stack.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    seen += 1
                    if seen >= max_entries:
                        truncated = True
                        break
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            dirs += 1
                            stack.append(entry.path)
                        else:
                            files += 1
                            total_bytes += entry.stat(follow_symlinks=False).st_size
                    except OSError:
                        continue
        except OSError:
            continue            # unreadable directory: skip, do not abort

    return {"target": str(path), "files": files, "dirs": dirs,
            "bytes": total_bytes, "truncated": truncated}


def format_blast_radius(blast: dict | None) -> str:
    """One line, for the CLI to print under the command."""
    if not blast:
        return ""
    size = _human_bytes(blast["bytes"])
    parts = [f"{blast['files']:,} files", size]
    if blast["dirs"]:
        parts.append(f"{blast['dirs']:,} directories")
    suffix = " (counting stopped early)" if blast["truncated"] else ""
    return ", ".join(parts) + suffix


def _human_bytes(count: int) -> str:
    value = float(count)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"
