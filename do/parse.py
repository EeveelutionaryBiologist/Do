"""Turn a command string into something the rule table can match against."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List

import shlex

try:
    import bashlex
    _HAS_BASHLEX = True
except ImportError:
    _HAS_BASHLEX = False


_REDIRECT_OPS = {">", ">>", "2>", "<"}

_WRAPPER_ARG_OPTS = {
    "sudo":  {"-u", "-g", "-p", "-C", "-U", "-r", "-t", "-T"},
    "xargs": {"-n", "-P", "-I", "-d", "-s", "-a", "-E", "-L"},
}


class Unresolved(Enum):
    """Why a stage could not be fully understood."""
    COMMAND_SUBST = "command_substitution"   # $(...) or backticks
    EVAL = "eval"                            # eval, sh -c
    UNQUOTED_VAR = "unquoted_variable"       # $DIR in a target position
    PARSE_FAILURE = "parse_failure"          # bashlex raised, or shlex gave up


@dataclass(frozen=True)
class Redirect:
    op: str          # ">", ">>", "2>", "<"
    target: str      # "/dev/sda", "out.txt"


@dataclass(frozen=True)
class Stage:
    """One command in a pipeline."""
    head: str                                # "rm", "curl" -- after unwrapping sudo
    flags: frozenset[str]                    # {"-r", "-f"} -- "-rf" already split
    args: tuple[str, ...]                    # positional targets, in order
    redirects: tuple[Redirect, ...] = ()
    wrappers: frozenset[str] = frozenset()   # {"sudo"}, {"xargs"}, {"find-exec"}
    unresolved: frozenset[Unresolved] = frozenset()
    raw: str = ""                            # this stage's original text


@dataclass(frozen=True)
class ParsedCommand:
    stages: tuple[Stage, ...]
    raw: str

    @property
    def unresolved(self) -> frozenset[Unresolved]:
        """Union across stages -- convenience for whole-command rules.
        Makes it so that 
            if parsed.unresolved:
        evaluates correctly.
        """
        return frozenset().union(*(stage.unresolved for stage in self.stages))


def parse(command: str) -> ParsedCommand:
    """Public entry point."""
    command = command.strip()

    if not command:
        return ParsedCommand(stages=(), raw=command)
    try:
        # TODO: prefer _parse_bashlex when _HAS_BASHLEX; shlex is the only
        # backend for now, so dispatching on it would just route to a stub.
        return _parse_shlex(command)
    except Exception:
        # _parse_shlex catches the ValueError it expects, but
        # a bug anywhere below must not crash the CLI -- and must not be
        # mistaken for a safe command either.
        return ParsedCommand(
            stages=(Stage(head="", flags=frozenset(), args=(),
                          unresolved=frozenset({Unresolved.PARSE_FAILURE}),
                          raw=command),),
            raw=command,
        )


def _lex(command: str) -> list[str]:
    """Tokenize once, quote-aware. Raises ValueError on unbalanced quotes."""
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    return list(lexer)


def _split_flags(tokens: list[str]) -> tuple[frozenset[str], tuple[str, ...]]:
    """Partition tokens into flags and positionals, expanding clustered short
        flags: "-rf" -> {"-r", "-f"}. Note "--" ends flag parsing, and a lone "-"
        is a positional (it means stdin)."""
    flags: set[str] = set()
    args: list[str] = []

    for token in tokens:
        if token.startswith("--") and len(token) > 2:
            flags.add(token)
        elif token.startswith("-") and len(token) > 1:
            for char in token[1:]:
                flags.add(f"-{char}")
        else:
            args.append(token)

    return frozenset(flags), tuple(args)


def _stage_from_tokens(tokens: List[str]) -> Stage:    
    # 1. If the token group is somehow empty -
    if not tokens:
        return Stage(head="", flags=frozenset(), args=(), raw="")

    raw_text = " ".join(tokens)

    # 2. Pull off leading wrappers (e.g., sudo, xargs) and wrapper flags
    wrappers: set[str] = set()
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token in _WRAPPER_ARG_OPTS.keys():
            wrapper = tokens[i]
            wrappers.add(wrapper)
            i += 1
            # Skip optional wrapper flags (e.g., -u root after sudo)

            while tokens[i].startswith("-"):
                opt = tokens[i]
                i += 1

                if "=" not in opt and opt in _WRAPPER_ARG_OPTS[wrapper]:
                    i += 1
        else:
            break

    remaining_tokens = tokens[i:]
    if not remaining_tokens:
        return Stage(head="", flags=frozenset(), args=(), wrappers=frozenset(wrappers), raw=raw_text)

    # 3. Head is the first remaining token
    head = remaining_tokens[0]
    remaining_tokens = remaining_tokens[1:]

    # 4. Scan for redirect operators (">", ">>", "2>", "<") and extract them
    redirects: list[Redirect] = []
    clean_tokens: list[str] = []
    
    j = 0
    while j < len(remaining_tokens):
        tok = remaining_tokens[j]
        if tok in _REDIRECT_OPS and j + 1 < len(remaining_tokens):
            redirects.append(Redirect(op=tok, target=remaining_tokens[j + 1]))
            j += 2
        else:
            clean_tokens.append(tok)
            j += 1

    # 5. Split flags and positional arguments
    flags, args = _split_flags(clean_tokens)

    # 6. Scan raw text for unresolved constructs ($(, `, bare $VAR)
    unresolved: set[Unresolved] = set()
    if "$(" in raw_text or "`" in raw_text:
        unresolved.add(Unresolved.COMMAND_SUBST)
    
    if "eval" in raw_text.split():
        unresolved.add(Unresolved.EVAL)

    # Simple heuristic for unquoted variables in target positions (e.g., unquoted $VAR in args)
    for arg in args:
        if "$" in arg and not (arg.startswith('"') and arg.endswith('"')) and not arg.startswith("'"):
            unresolved.add(Unresolved.UNQUOTED_VAR)

    return Stage(
        head=head,
        flags=flags,
        args=args,
        redirects=tuple(redirects),
        wrappers=frozenset(wrappers),
        unresolved=frozenset(unresolved),
        raw=raw_text
    )


def _parse_shlex(command: str) -> ParsedCommand:
    try:
        tokens = _lex(command)
    except ValueError:
        # Earliest possible fail case
        return ParsedCommand(
            stages=(Stage(head="", flags=frozenset(), args=(),
                          unresolved=frozenset({Unresolved.PARSE_FAILURE}),
                          raw=command),),
            raw=command,
        )

    groups: list[list[str]] = [[]]

    for tok in tokens:
        if tok == "|":
            groups.append([])
        else:
            groups[-1].append(tok)

    stages = tuple(_stage_from_tokens(g) for g in groups if g)

    return ParsedCommand(
        stages=stages,
        raw=command
    )


def _parse_bashlex(command: str) -> ParsedCommand:
    ...
