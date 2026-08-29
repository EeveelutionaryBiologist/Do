"""Turn a command string into something the rule table can match against."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List

import shlex

try:
    import bashlex
    from bashlex.errors import ParsingError as _BashlexParsingError
    _HAS_BASHLEX = True
except ImportError:
    _HAS_BASHLEX = False


_REDIRECT_OPS = {">", ">>", "2>", "<"}
_STAGE_SEPARATORS = {"|", "&&", "||", ";", ";;", "&"}

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
    backend = _parse_bashlex if _HAS_BASHLEX else _parse_shlex
    try:
        return backend(command)
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
        if tok in _STAGE_SEPARATORS:
            groups.append([])
        else:
            groups[-1].append(tok)

    # A backslash-escaped ";" in "find ... -exec rm {} \;" unescapes to the
    # same bare ";" token as a real separator -- shlex does not keep enough
    # state past tokenization to tell them apart. 
    stages = tuple(_stage_from_tokens(g) for g in groups if g)

    return ParsedCommand(
        stages=stages,
        raw=command
    )


class _Unsupported(Exception):
    """A bashlex node this tool has no reason to translate to and no reason
    to pretend to understand -- conditionals, case statements, function
    definitions. Unlike a loop or a subshell, these don't unconditionally
    run their contents as part of running this command line (an "if" might
    take either branch; a function definition doesn't call the function),
    so there's no single honest reading of "what stages does this line
    run" to extract. Falls back to a whole-command PARSE_FAILURE stage
    rather than guessing at one branch."""


# Kinds whose contents unconditionally run as part of running this command
# line -- a loop's body (and, for while/until, its condition) always
# executes, unlike an "if"/"case" branch or a function body.
_LOOP_KINDS = ("for", "while", "until")


def _flatten_to_commands(node) -> list:
    """Walk a bashlex parse tree, flattening lists, pipelines, loops, and
    subshell/brace groups down to the individual command nodes that become
    Stages. """
    if node.kind == "command":
        return [node]
    if node.kind == "pipeline":
        commands = []
        for part in node.parts:
            if part.kind != "pipe":
                commands.extend(_flatten_to_commands(part))
        return commands
    if node.kind == "list":
        commands = []
        for part in node.parts:
            if part.kind != "operator":
                commands.extend(_flatten_to_commands(part))
        return commands
    if node.kind == "compound":
        # A subshell "(...)" or brace group "{ ...; }" -- just a wrapper;
        # only the "(" / ")" / "{" / "}" reservedwords need skipping.
        commands = []
        for part in node.list:
            if part.kind != "reservedword":
                commands.extend(_flatten_to_commands(part))
        return commands
    if node.kind in _LOOP_KINDS:
        # Skip the syntax markers ("for"/"do"/"done"/...) and, for a "for"
        # loop, the "word" parts -- the loop variable and the
        # list-expression items, not commands. What's left is the body
        # (and, for while/until, the condition -- itself a command that
        # genuinely runs each iteration, e.g. "while read line").
        commands = []
        for part in node.parts:
            if part.kind in ("reservedword", "word"):
                continue
            commands.extend(_flatten_to_commands(part))
        return commands
    raise _Unsupported(node.kind)


def _contains_kind(node, kind: str) -> bool:
    """True if `node` or anything in its subtree has the given bashlex
    `.kind` -- e.g. a commandsubstitution nested inside a word, however
    deep (command substitutions can themselves contain command
    substitutions)."""
    if getattr(node, "kind", None) == kind:
        return True
    for attr in ("parts", "list"):
        for child in getattr(node, attr, None) or ():
            if _contains_kind(child, kind):
                return True
    for attr in ("command", "output", "input"):
        child = getattr(node, attr, None)
        if child is not None and hasattr(child, "kind") and _contains_kind(child, kind):
            return True
    return False


def _redirect_from_node(node, source: str) -> Redirect:
    target = node.output
    if isinstance(target, int):
        target_text = str(target)
    elif target is not None and hasattr(target, "word"):
        target_text = target.word
    else:
        target_text = source[node.pos[0]:node.pos[1]]
    return Redirect(op=node.type, target=target_text)


def _stage_from_command_node(node, source: str) -> Stage:
    raw = source[node.pos[0]:node.pos[1]]

    words = []
    redirects: list[Redirect] = []
    for part in node.parts:
        if part.kind == "redirect":
            redirects.append(_redirect_from_node(part, source))
        elif part.kind == "assignment":
            continue
        elif part.kind == "word":
            words.append(part)

    if not words:
        return Stage(head="", flags=frozenset(), args=(),
                     redirects=tuple(redirects), raw=raw)

    # Pull off leading wrappers (sudo, xargs) and their own flags, same as
    # the shlex path.
    wrappers: set[str] = set()
    i = 0
    while i < len(words):
        text = words[i].word
        if text in _WRAPPER_ARG_OPTS:
            wrappers.add(text)
            i += 1
            while i < len(words) and words[i].word.startswith("-"):
                opt = words[i].word
                i += 1
                if "=" not in opt and opt in _WRAPPER_ARG_OPTS[text]:
                    i += 1
        else:
            break

    remaining = words[i:]
    if not remaining:
        # A wrapper with nothing after it ("sudo -u" alone) -- fail toward
        # WARN rather than silently returning an empty, unremarkable OK
        # stage.
        return Stage(head="", flags=frozenset(), args=(),
                     wrappers=frozenset(wrappers), redirects=tuple(redirects),
                     unresolved=frozenset({Unresolved.PARSE_FAILURE}), raw=raw)

    head = remaining[0].word
    rest = remaining[1:]
    flags, args = _split_flags([w.word for w in rest])

    unresolved: set[Unresolved] = set()
    if any(_contains_kind(w, "commandsubstitution") for w in remaining):
        unresolved.add(Unresolved.COMMAND_SUBST)
    if any(w.word == "eval" for w in remaining):
        unresolved.add(Unresolved.EVAL)
    if any("$" in w.word for w in rest):
        unresolved.add(Unresolved.UNQUOTED_VAR)

    return Stage(
        head=head,
        flags=flags,
        args=args,
        redirects=tuple(redirects),
        wrappers=frozenset(wrappers),
        unresolved=frozenset(unresolved),
        raw=raw,
    )


def _parse_bashlex(command: str) -> ParsedCommand:
    try:
        trees = bashlex.parse(command)
    except _BashlexParsingError:
        return ParsedCommand(
            stages=(Stage(head="", flags=frozenset(), args=(),
                          unresolved=frozenset({Unresolved.PARSE_FAILURE}),
                          raw=command),),
            raw=command,
        )

    try:
        command_nodes = []
        for tree in trees:
            command_nodes.extend(_flatten_to_commands(tree))
    except _Unsupported:
        # Control-flow keywords, subshells, brace groups, function defs --
        return ParsedCommand(
            stages=(Stage(head="", flags=frozenset(), args=(),
                          unresolved=frozenset({Unresolved.PARSE_FAILURE}),
                          raw=command),),
            raw=command,
        )

    stages = tuple(_stage_from_command_node(node, command) for node in command_nodes)
    return ParsedCommand(stages=stages, raw=command)
