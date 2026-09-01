"""Tests for do.parse.

Everything above the "bashlex-specific" section runs against whichever
backend parse() actually dispatches to (bashlex when the optional `parse`
extra is installed, shlex otherwise) and asserts on the same Stage/
ParsedCommand contract either way -- these cases hold under both backends
by construction, so they aren't backend-aware at all."""

import pytest
from do import parse as parse_mod
from do.parse import ParsedCommand, Redirect, Stage, Unresolved, parse

requires_bashlex = pytest.mark.skipif(
    not parse_mod._HAS_BASHLEX,
    reason="needs the optional `parse` extra (bashlex) installed")


def one(command: str) -> Stage:
    """Most cases are single-stage; unwrap so assertions stay readable."""
    parsed = parse(command)
    assert len(parsed.stages) == 1, f"expected 1 stage, got {len(parsed.stages)}"
    return parsed.stages[0]


# --- flags and positionals -------------------------------------------------

@pytest.mark.parametrize("command, head, flags, args", [
    ("rm -rf ./build",      "rm",  {"-r", "-f"}, ("./build",)),
    ("rm -r -f ./build",    "rm",  {"-r", "-f"}, ("./build",)),
    ("chmod +x script.sh",  "chmod", set(),      ("+x", "script.sh")),
    ("ls -",                "ls",  set(),        ("-",)),
    # dd's key=value operands are not flags -- they land in args, which is
    # what the "dd of=/dev/sd*" DENY rule matches on.
    ("dd if=a of=/dev/sda", "dd",  set(),        ("if=a", "of=/dev/sda")),
])
def test_flag_splitting(command, head, flags, args):
    stage = one(command)
    assert stage.head == head
    assert stage.flags == flags
    assert stage.args == args


# --- wrappers --------------------------------------------------------------

@pytest.mark.parametrize("command, head, wrappers", [
    ("sudo apt update",       "apt", {"sudo"}),
    ("sudo -u root rm -rf /", "rm",  {"sudo"}),
    ("xargs rm -rf",          "rm",  {"xargs"}),
])
def test_wrapper_unwrapping(command, head, wrappers):
    stage = one(command)
    assert stage.head == head
    assert stage.wrappers == wrappers


# --- pipelines -------------------------------------------------------------

def test_pipeline_splits_into_ordered_stages():
    parsed = parse("curl x.com | sh")
    assert [s.head for s in parsed.stages] == ["curl", "sh"]


def test_quoted_pipe_is_not_a_stage_boundary():
    parsed = parse('echo "a|b" | wc -l')
    assert [s.head for s in parsed.stages] == ["echo", "wc"]


@pytest.mark.parametrize("command, heads", [
    ("cd /tmp && rm -rf ~",       ["cd", "rm"]),
    ("cd /tmp; rm -rf ~",         ["cd", "rm"]),
    ("cmd1 && cmd2 || cmd3",      ["cmd1", "cmd2", "cmd3"]),
    ("rm -rf ./safe & rm -rf ~",  ["rm", "rm"]),
    ("a; b; c",                   ["a", "b", "c"]),
])
def test_control_operators_are_stage_boundaries(command, heads):
    # Regression: only "|" split stages, so "rm" in "cd /tmp && rm -rf ~"
    # was never its own stage head -- every DELETE_HEADS matcher was blind
    # to it. See TODO.md item 1.
    parsed = parse(command)
    assert [s.head for s in parsed.stages] == heads


def test_quoted_control_operator_is_not_a_stage_boundary():
    parsed = parse('echo "a;b && c" | wc -l')
    assert [s.head for s in parsed.stages] == ["echo", "wc"]


# --- redirects -------------------------------------------------------------

def test_redirect_without_space_is_detected():
    # Matters because the "redirect into a block device" DENY rule compares
    # the operator by equality; ">/dev/sda" as one token would slip past it.
    stage = one("dd if=a >/dev/sda")
    assert stage.redirects == (Redirect(op=">", target="/dev/sda"),)


# --- unresolved / fail-closed ---------------------------------------------

def test_unresolved_unions_across_stages():
    parsed = parse('rm -rf $DIR | tee `date`.log')
    assert parsed.stages[0].unresolved == {Unresolved.UNQUOTED_VAR}
    assert parsed.stages[1].unresolved == {Unresolved.COMMAND_SUBST}
    assert parsed.unresolved == {Unresolved.UNQUOTED_VAR, Unresolved.COMMAND_SUBST}


def test_unbalanced_quote_fails_closed():
    stage = one("echo 'unterminated")
    assert Unresolved.PARSE_FAILURE in stage.unresolved


def test_swallowed_command_fails_closed():
    # Regression: "-n" takes no argument, so the naive "consume the next
    # non-flag token" heuristic ate the entire command and returned an empty
    # head with no failure marker -- rm -rf / scoring OK.
    stage = one("sudo -n rm -rf /")
    assert stage.head == "rm" or Unresolved.PARSE_FAILURE in stage.unresolved


# --- bashlex-specific ------------------------------------------------------
#
# These are real fidelity gains only the bashlex backend delivers -- the
# shlex fallback (a plain tokenizer with no grammar) has no way to draw
# either distinction below. Guarded so the suite still passes clean when the
# optional `parse` extra isn't installed; see the module docstring.

@requires_bashlex
def test_assignment_prefix_does_not_hide_the_head():
    # Regression: shlex has no notion of a leading env-var assignment, so
    # "FOO=bar" itself became the stage head and "rm" was just an arg to
    # it -- every DELETE_HEADS matcher was blind to the real command.
    # "FOO=bar rm -rf /" scored OK. bashlex tags this as its own
    # AssignmentNode kind, so it can be skipped rather than mistaken for
    # the command.
    stage = one("FOO=bar rm -rf /")
    assert stage.head == "rm"
    assert stage.args == ("/",)


@requires_bashlex
def test_escaped_semicolon_in_find_exec_is_not_a_stage_boundary():
    # Regression: shlex unescapes "\;" to a bare ";" before the stage
    # splitter ever sees it, indistinguishable from a real command
    # separator -- "find ... -exec rm {} \; -o -name foo -delete" split
    # into two stages, and the trailing "-o -name foo -delete" stopped
    # being analyzed as part of the find command at all. bashlex's
    # tokenizer understands the escape, so the semicolon stays a literal
    # word inside the find command.
    parsed = parse(r"find . -exec rm {} \; -o -name foo -delete")
    assert [s.head for s in parsed.stages] == ["find"]
    assert "-delete" in parsed.stages[0].raw


@requires_bashlex
def test_control_flow_construct_fails_closed():
    # if/case/function defs don't unconditionally run their contents (an
    # "if" might take either branch; a function definition doesn't call
    # the function) -- out of scope, must escalate rather than be guessed
    # at as some single simple command.
    stage = one("if true; then rm -rf /; fi")
    assert Unresolved.PARSE_FAILURE in stage.unresolved


@requires_bashlex
def test_for_loop_body_is_extracted_as_its_own_stage():
    # Regression: for/while/subshells/brace groups used to be treated the
    # same as if/case -- the whole line collapsed into one PARSE_FAILURE
    # stage, so nothing inside was ever actually analyzed. A loop's body
    # (unlike an "if" branch) unconditionally runs, so it should become a
    # real stage the rule table can see.
    parsed = parse('for file in do/*; do head -n 3 "$file"; done')
    assert [s.head for s in parsed.stages] == ["head"]
    assert parsed.stages[0].args == ("3", "$file")


@requires_bashlex
def test_while_condition_and_body_are_both_extracted():
    # The condition ("read line") is itself a command that runs each
    # iteration, same as the body.
    parsed = parse("while read line; do echo $line; done")
    assert [s.head for s in parsed.stages] == ["read", "echo"]


@requires_bashlex
def test_subshell_and_brace_group_contents_are_extracted():
    assert one("(rm -rf /)").head == "rm"
    assert one("{ rm -rf /; }").head == "rm"
