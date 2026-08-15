"""Tests for do.parse -- shlex backend."""

import pytest
from do.parse import ParsedCommand, Redirect, Stage, Unresolved, parse


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
