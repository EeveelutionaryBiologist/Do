
import argparse
import sys
from pathlib import Path

from do.config import Config
from do import parse as parse_mod
from do.connection import call, send_feedback, build_payload, build_analyze_payload, build_feedback, current_shell_name
from do.safety import Tier
from do.render import color_response, supports_color, denial, blast_line, render_yellow
from do.interaction import edit_command, prompt_for_action
from do.execution import execute


STATE_ONLY_HEADS = frozenset({"cd", "export", "source", "alias", "umask"})
SHELL_INITS = ("zsh", "bash", "fish")

SHELLINIT_DIR = Path(__file__).resolve().parent / "shellinit"

EXIT_OK           = 0
EXIT_CANCELLED    = 1
EXIT_DENIED       = 2
EXIT_NO_DAEMON    = 3
EXIT_INTERRUPTED  = 130


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='Do')
    parser.add_argument('message', default="", nargs='?')
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--no-cache', action='store_true')
    parser.add_argument('--forget', action='store_true')
    parser.add_argument('--export-corpus', nargs='?', const='', default=None,
                        metavar='PATH',
                        help="write the corpus as JSONL to PATH "
                             "(default: <data_dir>/corpus.jsonl)")
    parser.add_argument("--yolo", action='store_true')
    parser.add_argument("--no-color", action='store_true')
    parser.add_argument("--dumb", action="store_true")
    parser.add_argument("--shell-init", choices=SHELL_INITS, default=None,
                        help="print a shell snippet that turns Do into a "
                             "ZLE widget instead of a subprocess")

    return parser


def is_state_only(command: str) -> bool:
    """cd, export, source, alias, umask -- commands that change shell state
    and therefore do nothing from a child process."""
    stages = parse_mod.parse(command).stages
    return bool(stages) and stages[0].head in STATE_ONLY_HEADS


def shell_init_script(shell: str) -> str:
    """The snippet for `Do --shell-init <shell>` -- read verbatim from
    do/shellinit/, not built inline, so it stays six lines of real zsh
    rather than a Python string half the reader has to mentally unescape."""
    return (SHELLINIT_DIR / f"{shell}.sh").read_text()


def print_status(response: dict) -> int:
    if not response.get("ok", False):
        print(response.get("error", "status request failed"), file=sys.stderr)
        return EXIT_NO_DAEMON

    backend = response.get("backend", {})
    print(f"dod up {response['uptime_s']:.0f}s -- "
          f"{response['translations']} translations, "
          f"{response['cache_hits']} cache hits "
          f"(safety: {response['safety']}, cache: {response['cache']})")
    if backend.get("loaded"):
        print(f"backend loaded -- {backend['rss_mb']} MB, "
              f"idle {backend['idle_s']:.0f}s, "
              f"{backend['requests']} requests, {backend['starts']} starts")
    else:
        print("backend not loaded")
    return EXIT_OK


def print_export(response: dict, path: str) -> int:
    if not response.get("ok", False) or not response.get("exported", False):
        print(response.get("error", "export failed"), file=sys.stderr)
        return EXIT_NO_DAEMON
    count = response.get("count")
    if count is not None:
        print(f"Exported {count} rows to {path}")
    else:
        print(f"Exported corpus to {path}")
    return EXIT_OK


def print_forget(response: dict) -> int:
    if not response.get("ok", False) or not response.get("deleted", False):
        print(response.get("error", "forget failed"), file=sys.stderr)
        return EXIT_NO_DAEMON
    print("Cache and corpus cleared.")
    return EXIT_OK


def print_translation_noninteractive(response: dict, args) -> int:
    """Non-TTY output"""
    if not response.get("ok", False):
        print(response.get("error", "translation failed"), file=sys.stderr)
        return EXIT_NO_DAEMON

    if not response["command"]:
        reasons = response.get("reasons") or ["no shell command expresses that request"]
        print(reasons[0], file=sys.stderr)
        return EXIT_OK

    if response["tier"] == Tier.DENY.value and not args.yolo:
        for reason in response["reasons"]:
            print(reason, file=sys.stderr)
        return EXIT_DENIED

    print(response["command"])
    return EXIT_OK


def print_translation(response: dict, args, config: Config) -> int:
    color = supports_color(args)
    yolo = args.yolo
    was_edited = False

    if not response.get("ok", False):
        print(response.get("error", "translation failed"), file=sys.stderr)
        return EXIT_NO_DAEMON

    if not response["command"]:
        reasons = response.get("reasons") or ["no shell command expresses that request"]
        print(reasons[0])
        return EXIT_OK

    while True:
        if is_state_only(response["command"]):
            if color:
                print(render_yellow(response["command"]))
            else:
                print(f"{response['command']}")
            if not args.dry_run:
                shell = current_shell_name()
                hint = shell if shell in SHELL_INITS else SHELL_INITS[0]
                print(f"NOTE: Cannot be executed in-line here, as it would only changes this shell's state "
                    f"(cwd, an env var, ...); a child process can't make that stick to the parent "
                    f"once it exits. Run `Do --shell-init {hint}` once to wire up a "
                    f"shell function that can.", file=sys.stderr)
            return EXIT_CANCELLED

        if response["tier"] == Tier.DENY.value:
            if not yolo:
                print(denial(response["command"], reasons=response["reasons"], color=color, yolo=False),
                      file=sys.stderr)
                print(blast_line(response["blast_radius"], color), file=sys.stderr)
                if args.dry_run:
                    return EXIT_DENIED
                exit_code = EXIT_DENIED
                action = "cancel"
                break
            else:
                print(denial(response["command"], reasons=response["reasons"], color=color, yolo=True))
        else:
            print(color_response(response, color=color))

        blast_radius_ls = blast_line(response["blast_radius"], color)

        if blast_radius_ls:
            print(blast_radius_ls)

        if args.dry_run:
            return EXIT_OK

        action = prompt_for_action(response["tier"], yolo)
        exit_code = EXIT_OK

        if action == "cancel":
            exit_code = EXIT_CANCELLED
            break

        if action == "run":
            print()
            exit_code = execute(command=response["command"])
            break

        if action == "edit": 
            # re-tier before executing -- an edit can turn a
            # WARN command into something that deserves DENY
            was_edited = True
            edited = edit_command(command=response["command"])

            try:
                verdict = call(config, build_analyze_payload(edited))
            except ConnectionError as exc:
                print(exc, file=sys.stderr)
                return EXIT_NO_DAEMON

            if not verdict.get("ok", False):
                print(verdict.get("error", "could not re-check the edited command"),
                    file=sys.stderr)
                return EXIT_NO_DAEMON

            response = {**response, "command": verdict["command"], "tier": verdict["tier"],
                        "reasons": verdict["reasons"], "blast_radius": verdict["blast_radius"]}

    if was_edited:
        action = "edit"

    feedback = build_feedback(response, action, exit_code)

    if feedback:
        send_feedback(config, feedback=feedback)

    return exit_code


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.shell_init:
        sys.stdout.write(shell_init_script(args.shell_init))
        return EXIT_OK

    config = Config()

    export_path = None

    # --status, --export and --forget are caught here as they do
    # not lead into a translation operation.
    if args.status:
        payload = {"op": "status"}
    elif args.export_corpus is not None:
        export_path = (Path(args.export_corpus).expanduser().resolve()
                       if args.export_corpus else config.data_dir / "corpus.jsonl")
        payload = {"op": "export", "path": str(export_path)}
    elif args.forget:
        confirmation = input("Delete cache? [y/n]")
        if confirmation in ["Y", "y"]:
            payload = {"op": "delete"}
        else:
            return EXIT_OK
    else:
        payload = build_payload(args.message)

    try:
        response = call(config, payload)
    except ConnectionError as exc:
        print(exc, file=sys.stderr)
        return EXIT_NO_DAEMON

    if args.status:
        return print_status(response)

    if export_path is not None:
        return print_export(response, str(export_path))

    if args.forget:
        return print_forget(response)

    if sys.stdout.isatty():
        return print_translation(response, args, config)

    return print_translation_noninteractive(response, args)

if __name__ == "__main__":
    sys.exit(main())
