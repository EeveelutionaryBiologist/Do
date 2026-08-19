
import argparse
import socket
import os
import sys
import termios
import tty

from do.config import Config
from do.protocol import encode, decode, read_line
from do.safety import Tier
from do.render import color_response, denial, blast_line, key_hints


EXIT_OK           = 0
EXIT_CANCELLED    = 1
EXIT_DENIED       = 2
EXIT_NO_DAEMON    = 3
EXIT_INTERRUPTED  = 130


def listen_for_key() -> str:
    """One keypress, no Enter. Restores the terminal on every path."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        ch = sys.stdin.read(1)
        # Handle special case if user hits Enter (returns carriage return or newline)
        if ch in ('\r', '\n'):
            return 'ENTER'
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def prompt_for_action(tier: str, yolo: bool= False) -> bool:
    """
    -> 'run' | 'edit' | 'cancel'
    On return 'True', the code is passed to Execution
    """
    if tier == "deny" and not yolo:
        return False

    print(key_hints())

    while True:
        try:
            key = listen_for_key()
        except KeyboardInterrupt:
            print("\nInterrupted.", file=sys.stderr)
            sys.exit(EXIT_INTERRUPTED)

        match key:
            case "ENTER": 
                # Code execution goes here
                return True
            case "e":
                # TODO: Cmd editing goes here
                raise NotImplementedError("Editing is not yet build!")
            case "q":
                print("Aborted.")
                return False
            case _:
                continue


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='Do')
    parser.add_argument('message', default="", nargs='?')
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument("--yolo", action='store_true')
    parser.add_argument("--no-color", action='store_true')

    return parser


def build_payload(message: str) -> dict:
    return {
        "op": "translate",
        "prompt": message,
        "cwd": os.getcwd(),
        "shell": "zsh",
    }

def call(config: Config, payload: dict, timeout: float = 65.0) -> dict:
    """One request, one response. Raises ConnectionError with a
    if the daemon is not listening."""
    socket_path = str(config.socket_path)

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        try:
            sock.connect(socket_path)
        except (FileNotFoundError, ConnectionRefusedError) as exc:
            raise ConnectionError(
                f"dod is not running (no daemon at {socket_path}). "
                f"Start it with `dod --foreground`."
            ) from exc

        sock.sendall(encode(payload))
        sock.shutdown(socket.SHUT_WR)
        response = decode(read_line(sock))

    return response


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


def print_translation(response: dict, args) -> int:
    color = not args.no_color
    yolo = args.yolo

    if not response.get("ok", False):
        print(response.get("error", "translation failed"), file=sys.stderr)
        return EXIT_NO_DAEMON

    if not response["command"]:
        reasons = response.get("reasons") or ["no shell command expresses that request"]
        print(reasons[0])
        return EXIT_OK

    if response["tier"] == Tier.DENY.value: 
        if not yolo:
            print(denial(response["command"], reasons=response["reasons"], color=color, yolo=False),
                  file=sys.stderr)
            print(blast_line(response["blast_radius"], color), file=sys.stderr)
            return EXIT_DENIED
        else:
            print(denial(response["command"], reasons=response["reasons"], color=color, yolo=True))
    else:
        print(color_response(response))

    print(blast_line(response["blast_radius"], color))

    if args.dry_run:
        return EXIT_OK
         
    if prompt_for_action(response["tier"], yolo):
        print("Executing now.") # <-- Placeholder

    return EXIT_OK


def main(argv=None) -> int:
    config = Config()
    args = build_parser().parse_args(argv)

    payload = {"op": "status"} if args.status else build_payload(args.message)

    try:
        response = call(config, payload)
    except ConnectionError as exc:
        print(exc, file=sys.stderr)
        return EXIT_NO_DAEMON

    # print(response)

    if args.status:
        return print_status(response)

    if sys.stdout.isatty(): 
        return print_translation(response, args)

if __name__ == "__main__":
    sys.exit(main())
