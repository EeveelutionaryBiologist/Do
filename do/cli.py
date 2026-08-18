
import argparse
import socket
import os
import sys

from do.config import Config
from do.protocol import encode, decode, read_line


EXIT_OK           = 0
EXIT_CANCELLED    = 1
EXIT_DENIED       = 2
EXIT_NO_DAEMON    = 3
EXIT_INTERRUPTED  = 130


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='Do')
    parser.add_argument('message', default="", nargs='?')
    parser.add_argument('--status', action='store_true')

    return parser


def build_payload(message: str) -> dict:
    return {
        "op": "translate",
        "prompt": message,
        "cwd": os.getcwd(),
        "shell": "zsh",
    }


def call(config: Config, message: dict, timeout: float = 65.0) -> dict:
    """One request, one response. Raises ConnectionError with a
    human message if the daemon is not listening."""
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

        sock.sendall(encode(message))
        sock.shutdown(socket.SHUT_WR)
        return decode(read_line(sock))


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


def print_translation(response: dict) -> int:
    if not response.get("ok", False):
        print(response.get("error", "translation failed"), file=sys.stderr)
        return EXIT_NO_DAEMON

    if not response["command"]:
        reasons = response.get("reasons") or ["no shell command expresses that request"]
        print(reasons[0])
        return EXIT_OK

    print(response["command"])
    return EXIT_OK


def main(argv=None) -> int:
    config = Config()
    args = build_parser().parse_args(argv)

    message = {"op": "status"} if args.status else build_payload(args.message)

    try:
        response = call(config, message)
    except ConnectionError as exc:
        print(exc, file=sys.stderr)
        return EXIT_NO_DAEMON

    if args.status:
        return print_status(response)
    return print_translation(response)


if __name__ == "__main__":
    sys.exit(main())
