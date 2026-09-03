import os
import sys
import socket

from do.config import Config
from do.protocol import encode, decode, read_line


def current_shell_name() -> str:
    """Basename of $SHELL, e.g. "zsh" -- matching the "shell: zsh" shape
    prompt.py's context block expects, and the same $SHELL (with the same
    fallback) execution.py actually runs the command in, so the model is
    never told a different shell than the one about to receive what it
    writes."""
    return os.path.basename(os.environ.get("SHELL") or "/bin/sh")


def build_payload(message: str, use_cache: bool = True) -> dict:
    return {
        "op": "translate",
        "prompt": message,
        "cwd": os.getcwd(),
        "shell": current_shell_name(),
        "use_cache": use_cache,
    }

def build_analyze_payload(command: str) -> dict:
    return {
        "op": "analyze",
        "command": command,
        "cwd": os.getcwd(),
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
                f"Run `Do --setup`, or start it yourself with `dod`."
            ) from exc

        sock.sendall(encode(payload))
        sock.shutdown(socket.SHUT_WR)
        response = decode(read_line(sock))

    return response


def build_feedback(response: dict, action: str, exit_code: int) -> dict | None:
    try:
        return {
            "op": "feedback", 
            "id": response["id"], 
            "action": action, 
            "final_command": response["command"], 
            "exit_code": exit_code,
        }
    except Exception as e:
        return 


def send_feedback(config: Config, feedback: dict) -> bool:
    socket_path = str(config.socket_path)
    
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(65.0)
        try:
            sock.connect(socket_path)
            sock.sendall(encode(feedback))
            sock.shutdown(socket.SHUT_WR)
            response = decode(read_line(sock))
            if response["ok"]:
                return True
                
        except Exception as exc:
            print(f"Feedback not recorded.\n{exc}", file=sys.stderr)

    return False
