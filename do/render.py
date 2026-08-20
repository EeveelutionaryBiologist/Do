
from collections.abc import Sequence


class bcolors:
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    DENY = '\033[91m'
    ENDC = '\033[0m'


def supports_color(stream) -> bool:
    """isatty AND not NO_COLOR AND TERM != 'dumb'."""

def blast_line(blast: dict | None, color: bool) -> str:
    if not blast:
        return ""

    outlines = "Blast radius:"

    for (k, v) in blast.items():
        if color:
            outlines += f"\n{bcolors.WARNING}{k}:\t{v}{bcolors.ENDC}"
        else:
            outlines += f"\n{k}:\t{v}"

    return outlines

def key_hints(color: bool=False) -> str:
    return "[Enter] run   [e] edit   [q] cancel"

def ok_command(command: str, color: bool) -> str:
    if color:
        return f"{bcolors.OKGREEN}{command}{bcolors.ENDC}"
    else: 
        return f"{command}"

def render_yellow(text: str) -> str:
    return f"{bcolors.WARNING}{text}{bcolors.ENDC}"

def warning(command: str, reasons: Sequence[str], color: bool) -> str:
    reason_str = "\n".join(reasons)

    if color:
        return f"{bcolors.WARNING}{command}{bcolors.ENDC}\n{reason_str}"
    else: 
        return f"{command}\n{reason_str}"

def denial(command: str, reasons: Sequence[str], color: bool, yolo: bool=False) -> str:
    reason_str = "\n".join(reasons)

    if not yolo:
        if color:
            return f"{bcolors.DENY}[DENIED]{bcolors.ENDC}\n{reason_str}"
        else: 
            return f"[DENIED]{reason_str}"
    else:
        if color:
            return f"{bcolors.DENY}{command}{bcolors.ENDC}\n{reason_str}"
        else: 
            return f"{command}{reason_str}"

def color_response(response: dict):
    command = response["command"]

    if response["tier"] == "deny":
        return denial(response["reasons"], True)
    elif response["tier"] == "warn":
        return warning(command, response["reasons"], True)
    else:
        return ok_command(command, True)
    