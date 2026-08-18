
from collections.abc import Sequence


class bcolors:
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    DENY = '\033[91m'
    ENDC = '\033[0m'


def supports_color(stream) -> bool:
    """isatty AND not NO_COLOR AND TERM != 'dumb'."""

def blast_line(blast: dict | None, color: bool) -> str:
    pass

def key_hints(tier: str, color: bool) -> str:
    return "Accept: [Enter], Back: [Back]"

def ok_command(command: str, color: bool) -> str:
    if color:
        return f"{bcolors.OKGREEN}{command}{bcolors.ENDC}"
    else: 
        return f"{command}"

def warning(command: str, reasons: Sequence[str], color: bool) -> str:
    reason_str = "\n".join(reasons)

    if color:
        return f"{bcolors.WARNING}{command}{bcolors.ENDC}\n{reason_str}"
    else: 
        return f"{command}\n{reason_str}"

def denial(reasons: Sequence[str], color: bool) -> str:
    reason_str = "\n".join(reasons)

    # TODO: We could add an opt-in YOLO mode, where denied commands are surfaced in Red...
    if color:
        return f"{bcolors.DENY}[DENIED]{bcolors.ENDC}\n{reason_str}"
    else: 
        return f"[DENIED]{reason_str}"

def color_response(response: dict):
    command = response["command"]

    if response["tier"] == "deny":
        return denial(response["reasons"], True)
    elif response["tier"] == "warning":
        return warning(command, response["reasons"], True)
    else:
        return ok_command(command, True)
    