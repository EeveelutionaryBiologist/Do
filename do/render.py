
import sys
from collections.abc import Sequence

from do.safety import format_blast_radius


class bcolors:
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    DENY = '\033[91m'
    ENDC = '\033[0m'


def supports_color(args,) -> bool:
    """isatty AND not NO_COLOR AND TERM != 'dumb'."""
    if sys.stdout.isatty() and not args.no_color and not args.dumb:
        return True
    return False
 
def blast_line(blast: dict | None, color: bool) -> str:
    """Pass forward the blast string from the corresponding safety function"""
    if not blast:
        return ""
    
    blast_string = format_blast_radius(blast=blast)

    if color:
        return f"{bcolors.WARNING}Blast radius: {blast_string}{bcolors.ENDC}"

    return f"Blast radius: {blast_string}"

def key_hints(color: bool=False) -> str:
    """Input hints for the user"""
    return "[Enter] run   [e] edit   [q] cancel"

def ok_command(command: str, color: bool) -> str:
    """No warning/deny issued -> print green if color"""
    if color:
        return f"{bcolors.OKGREEN}{command}{bcolors.ENDC}"
    else: 
        return f"{command}"

def render_yellow(text: str) -> str:
    """Warning highlighting outside of the regular warning context"""
    return f"{bcolors.WARNING}{text}{bcolors.ENDC}"

def warning(command: str, reasons: Sequence[str], color: bool) -> str:
    """Render as warning"""
    reason_str = "\n".join(reasons)

    if color:
        return f"{bcolors.WARNING}{command}{bcolors.ENDC}\n{reason_str}"
    else: 
        return f"{command}\n{reason_str}"

def denial(command: str, reasons: Sequence[str], color: bool, yolo: bool=False) -> str:
    """Render asd denial"""
    reason_str = "\n".join(reasons)

    if not yolo:
        if color:
            return f"{bcolors.DENY}[DENIED]{bcolors.ENDC}\n{reason_str}"
        else: 
            return f"[DENIED]\n{reason_str}"
    else:
        if color:
            return f"{bcolors.DENY}{command}{bcolors.ENDC}\n{reason_str}"
        else: 
            return f"{command}\n{reason_str}"

def color_response(response: dict, color):
    """Passes reesponse forward to rendering functions"""
    command = response["command"]

    # NOTE: DENY is currently caught before, so we only handle warn/ok here - 
    if response["tier"] == "warn":
        return warning(command, response["reasons"], color)
    else:
        return ok_command(command, color)
    