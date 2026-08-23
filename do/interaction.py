import sys
import tty
import termios

try:
    import gnureadline as readline
except ImportError:
    import readline

from do.render import key_hints


EXIT_INTERRUPTED = 130


def listen_for_key() -> str:
    """One keypress, no Enter. Restores the terminal on every path."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
        # Handle special case if user hits Enter (returns carriage return or newline)
        if ch in ('\r', '\n'):
            return 'ENTER'
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def prompt_for_action(tier: str, yolo: bool = False) -> str:
    """-> 'run' | 'edit' | 'cancel'"""
    if tier == "deny" and not yolo:
        return "cancel"

    print(key_hints())

    while True:
        try:
            key = listen_for_key()
        except KeyboardInterrupt:
            print("\nInterrupted.", file=sys.stderr)
            sys.exit(EXIT_INTERRUPTED)

        match key:
            case "ENTER":
                return "run"
            case "e":
                return "edit"
            case "q":
                return "cancel"
            case _:
                continue


def edit_command(command: str) -> str:
    """readline, pre-filled with `command`. Returns the edited line."""
    def _prefill():
        readline.insert_text(command)
        readline.redisplay()

    readline.set_pre_input_hook(_prefill)
    try:
        edited_cmd = input()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(EXIT_INTERRUPTED)
    finally:
        readline.set_pre_input_hook()

    return edited_cmd
