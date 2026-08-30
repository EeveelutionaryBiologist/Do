"""The system prompt, the few-shots, and the grammar."""


PROMPT_VERSION = "1.0"
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

SYSTEM = """You translate a natural-language request into a single shell command.

1. Output exactly one line: the shell command. No prose, no markdown, no code fences, no explanation.
2. Translate only what was asked. Add no flags, filters, sorting, error handling, or safety guards that were not requested.
3. Never add sudo unless the request says so.
4. Prefer the plainest coreutils/POSIX tool that does the job.
5. Multi-step requests chain with && in the order stated.
6. Reproduce user-supplied literals -- filenames, paths, quoted strings -- exactly.
7. If the request cannot be expressed as a shell command, output nothing."""

# Chosen to demonstrate restraint, not capability. Every pair here is one where
# an eager model over-delivers: the wrong answer in the comment is the one the
# model produces without the example, and each is a different flavour of
# assuming intent -- adding a flag, adding a pipeline, adding interactivity,
# adding a filter nobody asked for.
FEW_SHOTS = (
    ("list all files, also hidden ones",        "ls -a"),           # not ls -la
    ("show me the running processes",           "ps aux"),          # not ps aux | grep
    ("delete the log files",                    "rm *.log"),        # not rm -i, not find
    ("count the lines in main.py",              "wc -l main.py"),   # not cat | wc -l
    ("copy config.json to config.json.bak",     "cp config.json config.json.bak"),
    ("make a folder called notes and move into it", "mkdir notes && cd notes"),
    ("find files over 100 megabytes",           "find . -size +100M"),  # not -type f
    ("show the git log",                        "git log"),         # not --oneline
)

GRAMMAR = r"""root  ::= ( first rest* )?
first ::= [^`\n\r]
rest  ::= [^\n\r]"""


def _turn(role: str, content: str) -> str:
    return f"{IM_START}{role}\n{content}{IM_END}\n"


def build(request: str, cwd: str = "", shell: str = "", os_name: str = "",
          listing: str = "") -> str:
    """Render the full ChatML prompt for one request."""
    parts = [_turn("system", SYSTEM)]

    for example_request, example_command in FEW_SHOTS:
        parts.append(_turn("user", example_request))
        parts.append(_turn("assistant", example_command))

    context = []
    if shell:
        context.append(f"shell: {shell}")
    if os_name:
        context.append(f"os: {os_name}")
    if cwd:
        context.append(f"cwd: {cwd}")
    if listing:
        context.append(f"files here: {listing}")

    user = "\n".join(context + [request]) if context else request
    parts.append(_turn("user", user))

    parts.append(f"{IM_START}assistant\n")
    return "".join(parts)


def stop_tokens() -> list[str]:
    """Terminal characters - the newline is the real one"""
    return [IM_END, "<|endoftext|>", "\n"]


def extract(raw: str) -> str:
    """Clean a completion into a bare command. Mostly for when the model misbehaves."""
    text = raw.strip()

    if text.startswith("```"):
        text = text[3:]
        if "\n" in text:                    # drop a ```bash language tag
            first, _, remainder = text.partition("\n")
            if first.strip().isalpha():
                text = remainder
    text = text.replace("```", "").strip()

    # A leading "$ " or "A: " is the other thing small models do when they
    # imitate the shape of documentation instead of answering.
    # Hacky, but alas.
    for prefix in ("$ ", "# ", "A: ", "a: "):
        if text.startswith(prefix):
            text = text[len(prefix):]

    return text.splitlines()[0].strip() if text else ""
