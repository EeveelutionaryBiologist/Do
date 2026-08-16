
import json


ENCODING = "utf-8"


def encode(message: dict) -> bytes:
    return (json.dumps(message, ensure_ascii=False,
                       separators=(",", ":")) + "\n").encode(ENCODING)


def decode(line: bytes) -> dict:
    return json.loads(line.decode(ENCODING))


def read_line(sock, limit: int = 1 << 20) -> bytes:
    """Read one newline-terminated message. Returns b"" on clean close."""
    chunks = []
    total = 0
    while True:
        chunk = sock.recv(4096)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)
        total += len(chunk)
        if b"\n" in chunk or total > limit:
            break
    return b"".join(chunks).split(b"\n", 1)[0]


def error(message: str, code: str = "error") -> dict:
    return {"ok": False, "code": code, "error": message}
