import os
import subprocess

def execute(command: str) -> int:
    shell = os.environ.get("SHELL") or "/bin/sh"
    result = subprocess.run([shell, "-c", command])

    return result.returncode
