import os
import subprocess

def execute(command: str) -> int:
    """Subprocess dispatch of the actual command"""
    shell = os.environ.get("SHELL") or "/bin/sh"
    result = subprocess.run([shell, "-c", command])

    return result.returncode
