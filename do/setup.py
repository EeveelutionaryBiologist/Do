"""`Do --setup` -- fetch the backend and model, install the user service."""

import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path


LLAMA_RELEASE_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
LLAMA_TAG_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{tag}"
LLAMA_ASSET_SUFFIX = "-bin-ubuntu-x64.tar.gz"
LLAMA_TAG_ASSET = "nightly-tag.txt"
LLAMA_RELEASES_PAGE = "https://github.com/ggml-org/llama.cpp/releases"
HF_RESOLVE = "https://huggingface.co/{repo}/resolve/main/{filename}"

UNIT_NAME = "dod.service"
UNIT_TEMPLATE = """[Unit]
Description=Do translation daemon
After=default.target

[Service]
Type=simple
ExecStart={dod}
Restart=on-failure
RestartSec=5

Nice=10
CPUWeight=20

NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectKernelTunables=true
ProtectControlGroups=true
RestrictSUIDSGID=true
RestrictNamespaces=true
MemoryHigh=2G
MemoryMax=3G

[Install]
WantedBy=default.target
"""


class SetupError(Exception):
    """A setup step that could not finish, with a message worth printing."""


def _download(url: str, dest: Path, label: str) -> None:
    """Stream `url` to `dest` via a temp file, reporting progress."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            with open(tmp, "wb") as handle:
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    handle.write(chunk)
                    done += len(chunk)
                    if total:
                        pct = done * 100 // total
                        print(f"\r  {label}: {pct}% "
                              f"({done >> 20} / {total >> 20} MB)",
                              end="", file=sys.stderr, flush=True)
            print(file=sys.stderr)
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        tmp.unlink(missing_ok=True)
        raise SetupError(f"could not download {url}: {exc}") from exc
    tmp.replace(dest)


def _get_json(url: str):
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return json.loads(response.read())
    except (urllib.error.URLError, OSError, ValueError, TimeoutError) as exc:
        raise SetupError(
            f"could not reach {url}: {exc}\n"
            f"  install llama.cpp yourself from {LLAMA_RELEASES_PAGE}") from exc


def _match_asset(release) -> tuple[str, str] | None:
    for asset in release.get("assets", []):
        name = asset.get("name", "")
        if name.endswith(LLAMA_ASSET_SUFFIX):
            return name, asset["browser_download_url"]
    return None


def _latest_llama_asset() -> tuple[str, str]:
    """Return (name, url) of the newest prebuilt plain-CPU linux build."""
    release = _get_json(LLAMA_RELEASE_API)
    match = _match_asset(release)
    if match:
        return match

    pointer = next((a for a in release.get("assets", [])
                    if a.get("name") == LLAMA_TAG_ASSET), None)
    if pointer is None:
        raise SetupError(
            f"no {LLAMA_ASSET_SUFFIX} asset and no {LLAMA_TAG_ASSET} in "
            f"{release.get('tag_name', 'latest')}\n"
            f"  install llama.cpp yourself from {LLAMA_RELEASES_PAGE}")

    try:
        with urllib.request.urlopen(pointer["browser_download_url"],
                                    timeout=30) as response:
            tag = response.read().decode("utf-8", "replace").strip()
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise SetupError(f"could not read {LLAMA_TAG_ASSET}: {exc}") from exc

    match = _match_asset(_get_json(LLAMA_TAG_API.format(tag=tag)))
    if match is None:
        raise SetupError(
            f"no {LLAMA_ASSET_SUFFIX} asset in {tag}\n"
            f"  install llama.cpp yourself from {LLAMA_RELEASES_PAGE}")
    return match


def _extract_into(archive: Path, dest: Path) -> None:
    """Unpack the archive's single top-level directory flat into dest."""
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent) as staging:
        with tarfile.open(archive) as tar:
            try:
                tar.extractall(staging, filter="data")
            except TypeError:
                tar.extractall(staging)
        roots = list(Path(staging).iterdir())
        source = roots[0] if len(roots) == 1 and roots[0].is_dir() else Path(staging)
        for entry in source.iterdir():
            target = dest / entry.name
            if target.is_symlink() or target.exists():
                target.unlink()
            shutil.move(str(entry), str(target))


def ensure_backend(config) -> bool:
    """Put a runnable llama-server on disk if one is not reachable already."""
    found = shutil.which(config.server_binary)
    if found:
        print(f"backend: {found} (already on PATH)")
        return False

    managed = config.bin_dir / config.server_binary
    if managed.is_file() and os.access(managed, os.X_OK):
        print(f"backend: {managed} (already installed)")
        return False

    name, url = _latest_llama_asset()
    config.bin_dir.mkdir(parents=True, exist_ok=True)
    archive = config.bin_dir / name
    print(f"backend: fetching {name}")
    _download(url, archive, "backend")
    _extract_into(archive, config.bin_dir)
    archive.unlink(missing_ok=True)

    if not (managed.is_file() and os.access(managed, os.X_OK)):
        raise SetupError(f"{name} did not contain {config.server_binary}")
    print(f"backend: {managed}")
    return True


def ensure_model(config) -> bool:
    """Download the GGUF into model_dir unless it is already there."""
    if config.model_path.is_file():
        print(f"model: {config.model_path} (already present)")
        return False

    config.model_dir.mkdir(parents=True, exist_ok=True)
    url = HF_RESOLVE.format(repo=config.repo_id, filename=config.model_file)
    print(f"model: fetching {config.model_file}")
    _download(url, config.model_path, "model")
    print(f"model: {config.model_path}")
    return True


def _dod_path() -> str:
    found = shutil.which("dod")
    if found:
        return found
    sibling = Path(sys.executable).resolve().parent / "dod"
    if sibling.is_file():
        return str(sibling)
    raise SetupError("could not locate the `dod` entry point on PATH")


def ensure_service(config) -> bool:
    """Write, reload and enable the dod systemd user unit."""
    if not shutil.which("systemctl"):
        print("service: systemctl not found, skipping "
              "(start the daemon yourself with `dod`)")
        return False

    unit_dir = config.config_dir.parent / "systemd" / "user"
    unit_dir.mkdir(parents=True, exist_ok=True)
    unit_path = unit_dir / UNIT_NAME
    unit = UNIT_TEMPLATE.format(dod=_dod_path())

    if unit_path.is_file() and unit_path.read_text() == unit:
        print(f"service: {unit_path} (already installed)")
    else:
        unit_path.write_text(unit)
        print(f"service: {unit_path}")

    for command in (["systemctl", "--user", "daemon-reload"],
                    ["systemctl", "--user", "enable", "--now", UNIT_NAME]):
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise SetupError(f"{' '.join(command)} failed: "
                             f"{result.stderr.strip() or result.returncode}")
    print(f"service: enabled and started ({UNIT_NAME})")
    return True


def run(config) -> int:
    """Run every setup step in order, skipping whatever is already done."""
    config.data_dir.mkdir(parents=True, exist_ok=True)
    steps = (ensure_backend, ensure_model, ensure_service)
    try:
        for step in steps:
            step(config)
    except SetupError as exc:
        print(f"setup failed: {exc}", file=sys.stderr)
        return 1
    print("\nReady. Try: Do \"list all files, also hidden ones\"")
    return 0
