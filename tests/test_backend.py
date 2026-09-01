"""Tests for do.backend's GPU offload switch. No GPU exists on this machine."""

import subprocess
import tomllib
from dataclasses import replace
from pathlib import Path

import pytest

from do import backend as backend_mod
from do.backend import LlamaServer
from do.config import Config, load as load_config


def make_server(tmp_path, monkeypatch, **overrides) -> LlamaServer:
    """Constructs a LlamaServer isolated from any real running daemon's state."""
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setattr(backend_mod.shutil, "which", lambda name: "/opt/llama-server")
    return LlamaServer(replace(Config(), **overrides))


def test_use_gpu_defaults_to_false():
    assert Config().use_gpu is False


def test_use_gpu_read_from_backend_section(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("[backend]\nuse_gpu = true\n")
    assert load_config(path).use_gpu is True


def test_use_gpu_absent_from_toml_keeps_default(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("[backend]\nthreads = 8\n")
    assert load_config(path).use_gpu is False


def test_shipped_example_config_documents_use_gpu():
    """The shipped config.toml must not drift from the dataclass default."""
    path = Path(__file__).resolve().parent.parent / "config" / "config.toml"
    with open(path, "rb") as handle:
        raw = tomllib.load(handle)
    assert raw["backend"]["use_gpu"] == Config().use_gpu


@pytest.mark.parametrize("stdout, expected", [
    ("Available devices:\n  (none)\n", False),
    ("Available devices:\n", False),
    ("", False),
    ("garbage", False),
    ("Available devices:\n  CUDA0: NVIDIA GeForce RTX 3090 (24576 MiB)\n", True),
    ("Available devices:\n  CUDA0: ...\n  CUDA1: ...\n", True),
])
def test_parse_list_devices(stdout, expected):
    assert backend_mod._parse_list_devices(stdout) is expected


def test_probe_returns_false_when_binary_missing(tmp_path, monkeypatch):
    server = make_server(tmp_path, monkeypatch, use_gpu=True)
    monkeypatch.setattr(backend_mod.subprocess, "run",
                         lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError()))
    assert server._probe_gpu_available("/no/such/llama-server") is False


def test_probe_returns_false_on_timeout(tmp_path, monkeypatch):
    server = make_server(tmp_path, monkeypatch, use_gpu=True)

    def raise_timeout(*a, **k):
        raise subprocess.TimeoutExpired(cmd="llama-server --list-devices", timeout=5)

    monkeypatch.setattr(backend_mod.subprocess, "run", raise_timeout)
    assert server._probe_gpu_available("llama-server") is False


def test_probe_returns_false_on_nonzero_exit(tmp_path, monkeypatch):
    """Covers a llama-server build too old to support --list-devices."""
    server = make_server(tmp_path, monkeypatch, use_gpu=True)
    fake = subprocess.CompletedProcess(args=[], returncode=1, stdout="",
                                        stderr="error: unrecognized argument")
    monkeypatch.setattr(backend_mod.subprocess, "run", lambda *a, **k: fake)
    assert server._probe_gpu_available("llama-server") is False


def test_argv_forces_cpu_only_and_never_probes_when_use_gpu_false(tmp_path, monkeypatch):
    server = make_server(tmp_path, monkeypatch, use_gpu=False)

    def boom(*a, **k):
        raise AssertionError("--list-devices must not run when use_gpu is False")

    monkeypatch.setattr(backend_mod.subprocess, "run", boom)
    assert server._argv(4242)[-2:] == ["-ngl", "0"]


def test_argv_offloads_when_use_gpu_true_and_device_found(tmp_path, monkeypatch):
    server = make_server(tmp_path, monkeypatch, use_gpu=True)
    fake = subprocess.CompletedProcess(
        args=[], returncode=0,
        stdout="Available devices:\n  CUDA0: NVIDIA GeForce RTX 3090 (24576 MiB)\n",
        stderr="")
    monkeypatch.setattr(backend_mod.subprocess, "run", lambda *a, **k: fake)
    assert server._argv(4242)[-2:] == ["-ngl", "all"]


def test_argv_falls_back_to_cpu_when_use_gpu_true_but_no_device(tmp_path, monkeypatch):
    """The exact --list-devices output verified empirically on this machine."""
    server = make_server(tmp_path, monkeypatch, use_gpu=True)
    fake = subprocess.CompletedProcess(args=[], returncode=0,
                                        stdout="Available devices:\n  (none)\n", stderr="")
    monkeypatch.setattr(backend_mod.subprocess, "run", lambda *a, **k: fake)
    assert server._argv(4242)[-2:] == ["-ngl", "0"]


def test_gpu_offload_probed_once_and_cached(tmp_path, monkeypatch):
    server = make_server(tmp_path, monkeypatch, use_gpu=True)
    calls = []

    def fake_run(*a, **k):
        calls.append(a)
        return subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout="Available devices:\n  CUDA0: ...\n", stderr="")

    monkeypatch.setattr(backend_mod.subprocess, "run", fake_run)
    server._argv(1111)
    server._argv(2222)
    server._argv(3333)
    assert len(calls) == 1


def test_status_gpu_offload_is_none_before_first_start_attempt(tmp_path, monkeypatch):
    server = make_server(tmp_path, monkeypatch, use_gpu=True)
    assert server.status()["gpu_offload"] is None


def test_status_reflects_resolved_gpu_offload(tmp_path, monkeypatch):
    server = make_server(tmp_path, monkeypatch, use_gpu=True)
    fake = subprocess.CompletedProcess(args=[], returncode=0,
                                        stdout="Available devices:\n  (none)\n", stderr="")
    monkeypatch.setattr(backend_mod.subprocess, "run", lambda *a, **k: fake)
    server._argv(9999)
    assert server.status()["gpu_offload"] is False
