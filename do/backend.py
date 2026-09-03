
"""Supervision of the llama-server child process, and the HTTP client for it.
The model is served by a separate process rather than embedded in this one.
"""

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import prompt as prompt_mod
from .config import performance_cores

_LIST_DEVICES_TIMEOUT = 5.0
_GPU_OFFLOAD_ARG = "all"
_GPU_CPU_ONLY_ARG = "0"


def _resolve_binary(config) -> str | None:
    """PATH first, then whatever `Do --setup` installed under bin_dir."""
    found = shutil.which(config.server_binary)
    if found:
        return found
    managed = config.bin_dir / config.server_binary
    if managed.is_file() and os.access(managed, os.X_OK):
        return str(managed)
    return None


def _parse_list_devices(stdout: str) -> bool:
    """True if `llama-server --list-devices` reported at least one device."""
    lines = [ln.strip() for ln in stdout.splitlines() if ln.strip()]
    body = lines[1:]
    if not body or (len(body) == 1 and body[0].lower() == "(none)"):
        return False
    return True


class BackendError(Exception):
    """Carries a machine-readable code so cli.py can distinguish 'go install
    llama.cpp' from 'the model is still loading, try again'."""

    def __init__(self, message: str, code: str = "backend"):
        super().__init__(message)
        self.code = code


def _free_port() -> int:
    """Ask the kernel for an unused port, then hand it to the child."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _rss_kb(pid: int) -> int:
    """Resident size of a process in KB, or 0 if it is gone."""
    try:
        with open(f"/proc/{pid}/status") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return 0


def _cmdline(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as handle:
            return handle.read().decode("utf-8", "replace").replace("\0", " ")
    except OSError:
        return ""


class LlamaServer:
    """Lazily started, idle-unloaded llama-server, with an HTTP client."""

    def __init__(self, config):
        self.config = config
        self._lock = threading.Lock()
        self._process: subprocess.Popen | None = None
        self._port: int | None = None
        self._last_used = 0.0
        self._loaded_at = 0.0
        self._failures = 0
        self._retry_after = 0.0
        self._gpu_offload: bool | None = None

        self.starts = 0
        self.requests = 0

        self._reap_orphan()

    # -- lifecycle -------------------------------------------------------

    @property
    def loaded(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def ensure_running(self):
        with self._lock:
            self._ensure_running_locked()

    def _ensure_running_locked(self):
        if self.loaded:
            self._last_used = time.monotonic()
            return
        if self._process is not None:       # exited on its own since last use
            self._collect_locked("child exited")

        now = time.monotonic()
        if now < self._retry_after:
            raise BackendError(
                f"backend failed to start {self._failures}x; retrying in "
                f"{self._retry_after - now:.0f}s",
                code="backend_backoff")

        try:
            self._start_locked()
        except BackendError:
            self._failures += 1
            # 2, 4, 8 ... 60s. Capped so a transient cause (a machine that was
            # briefly out of memory) recovers without a daemon restart.
            self._retry_after = time.monotonic() + min(2 ** self._failures, 60)
            raise
        self._failures = 0
        self._retry_after = 0.0

    def _probe_gpu_available(self, binary: str) -> bool:
        """Ask this exact llama-server binary whether it sees a usable device."""
        try:
            result = subprocess.run(
                [binary, "--list-devices"], capture_output=True,
                encoding="utf-8", errors="replace",
                timeout=_LIST_DEVICES_TIMEOUT)
        except (OSError, subprocess.TimeoutExpired):
            return False
        if result.returncode != 0:
            return False
        return _parse_list_devices(result.stdout)

    def _resolve_gpu_offload(self, binary: str) -> bool:
        """Resolves and caches, for this process's life, whether to offload."""
        if self._gpu_offload is None:
            self._gpu_offload = bool(self.config.use_gpu) and self._probe_gpu_available(binary)
        return self._gpu_offload

    def _argv(self, port: int) -> list[str]:
        binary = _resolve_binary(self.config)
        if binary is None:
            raise BackendError(
                f"{self.config.server_binary} not found on PATH or in "
                f"{self.config.bin_dir}. Run `Do --setup` to fetch it.",
                code="no_backend")

        argv = [
            binary,
            "-m", str(self.config.model_path),
            "--host", "127.0.0.1",
            "--port", str(port),
            "-c", str(self.config.ctx_size),
            "-t", str(self.config.threads),
            "-tb", str(self.config.threads),
            "--parallel", "1",
            "-ngl", _GPU_OFFLOAD_ARG if self._resolve_gpu_offload(binary) else _GPU_CPU_ONLY_ARG,
        ]

        if self.config.pin_to_fast_cores:
            cores = performance_cores(self.config.threads)
            if cores:
                argv = ["taskset", "-c", ",".join(str(c) for c in cores)] + argv

        return argv

    def _start_locked(self):
        if not self.config.model_path.is_file():
            raise BackendError(
                f"model not found at {self.config.model_path}\n"
                f"  {self.config.download_hint}",
                code="no_model")

        last_error = None
        for attempt in range(3):
            port = _free_port()
            argv = self._argv(port)
            sink = None if self.config.server_log else subprocess.DEVNULL

            try:
                process = subprocess.Popen(
                    argv,
                    stdout=sink, stderr=sink,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
            except OSError as exc:
                raise BackendError(f"could not spawn {argv[0]}: {exc}",
                                   code="spawn_failed") from exc

            self._process = process
            self._port = port
            self._write_state()

            try:
                self._await_health(process, port)
            except BackendError as exc:
                last_error = exc
                self._collect_locked(f"health check failed: {exc}")
                if self._gpu_offload:
                    self._gpu_offload = False
                    print(f"GPU offload failed ({exc}); falling back to "
                          "CPU-only for this session", file=sys.stderr)
                elif exc.code == "model_load_failed":
                    raise               
                time.sleep(0.2 * (attempt + 1))
                continue

            self.starts += 1
            self._loaded_at = time.time()
            self._last_used = time.monotonic()
            return

        raise last_error or BackendError("backend did not become healthy",
                                         code="unhealthy")

    def _await_health(self, process: subprocess.Popen, port: int):
        """Poll /health until the model is resident."""
        deadline = time.monotonic() + self.config.startup_timeout
        url = f"http://127.0.0.1:{port}/health"

        while time.monotonic() < deadline:
            code = process.poll()
            if code is not None:
                raise BackendError(
                    f"llama-server exited with code {code} before serving"
                    + ("" if self.config.server_log else
                       " (set backend.server_log = true to see why)"),
                    code="model_load_failed")
            try:
                with urllib.request.urlopen(url, timeout=2) as response:
                    if response.status == 200:
                        return
            except urllib.error.HTTPError as exc:
                if exc.code not in (503, 500):
                    raise BackendError(f"health check returned {exc.code}",
                                       code="unhealthy") from exc
            except (urllib.error.URLError, OSError, TimeoutError):
                pass                        # not listening yet
            time.sleep(0.1)

        raise BackendError(
            f"backend did not report healthy within "
            f"{self.config.startup_timeout:.0f}s", code="startup_timeout")

    def stop(self, reason: str = "requested") -> bool:
        with self._lock:
            if self._process is None:
                return False
            self._collect_locked(reason)
            return True

    def _collect_locked(self, reason: str):
        process, self._process = self._process, None
        self._port = None
        self._loaded_at = 0.0
        self._clear_state()

        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        print(f"backend stopped ({reason})", file=sys.stderr)

    def stop_if_idle(self) -> bool:
        """Called from the reaper thread. Returns True if it unloaded."""
        with self._lock:
            if not self.loaded:
                return False
            idle = time.monotonic() - self._last_used
            if idle < self.config.idle_timeout:
                return False
            self._collect_locked(f"idle {idle / 60:.0f}m")
            return True

    # -- orphan recovery -------------------------------------------------

    def _write_state(self):
        """Record pid and port so a daemon starting after a crash can find the
        child that outlived it. Without this, a SIGKILLed daemon leaks a
        1.2 GB process that nothing will ever reap."""
        try:
            path = self.config.backend_state_path
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"pid": self._process.pid,
                                       "port": self._port,
                                       "model": str(self.config.model_path)}))
            tmp.replace(path)
        except OSError:
            pass                            # advisory only

    def _clear_state(self):
        try:
            self.config.backend_state_path.unlink(missing_ok=True)
        except OSError:
            pass

    def _reap_orphan(self):
        path = self.config.backend_state_path
        try:
            state = json.loads(path.read_text())
        except (OSError, ValueError):
            return

        pid = state.get("pid")
        
        if isinstance(pid, int) and self.config.server_binary in _cmdline(pid):
            try:
                os.kill(pid, signal.SIGTERM)
                print(f"reaped orphaned backend (pid {pid})", file=sys.stderr)
            except OSError:
                pass
        self._clear_state()

    # -- inference -------------------------------------------------------

    def generate(self, request: str, cwd: str = "", shell: str = "",
                 os_name: str = "", listing: str = "") -> dict:
        """Translate one request. Returns {command, timings}."""
        self.ensure_running()

        with self._lock:
            port = self._port
            self._last_used = time.monotonic()
        if port is None:
            raise BackendError("backend went away mid-request", code="unhealthy")

        payload = {
            "prompt": prompt_mod.build(request, cwd=cwd, shell=shell,
                                       os_name=os_name, listing=listing),
            "temperature": self.config.temperature,
            "n_predict": self.config.n_predict,
            "stop": prompt_mod.stop_tokens(),
            "grammar": prompt_mod.GRAMMAR,
            "cache_prompt": True,
            "stream": False,
        }

        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/completion", data=body,
            headers={"Content-Type": "application/json"}, method="POST")

        try:
            with urllib.request.urlopen(
                    req, timeout=self.config.request_timeout) as response:
                result = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:200]
            raise BackendError(f"backend returned {exc.code}: {detail}",
                               code="inference_failed") from exc
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            self.stop(f"request failed: {exc}")
            raise BackendError(f"backend request failed: {exc}",
                               code="inference_failed") from exc

        self.requests += 1
        timings = result.get("timings") or {}
        return {
            "command": prompt_mod.extract(result.get("content", "")),
            "prefill_ms": round(timings.get("prompt_ms", 0.0), 1),
            "decode_ms": round(timings.get("predicted_ms", 0.0), 1),
            "tokens": timings.get("predicted_n", 0),
        }

    # -- introspection ---------------------------------------------------

    def status(self) -> dict:
        with self._lock:
            loaded = self.loaded
            pid = self._process.pid if loaded else None
            idle = (time.monotonic() - self._last_used) if self._last_used else None
            return {
                "loaded": loaded,
                "pid": pid,
                "port": self._port,
                "rss_mb": round(_rss_kb(pid) / 1024, 1) if pid else 0.0,
                "loaded_for_s": round(time.time() - self._loaded_at, 1) if loaded else 0.0,
                "idle_s": round(idle, 1) if idle is not None else None,
                "starts": self.starts,
                "requests": self.requests,
                "model": self.config.model_file,
                "threads": self.config.threads,
                "gpu_offload": self._gpu_offload,
            }
