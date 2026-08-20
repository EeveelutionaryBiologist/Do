
"""`dod` -- the socket server and the model's lifecycle.

The daemon holds the socket, the cache, and the safety verdict; the model lives
in a child process that this file starts on demand and kills when it goes cold.
"""

import concurrent.futures
import platform
import signal
import socket
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path

from . import config as config_mod
from . import protocol
from . import safety
from . import store as store_mod
from .backend import BackendError, LlamaServer
from .prompt import PROMPT_VERSION

SERVER_WORKERS = 4

# How often the reaper wakes to check whether the model has gone cold. 
REAP_INTERVAL = 30.0


class Daemon:
    """Owns the socket, the backend, and the verdict.
    The safety analysis runs here rather than in the client so that a future
    `Do!`, an editor plugin, and an eventual Rust client all get the same
    answer without reimplementing the rule table. The client owns the terminal
    and nothing else.
    """

    def __init__(self, config):
        self.config = config
        self.backend = LlamaServer(config)
        self.stop_event = threading.Event()
        self.config.data_dir.mkdir(parents=True, exist_ok=True)
        self._analyze_fn = getattr(safety, "analyze", None)
        self._store = self._open_store()

        self.started_at = time.time()
        self.translations = 0
        self.cache_hits = 0

    def _open_store(self):
        opener = getattr(store_mod, "Store", None)
        if opener is None:
            return None
        try:
            return opener(self.config)
        except Exception:
            traceback.print_exc()
            return None

    # -- verdict ---------------------------------------------------------

    def _analyze(self, command: str, cwd: str = "") -> dict:
        """Tier a command, or say plainly that it cannot."""
        if self._analyze_fn is None:
            return {"tier": "warn",
                    "reasons": ["safety layer not implemented (do/safety.py)"],
                    "blast_radius": None}

        # cwd matters: warn.recursive_delete_outside_cwd is the rule that tells
        # `rm -rf ./build` from `rm -rf /tmp/build`, and it cannot without it.
        verdict = self._analyze_fn(command, cwd=cwd or None)

        if isinstance(verdict, dict):
            tier = verdict.get("tier", "warn")
            reasons = verdict.get("reasons", [])
            blast = verdict.get("blast_radius")
        else:
            tier = getattr(verdict, "tier", "warn")
            reasons = getattr(verdict, "reasons", [])
            blast = getattr(verdict, "blast_radius", None)

        return {"tier": str(getattr(tier, "value", tier)).lower(),
                "reasons": list(reasons),
                "blast_radius": blast}

    # -- operations ------------------------------------------------------

    def handle(self, request: dict) -> dict:
        op = request.get("op", "translate")

        if op == "translate":
            return self._translate(request)
        if op == "analyze":
            return self._analyze_op(request)
        if op == "feedback":
            return self._feedback(request)
        if op == "status":
            return self._status()
        if op == "unload":
            return {"ok": True, "unloaded": self.backend.stop("requested")}
        if op == "shutdown":
            self.stop_event.set()
            return {"ok": True, "stopping": True}

        return protocol.error(f"unknown op: {op}", "bad_request")

    def _translate(self, request: dict) -> dict:
        text = (request.get("prompt") or "").strip()
        if not text:
            return protocol.error("empty prompt", "bad_request")

        cwd = request.get("cwd") or ""
        shell = request.get("shell") or ""
        started = time.perf_counter()

        cached = False
        command = None

        if self._store is not None and request.get("use_cache", True):
            lookup = getattr(self._store, "lookup", None)
            if lookup is not None:
                hit = lookup(text)
                if hit:
                    command, cached = hit, True
                    self.cache_hits += 1

        if command is None:
            try:
                result = self.backend.generate(
                    text, cwd=cwd, shell=shell,
                    os_name=platform.system(),
                    listing=request.get("listing") or "")
            except BackendError as exc:
                return protocol.error(str(exc), exc.code)
            command = result["command"]

        latency_ms = round((time.perf_counter() - started) * 1000, 1)
        self.translations += 1

        if not command:
            # The model declines rather than inventing something. 
            # Basically, there is no executable outcome here. 
            # Likely result of a nonsensical query.
            return {"ok": True, "id": None, "command": "", "tier": "ok",
                    "reasons": ["no shell command expresses this request"],
                    "blast_radius": None, "cached": cached,
                    "latency_ms": latency_ms}

        verdict = self._analyze(command, cwd)
        request_id = self._record(text, cwd, shell, command, verdict,
                                  latency_ms, cached)

        return {"ok": True, "id": request_id, "command": command,
                "tier": verdict["tier"], "reasons": verdict["reasons"],
                "blast_radius": verdict["blast_radius"],
                "cached": cached, "latency_ms": latency_ms}

    def _analyze_op(self, request: dict) -> dict:
        """Tier a command without generating one -- used to re-check an edit,
        so the client never has to re-`translate` a prompt that didn't change."""
        command = (request.get("command") or "").strip()
        if not command:
            return protocol.error("empty command", "bad_request")

        verdict = self._analyze(command, request.get("cwd") or "")
        return {"ok": True, "command": command, "tier": verdict["tier"],
                "reasons": verdict["reasons"], "blast_radius": verdict["blast_radius"]}

    def _record(self, text, cwd, shell, command, verdict, latency_ms, cached):
        if self._store is None:
            return uuid.uuid4().hex[:12]
        record = getattr(self._store, "record_request", None)
        if record is None:
            return uuid.uuid4().hex[:12]
        try:
            return record(prompt=text, cwd=cwd, shell=shell,
                          os=platform.system(), model=self.config.model_file,
                          prompt_version=PROMPT_VERSION, command=command,
                          tier=verdict["tier"], latency_ms=latency_ms,
                          cached=cached)
        except Exception:
            traceback.print_exc()
            return uuid.uuid4().hex[:12]

    def _feedback(self, request: dict) -> dict:
        if self._store is None:
            return {"ok": True, "recorded": False}
        record = getattr(self._store, "record_outcome", None)
        if record is None:
            return {"ok": True, "recorded": False}
        try:
            record(request_id=request.get("id"),
                   action=request.get("action"),
                   final_command=request.get("final_command"),
                   exit_code=request.get("exit_code"))
        except Exception:
            traceback.print_exc()
            return {"ok": True, "recorded": False}
        return {"ok": True, "recorded": True}

    def _status(self) -> dict:
        return {
            "ok": True,
            "uptime_s": round(time.time() - self.started_at, 1),
            "translations": self.translations,
            "cache_hits": self.cache_hits,
            "prompt_version": PROMPT_VERSION,
            "safety": "ready" if self._analyze_fn else "unimplemented",
            "cache": "ready" if self._store else "unimplemented",
            "idle_timeout_s": self.config.idle_timeout,
            "backend": self.backend.status(),
        }

    # -- idle unload -----------------------------------------------------

    def _reap_loop(self):
        while not self.stop_event.wait(REAP_INTERVAL):
            try:
                self.backend.stop_if_idle()
            except Exception:
                traceback.print_exc()

    # -- serving ---------------------------------------------------------

    def serve(self) -> int:
        path = self.config.socket_path
        if path.exists():
            # A stale socket from a crash would otherwise make bind fail.
            try:
                probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                probe.connect(str(path))
                probe.close()
                print(f"another daemon is live at {path}", file=sys.stderr)
                return 1
            except OSError:
                path.unlink(missing_ok=True)

        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(path))
        path.chmod(0o600)       # the corpus is a log of everything ever queried for
        server.listen(64)
        server.settimeout(0.5)

        # A fixed pool, not a thread per connection: llama-server has one slot,
        # so concurrency past a handful of workers only queues up inside the
        # backend anyway.
        pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=SERVER_WORKERS, thread_name_prefix="translate")

        threading.Thread(target=self._reap_loop, name="reap",
                         daemon=True).start()

        # systemd stops the unit with SIGTERM; without this the child would be
        # orphaned rather than unloaded, and the state file left behind for the
        # next start to reap.
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_: self.stop_event.set())

        print(f"dod listening on {path} "
              f"(model {self.config.model_file}, idle unload "
              f"{self.config.idle_timeout / 60:.0f}m)", file=sys.stderr)

        try:
            while not self.stop_event.is_set():
                try:
                    connection, _ = server.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break
                pool.submit(self._serve_one, connection)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop_event.set()
            pool.shutdown(wait=False)
            server.close()
            path.unlink(missing_ok=True)
            self.backend.stop("daemon shutting down")
            closer = getattr(self._store, "close", None)
            if closer is not None:
                closer()
        return 0

    def _serve_one(self, connection):
        try:
            connection.settimeout(self.config.request_timeout + 30)
            line = protocol.read_line(connection)
            if not line:
                return
            try:
                request = protocol.decode(line)
            except ValueError as exc:
                connection.sendall(protocol.encode(
                    protocol.error(f"bad json: {exc}", "bad_request")))
                return
            try:
                response = self.handle(request)
            except Exception as exc:
                traceback.print_exc()
                response = protocol.error(str(exc), "internal")
            connection.sendall(protocol.encode(response))
        except OSError:
            pass
        finally:
            connection.close()


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="dod", description="Do's translation daemon.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--foreground", action="store_true",
                        help="accepted for symmetry with the docs; dod never "
                             "forks, systemd owns backgrounding")
    parser.add_argument("--verbose", action="store_true",
                        help="let llama-server's own logging through to stderr")
    args = parser.parse_args(argv)

    config = config_mod.load(args.config)
    if args.verbose:
        config = config_mod.replace(config, server_log=True)
    return Daemon(config).serve()


if __name__ == "__main__":
    sys.exit(main())
