
import os
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path


DEFAULT_REPO = "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF"
DEFAULT_MODEL_FILE = "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
DEFAULT_THREADS = 4
DEFAULT_CTX = 4096


def _xdg(var: str, fallback: str) -> Path:
    value = os.environ.get(var)
    return Path(value) if value else Path.home() / fallback


def performance_cores(count: int) -> list[int]:
    """
    Pick `count` CPUs to pin llama-server to, fastest physical cores first.
    In all honesty, this is overengineered for what it does. 
    """
    base = Path("/sys/devices/system/cpu")
    seen_cores: set[str] = set()
    candidates: list[tuple[int, int]] = []   

    try:
        cpu_dirs = sorted(base.glob("cpu[0-9]*"),
                          key=lambda p: int(p.name[3:]))
    except OSError:
        return []

    for cpu_dir in cpu_dirs:
        index = int(cpu_dir.name[3:])
        try:
            siblings = (cpu_dir / "topology/thread_siblings_list").read_text().strip()
            max_khz = int((cpu_dir / "cpufreq/cpuinfo_max_freq").read_text())
        except (OSError, ValueError):
            return []                           # partial data is worse than none
        if siblings in seen_cores:
            continue                            # already took this physical core
        seen_cores.add(siblings)
        candidates.append((max_khz, index))

    # Fastest first, then by CPU index so the choice is stable across runs.
    candidates.sort(key=lambda pair: (-pair[0], pair[1]))
    return sorted(index for _, index in candidates[:count])


@dataclass
class Config:
    model_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "models")
    model_file: str = DEFAULT_MODEL_FILE
    repo_id: str = DEFAULT_REPO

    server_binary: str = "llama-server"     # resolved against PATH
    threads: int = DEFAULT_THREADS
    ctx_size: int = DEFAULT_CTX
    pin_to_fast_cores: bool = True
    use_gpu: bool = False
    server_log: bool = False                # inherit llama-server's stderr

    # Lazy load, idle unload. 
    idle_timeout: float = 15 * 60
    startup_timeout: float = 90.0           # first load faults in ~1 GB from disk
    request_timeout: float = 60.0

    # Greedy. The SQLite cache in store.py returns the previously
    # generated command for a repeated prompt.
    temperature: float = 0.0
    n_predict: int = 96

    data_dir: Path = field(default_factory=lambda: _xdg("XDG_DATA_HOME", ".local/share") / "do")
    config_dir: Path = field(default_factory=lambda: _xdg("XDG_CONFIG_HOME", ".config") / "do")

    @property
    def model_path(self) -> Path:
        return self.model_dir / self.model_file

    @property
    def db_path(self) -> Path:
        return self.data_dir / "do.db"

    @property
    def runtime_dir(self) -> Path:
        runtime = os.environ.get("XDG_RUNTIME_DIR")
        return Path(runtime) if runtime else Path("/tmp")

    @property
    def socket_path(self) -> Path:
        return self.runtime_dir / "do.sock"

    @property
    def backend_state_path(self) -> Path:
        """Where the running llama-server's pid and port are recorded, so a
        daemon that starts after a crash can reap the orphan it left behind."""
        return self.runtime_dir / "do-backend.json"

    @property
    def download_hint(self) -> str:
        """Fixes a missing model. Shown verbatim in the
        error, because 'model not found' without the remedy is a bug report."""
        return (f"hf download {self.repo_id} {self.model_file} "
                f"--local-dir {self.model_dir}")


def load(path: Path = None) -> Config:
    """Read config.toml if present. Every key is optional; anything absent keeps
    its dataclass default."""
    cfg = Config()
    path = path or (cfg.config_dir / "config.toml")

    if not path.is_file():
        return cfg

    with open(path, "rb") as handle:
        raw = tomllib.load(handle)

    updates = {}

    model = raw.get("model", {})
    for key in ("model_file", "repo_id", "temperature", "n_predict"):
        if key in model:
            updates[key] = model[key]
    if "model_dir" in model:
        updates["model_dir"] = Path(model["model_dir"]).expanduser()

    backend = raw.get("backend", {})
    for key in ("server_binary", "threads", "ctx_size", "pin_to_fast_cores",
                "use_gpu", "server_log", "idle_timeout", "startup_timeout",
                "request_timeout"):
        if key in backend:
            updates[key] = backend[key]

    for key in ("data_dir", "config_dir"):
        if key in raw.get("paths", {}):
            updates[key] = Path(raw["paths"][key]).expanduser()

    return replace(cfg, **updates)
