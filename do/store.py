"""SQLite cache and corpus.
Owns one connection for the life of the process. A single internal lock
serializes access from daemon.py's thread pool.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid


SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id             TEXT PRIMARY KEY,          -- uuid4().hex[:12], same shape as daemon.py's own fallback
    ts             REAL    NOT NULL,           -- time.time()
    prompt         TEXT    NOT NULL,           -- raw, exactly as typed -- corpus fidelity matters
    prompt_norm    TEXT    NOT NULL,           -- lower() + whitespace-collapsed; the cache key
    cwd            TEXT    NOT NULL DEFAULT '',
    shell          TEXT    NOT NULL DEFAULT '',
    os             TEXT    NOT NULL DEFAULT '',
    model          TEXT    NOT NULL DEFAULT '',
    prompt_version TEXT    NOT NULL DEFAULT '',
    command        TEXT    NOT NULL,
    tier           TEXT    NOT NULL,
    latency_ms     REAL,
    cached         INTEGER NOT NULL DEFAULT 0  -- daemon.py's _record always passes this kwarg
);

-- Serves lookup()'s "WHERE prompt_norm=? ORDER BY ts DESC LIMIT 1" as one
-- index seek regardless of corpus size.
CREATE INDEX IF NOT EXISTS idx_requests_prompt_norm_ts
    ON requests(prompt_norm, ts DESC);

CREATE TABLE IF NOT EXISTS outcomes (
    request_id     TEXT,               -- requests.id -- not FK-enforced, see record_outcome
    action         TEXT,               -- 'run' | 'edit' | 'cancel', nullable: see record_outcome
    final_command  TEXT,
    exit_code      INTEGER,
    ts             REAL    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_outcomes_request_id ON outcomes(request_id);
"""

_EXPORT_QUERY = """
    SELECT r.*, o.action AS outcome_action,
           o.final_command AS outcome_final_command,
           o.exit_code AS outcome_exit_code, o.ts AS outcome_ts
    FROM requests r
    LEFT JOIN (
        SELECT o1.* FROM outcomes o1
        INNER JOIN (
            SELECT request_id, MAX(ts) AS max_ts FROM outcomes GROUP BY request_id
        ) latest ON o1.request_id = latest.request_id AND o1.ts = latest.max_ts
    ) o ON o.request_id = r.id
    ORDER BY r.ts
"""


def _normalize(prompt: str) -> str:
    """Lowercased, whitespace-collapsed -- the cache key, so a request
    retyped with different casing or spacing still hits."""
    return " ".join(prompt.lower().split())


class Store:
    def __init__(self, config) -> None:
        self.config = config
        self.config.data_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

        self.db = sqlite3.connect(str(config.db_path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.execute("PRAGMA busy_timeout=5000")
        self.db.executescript(SCHEMA)
        self.db.commit()

    # -- cache ------------------------------------------------------------

    def lookup(self, prompt: str) -> str | None:
        prompt_norm = _normalize(prompt)
        with self._lock:
            row = self.db.execute(
                "SELECT command FROM requests WHERE prompt_norm=? "
                "ORDER BY ts DESC LIMIT 1",
                (prompt_norm,),
            ).fetchone()
        return row["command"] if row else None

    # -- corpus -------------------------------------------------------------

    def record_request(self, *, prompt: str, cwd: str, shell: str, os: str,
                        model: str, prompt_version: str, command: str,
                        tier: str, latency_ms: float, cached: bool) -> str:
        prompt_norm = _normalize(prompt)
        ts = time.time()

        with self._lock:
            for _ in range(5):
                request_id = uuid.uuid4().hex[:12]
                try:
                    self.db.execute(
                        "INSERT INTO requests (id, ts, prompt, prompt_norm, cwd, "
                        "shell, os, model, prompt_version, command, tier, "
                        "latency_ms, cached) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (request_id, ts, prompt, prompt_norm, cwd, shell, os,
                         model, prompt_version, command, tier, latency_ms,
                         int(cached)),
                    )
                    self.db.commit()
                    return request_id
                except sqlite3.IntegrityError:
                    continue
            raise sqlite3.IntegrityError(
                "could not generate a unique request id after 5 attempts")

    def record_outcome(self, *, request_id: str | None, action: str | None,
                        final_command: str | None, exit_code: int | None) -> None:
        
        with self._lock:
            self.db.execute(
                "INSERT INTO outcomes (request_id, action, final_command, "
                "exit_code, ts) VALUES (?,?,?,?,?)",
                (request_id, action, final_command, exit_code, time.time()),
            )
            self.db.commit()

    # -- maintenance ------------------------------------------------------

    def forget(self) -> None:
        """Drop and recreate both tables in place"""
        with self._lock:
            self.db.executescript(
                "DROP TABLE IF EXISTS requests; DROP TABLE IF EXISTS outcomes;"
            )
            self.db.executescript(SCHEMA)
            self.db.commit()

    def export_jsonl(self, path: str) -> int:
        """One JSON object per request, each carrying its most recent
        outcome (or null) embedded. Returns the row count written."""
        with self._lock:
            rows = self.db.execute(_EXPORT_QUERY).fetchall()

        count = 0
        with open(path, "w") as handle:
            for row in rows:
                outcome = None
                if row["outcome_action"] is not None:
                    outcome = {
                        "action": row["outcome_action"],
                        "final_command": row["outcome_final_command"],
                        "exit_code": row["outcome_exit_code"],
                        "ts": row["outcome_ts"],
                    }
                record = {
                    "id": row["id"], "ts": row["ts"], "prompt": row["prompt"],
                    "prompt_norm": row["prompt_norm"], "cwd": row["cwd"],
                    "shell": row["shell"], "os": row["os"], "model": row["model"],
                    "prompt_version": row["prompt_version"],
                    "command": row["command"], "tier": row["tier"],
                    "latency_ms": row["latency_ms"], "cached": bool(row["cached"]),
                    "outcome": outcome,
                }
                handle.write(json.dumps(record) + "\n")
                count += 1
        return count

    # -- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            self.db.commit()
            self.db.close()
