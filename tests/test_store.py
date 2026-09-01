"""do.store -- the cache and corpus. No daemon, no model needed."""

import concurrent.futures
import json
import re
import sqlite3
import threading
import time
from dataclasses import replace

import pytest

from do.config import Config
from do.store import Store


def make_config(tmp_path):
    return replace(Config(), data_dir=tmp_path)


def a_request(store, prompt="list files", command="ls", **overrides):
    kwargs = dict(prompt=prompt, cwd="/home/chris", shell="zsh", os="Linux",
                  model="qwen2.5-coder-1.5b", prompt_version="1.0",
                  command=command, tier="ok", latency_ms=12.3, cached=False)
    kwargs.update(overrides)
    return store.record_request(**kwargs)


def direct_connection(config):
    """A second, independent connection -- for asserting on-disk state
    without going through the Store instance under test."""
    conn = sqlite3.connect(str(config.db_path))
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def test_schema_created_fresh(tmp_path):
    config = make_config(tmp_path)
    store = Store(config)
    assert config.db_path.is_file()
    assert store.lookup("anything") is None
    store.close()


def test_reopening_same_path_is_idempotent(tmp_path):
    config = make_config(tmp_path)
    Store(config).close()
    store = Store(config)  # second open, no writes in between
    store.close()

    conn = direct_connection(config)
    tables = [r["name"] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")]
    assert tables.count("requests") == 1
    assert tables.count("outcomes") == 1
    conn.close()


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def test_lookup_miss_returns_none(tmp_path):
    store = Store(make_config(tmp_path))
    assert store.lookup("some prompt never recorded") is None
    store.close()


def test_record_request_then_lookup_hits(tmp_path):
    store = Store(make_config(tmp_path))
    a_request(store, prompt="list files", command="ls -a")
    assert store.lookup("list files") == "ls -a"
    store.close()


def test_lookup_normalizes_case_and_whitespace(tmp_path):
    config = make_config(tmp_path)
    store = Store(config)
    a_request(store, prompt="  List   Files  ", command="ls -a")

    assert store.lookup("list files") == "ls -a"

    # Corpus fidelity: the raw prompt keeps its original casing/spacing even
    # though prompt_norm is what the cache key is built from.
    conn = direct_connection(config)
    row = conn.execute("SELECT prompt, prompt_norm FROM requests").fetchone()
    assert row["prompt"] == "  List   Files  "
    assert row["prompt_norm"] == "list files"
    conn.close()
    store.close()


def test_lookup_returns_most_recent_on_duplicate_prompt_norm(tmp_path):
    store = Store(make_config(tmp_path))
    a_request(store, prompt="list files", command="ls -a")
    time.sleep(0.01)  # guarantee a distinct, later ts
    a_request(store, prompt="list files", command="ls -la")

    assert store.lookup("list files") == "ls -la"
    store.close()


# ---------------------------------------------------------------------------
# Ids
# ---------------------------------------------------------------------------

def test_record_request_returns_id_shaped_like_daemon_fallback(tmp_path):
    store = Store(make_config(tmp_path))
    request_id = a_request(store)
    assert re.fullmatch(r"[0-9a-f]{12}", request_id)
    store.close()


def test_record_request_ids_are_unique(tmp_path):
    store = Store(make_config(tmp_path))
    ids = {a_request(store, prompt=f"prompt {i}") for i in range(50)}
    assert len(ids) == 50
    store.close()


# ---------------------------------------------------------------------------
# Outcomes
# ---------------------------------------------------------------------------

def test_record_outcome_associates_with_prior_request(tmp_path):
    config = make_config(tmp_path)
    store = Store(config)
    request_id = a_request(store)
    store.record_outcome(request_id=request_id, action="edit",
                         final_command="ls -la", exit_code=0)

    conn = direct_connection(config)
    rows = conn.execute("SELECT * FROM outcomes WHERE request_id=?",
                        (request_id,)).fetchall()
    assert len(rows) == 1
    assert rows[0]["action"] == "edit"
    assert rows[0]["final_command"] == "ls -la"
    assert rows[0]["exit_code"] == 0
    conn.close()
    store.close()


def test_record_outcome_does_not_change_lookup(tmp_path):
    """Regression guard: lookup() must stay a single-table query against
    requests.command. If a future change joins in outcomes so an edit
    silently changes what a later identical prompt returns, that has to be
    a deliberate, reviewed decision -- this test is the one that should fail
    and force that conversation, not let it happen as a side effect."""
    store = Store(make_config(tmp_path))
    request_id = a_request(store, prompt="delete the log files", command="rm *.log")
    before = store.lookup("delete the log files")

    store.record_outcome(request_id=request_id, action="edit",
                         final_command="rm -f *.log", exit_code=0)

    after = store.lookup("delete the log files")
    assert before == after == "rm *.log"
    store.close()


def test_record_outcome_tolerates_missing_fields(tmp_path):
    """A malformed or future client's feedback shouldn't raise -- PLAN-CLI.md:
    a feedback failure must never affect the user's exit code."""
    store = Store(make_config(tmp_path))
    store.record_outcome(request_id=None, action=None, final_command=None,
                         exit_code=None)
    store.close()


def test_record_outcome_does_not_require_an_existing_request(tmp_path):
    """daemon.py's own fallback hands out a bare uuid when record_request
    itself fails, so a feedback call can legitimately reference a request_id
    with no row in requests -- must not raise (no FK enforcement)."""
    store = Store(make_config(tmp_path))
    store.record_outcome(request_id="deadbeef0000", action="run",
                         final_command="ls", exit_code=0)
    store.close()


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

def test_close_persists_data_across_reopen(tmp_path):
    config = make_config(tmp_path)
    store = Store(config)
    a_request(store, prompt="list files", command="ls -a")
    store.close()

    reopened = Store(config)
    assert reopened.lookup("list files") == "ls -a"
    reopened.close()


# ---------------------------------------------------------------------------
# Maintenance
# ---------------------------------------------------------------------------

def test_forget_clears_both_tables_and_store_stays_usable(tmp_path):
    config = make_config(tmp_path)
    store = Store(config)
    request_id = a_request(store, prompt="list files", command="ls -a")
    store.record_outcome(request_id=request_id, action="run",
                         final_command="ls -a", exit_code=0)

    store.forget()

    assert store.lookup("list files") is None
    conn = direct_connection(config)
    assert conn.execute("SELECT COUNT(*) AS n FROM requests").fetchone()["n"] == 0
    assert conn.execute("SELECT COUNT(*) AS n FROM outcomes").fetchone()["n"] == 0
    conn.close()

    # The same instance, no restart, must still work afterward.
    a_request(store, prompt="list files", command="ls -a")
    assert store.lookup("list files") == "ls -a"
    store.close()


def test_export_jsonl_writes_expected_rows(tmp_path):
    store = Store(make_config(tmp_path))
    id_a = a_request(store, prompt="list files", command="ls -a")
    time.sleep(0.01)
    id_b = a_request(store, prompt="delete the log files", command="rm *.log")
    time.sleep(0.01)
    a_request(store, prompt="show git log", command="git log")

    store.record_outcome(request_id=id_b, action="edit",
                         final_command="rm -f *.log", exit_code=0)

    out_path = tmp_path / "corpus.jsonl"
    count = store.export_jsonl(out_path)
    assert count == 3

    lines = out_path.read_text().splitlines()
    assert len(lines) == 3
    records = [json.loads(line) for line in lines]
    by_id = {r["id"]: r for r in records}

    assert by_id[id_a]["outcome"] is None
    assert by_id[id_b]["outcome"]["action"] == "edit"
    assert by_id[id_b]["outcome"]["final_command"] == "rm -f *.log"
    assert by_id[id_b]["command"] == "rm *.log"  # rejected, per the DPO pairing
    store.close()


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------

def test_concurrent_access_from_multiple_threads(tmp_path):
    """The real production shape: daemon.py serves every connection from a
    ThreadPoolExecutor(max_workers=4). This isn't proving SQLite itself is
    thread-safe -- it's regression coverage that every method touching
    self.db stays inside self._lock, now and in any future addition."""
    store = Store(make_config(tmp_path))
    workers, iterations = 8, 20
    errors = []
    all_ids = []
    ids_lock = threading.Lock()

    def worker(worker_index):
        try:
            for i in range(iterations):
                prompt = f"worker {worker_index} iteration {i}"
                assert store.lookup(prompt) is None
                request_id = a_request(store, prompt=prompt, command="ls")
                store.record_outcome(request_id=request_id, action="run",
                                     final_command="ls", exit_code=0)
                with ids_lock:
                    all_ids.append(request_id)
        except Exception as exc:  # pragma: no cover - re-raised below
            errors.append(exc)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(worker, w) for w in range(workers)]
        concurrent.futures.wait(futures)

    if errors:
        raise errors[0]

    assert len(all_ids) == workers * iterations
    assert len(set(all_ids)) == len(all_ids)

    conn = direct_connection(store.config)
    assert conn.execute("SELECT COUNT(*) AS n FROM requests").fetchone()["n"] == workers * iterations
    assert conn.execute("SELECT COUNT(*) AS n FROM outcomes").fetchone()["n"] == workers * iterations
    conn.close()
    store.close()
