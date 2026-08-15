"""SQLite response cache.

The predecessor engine wrote its cache key as sha256("claude-{MODEL}:{hash}") but
read it back by prompt hash alone. Changing model or effort therefore returned
results produced by the old configuration, silently and with no way to tell from
the output. Here one function computes the key, both paths call it, and every
input that can change an answer is inside it.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

_DDL = """
CREATE TABLE IF NOT EXISTS response_cache (
    cache_key    TEXT PRIMARY KEY,
    request_key  TEXT NOT NULL,
    model        TEXT NOT NULL,
    effort       TEXT NOT NULL,
    prompt_hash  TEXT NOT NULL,
    schema_hash  TEXT NOT NULL,
    policy_hash  TEXT NOT NULL,
    payload      TEXT NOT NULL,
    cost_usd     REAL NOT NULL DEFAULT 0,
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_response_cache_request ON response_cache(request_key);
"""


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_cache_key(
    *,
    prompt: str,
    schema: dict[str, Any],
    model: str,
    effort: str,
    policy_version: str,
    system: str = "",
    sample_index: int = 0,
) -> str:
    """The single definition of cache identity, used by both get and put.

    Anything that can change the answer belongs here. A prompt hash alone is not
    an identity.

    `sample_index` is what makes repeated sampling possible. Everything else in
    the key is fixed across the samples of one judgment -- that is the point of a
    repeat -- so without it the second sample would hit the first one's row and
    return the same answer, and N samples of a noisy judge would cost N calls and
    carry one call's worth of information. Separating the keys makes each sample a
    real call to a sampling model, while leaving every sample individually
    cacheable, so an interrupted run resumes rather than re-drawing.

    It is folded in only when nonzero, so sample 0's key is byte-identical to the
    key this function produced before sampling existed. That is not cosmetic: it
    keeps roughly $95 of already-paid-for responses readable instead of orphaning
    the whole cache the moment this landed.
    """
    material_dict: dict[str, Any] = {
        "v": SCHEMA_VERSION,
        "system": system,
        "prompt": prompt,
        "schema": schema,
        "model": model,
        "effort": effort,
        "policy": policy_version,
    }
    if sample_index:
        material_dict["sample"] = sample_index
    material = json.dumps(material_dict, sort_keys=True, separators=(",", ":"))
    return _sha(material)


@dataclass
class CacheEntry:
    payload: dict
    cost_usd: float
    created_at: str


class ResponseCache:
    """Thread-safe cache. Never raises into the caller: a cache problem must not
    fail a run, it should just cost another model call."""

    def __init__(self, path: str | Path | None):
        self.path = str(path) if path else None
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        self.hits = 0
        self.misses = 0
        if self.path:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
            self._conn.row_factory = sqlite3.Row
            with self._lock:
                self._conn.executescript(_DDL)
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._conn.commit()

    @property
    def enabled(self) -> bool:
        return self._conn is not None

    def get(self, cache_key: str) -> CacheEntry | None:
        if not self._conn:
            return None
        try:
            with self._lock:
                row = self._conn.execute(
                    "SELECT payload, cost_usd, created_at FROM response_cache "
                    "WHERE cache_key = ?",
                    (cache_key,),
                ).fetchone()
        except Exception:
            return None
        if not row:
            self.misses += 1
            return None
        try:
            entry = CacheEntry(
                payload=json.loads(row["payload"]),
                cost_usd=row["cost_usd"],
                created_at=row["created_at"],
            )
        except Exception:
            self.misses += 1
            return None
        self.hits += 1
        return entry

    def put(
        self,
        cache_key: str,
        payload: dict,
        *,
        request_key: str,
        model: str,
        effort: str,
        prompt: str,
        schema: dict,
        policy_version: str,
        cost_usd: float = 0.0,
    ) -> None:
        if not self._conn:
            return
        row = (
            cache_key,
            request_key,
            model,
            effort,
            _sha(prompt),
            _sha(json.dumps(schema, sort_keys=True)),
            _sha(policy_version),
            json.dumps(payload, separators=(",", ":")),
            cost_usd,
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        for attempt in range(5):
            try:
                with self._lock:
                    self._conn.execute(
                        """INSERT INTO response_cache
                           (cache_key, request_key, model, effort, prompt_hash,
                            schema_hash, policy_hash, payload, cost_usd, created_at)
                           VALUES (?,?,?,?,?,?,?,?,?,?)
                           ON CONFLICT(cache_key) DO UPDATE SET
                             payload=excluded.payload,
                             cost_usd=excluded.cost_usd,
                             created_at=excluded.created_at""",
                        row,
                    )
                    self._conn.commit()
                return
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower() or attempt == 4:
                    return
                time.sleep(0.1 * (2**attempt))
            except Exception:
                return

    def close(self) -> None:
        if self._conn:
            with self._lock:
                self._conn.close()
            self._conn = None

    def stats(self) -> dict:
        return {"enabled": self.enabled, "hits": self.hits, "misses": self.misses}
