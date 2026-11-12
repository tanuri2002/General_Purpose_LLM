"""
storage.py – SQLite persistence layer for strategy-comparison runs.

Uses only the Python standard library (sqlite3).  The database file
``results.db`` is created automatically next to this module on first use.

Schema
------
runs
    id              INTEGER  PRIMARY KEY AUTOINCREMENT
    created_at      TEXT     ISO-8601 timestamp (UTC)
    task            TEXT     The raw user task
    strategy        TEXT     Strategy identifier, e.g. "zero_shot"
    system_prompt   TEXT     System prompt sent (NULL when none)
    user_message    TEXT     Full user message sent to the model
    reply           TEXT     Model reply (NULL on error)
    error           TEXT     Error message (NULL on success)
    temperature     REAL     Sampling temperature
    latency_seconds REAL     API round-trip time in seconds (NULL on error)
    prompt_tokens   INTEGER  (NULL on error)
    output_tokens   INTEGER  (NULL on error)
    total_tokens    INTEGER  (NULL on error)
    rating          INTEGER  1-5 star rating set by the user (NULL until rated)
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Database location — same directory as this file
# ---------------------------------------------------------------------------
DB_PATH = Path(__file__).resolve().parent / "results.db"


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------

def _connect() -> sqlite3.Connection:
    """Return a connection with foreign-key support and row_factory set."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row          # rows accessible as dicts
    conn.execute("PRAGMA journal_mode=WAL")  # safe for concurrent reads
    return conn


# ---------------------------------------------------------------------------
# Schema initialisation – called once at server startup
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Create the ``runs`` table if it doesn't already exist."""
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                id              INTEGER  PRIMARY KEY AUTOINCREMENT,
                created_at      TEXT     NOT NULL,
                task            TEXT     NOT NULL,
                strategy        TEXT     NOT NULL,
                system_prompt   TEXT,
                user_message    TEXT     NOT NULL,
                reply           TEXT,
                error           TEXT,
                temperature     REAL,
                latency_seconds REAL,
                prompt_tokens   INTEGER,
                output_tokens   INTEGER,
                total_tokens    INTEGER,
                rating          INTEGER  CHECK(rating IS NULL OR (rating >= 1 AND rating <= 5))
            )
        """)
        conn.commit()


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------

def save_run(
    task: str,
    strategy: str,
    system_prompt: Optional[str],
    user_message: str,
    reply: Optional[str],
    error: Optional[str],
    temperature: float,
    latency_seconds: Optional[float],
    prompt_tokens: Optional[int],
    output_tokens: Optional[int],
    total_tokens: Optional[int],
) -> int:
    """Insert one strategy result and return the new ``id``."""
    created_at = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO runs
                (created_at, task, strategy, system_prompt, user_message,
                 reply, error, temperature, latency_seconds,
                 prompt_tokens, output_tokens, total_tokens, rating)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,NULL)
            """,
            (
                created_at, task, strategy, system_prompt, user_message,
                reply, error, temperature, latency_seconds,
                prompt_tokens, output_tokens, total_tokens,
            ),
        )
        conn.commit()
        return cur.lastrowid


def set_rating(run_id: int, rating: int) -> bool:
    """Update the rating for a run.  Returns True if a row was updated."""
    if not (1 <= rating <= 5):
        raise ValueError(f"Rating must be 1-5, got {rating!r}")
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE runs SET rating = ? WHERE id = ?",
            (rating, run_id),
        )
        conn.commit()
        return cur.rowcount > 0


# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------

def get_history(limit: int = 50) -> list[dict]:
    """Return the most recent *limit* runs, newest first."""
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, created_at, task, strategy,
                   system_prompt, user_message,
                   reply, error,
                   temperature, latency_seconds,
                   prompt_tokens, output_tokens, total_tokens,
                   rating
            FROM runs
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_summary() -> list[dict]:
    """Return aggregated benchmark metrics grouped by strategy."""
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT
                strategy,
                COUNT(*) AS total_runs,
                ROUND(AVG(rating), 2) AS avg_rating,
                ROUND(AVG(latency_seconds), 3) AS avg_latency,
                ROUND(AVG(total_tokens), 1) AS avg_total_tokens
            FROM runs
            GROUP BY strategy
            """
        ).fetchall()
    return [dict(row) for row in rows]


