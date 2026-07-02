import os
from datetime import datetime, timezone

import psycopg2
from psycopg2.extras import Json
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not set.")


def _connect(dsn: str = DATABASE_URL):
    return psycopg2.connect(dsn)


def init_db(dsn: str = DATABASE_URL) -> None:
    conn = _connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    id SERIAL PRIMARY KEY,
                    kind TEXT NOT NULL CHECK (kind IN ('job', 'cv')),
                    name TEXT NOT NULL,
                    source_text TEXT NOT NULL,
                    vector JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL
                )
                """
            )
        conn.commit()
    finally:
        conn.close()


def save_embedding(kind: str, name: str, source_text: str, vector: list[float], dsn: str = DATABASE_URL) -> int:
    conn = _connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO embeddings (kind, name, source_text, vector, created_at)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """,
                (kind, name, source_text, Json(vector), datetime.now(timezone.utc)),
            )
            row_id = cur.fetchone()[0]
        conn.commit()
        return row_id
    finally:
        conn.close()


def get_embedding(row_id: int, dsn: str = DATABASE_URL) -> list[float] | None:
    conn = _connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT vector FROM embeddings WHERE id = %s", (row_id,))
            row = cur.fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def delete_embedding(row_id: int, dsn: str = DATABASE_URL) -> None:
    conn = _connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM embeddings WHERE id = %s", (row_id,))
        conn.commit()
    finally:
        conn.close()
