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
            cur.execute("DROP TABLE IF EXISTS embeddings")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS job_postings (
                    id SERIAL PRIMARY KEY,
                    position TEXT NOT NULL,
                    description TEXT,
                    requirements TEXT,
                    vector JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS candidates (
                    id SERIAL PRIMARY KEY,
                    file_name TEXT NOT NULL,
                    ad_soyad TEXT,
                    e_posta TEXT,
                    telefon TEXT,
                    deneyim_yili NUMERIC,
                    yetenekler JSONB,
                    egitim JSONB,
                    ozet TEXT,
                    source_text TEXT NOT NULL,
                    vector JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL
                )
                """
            )
        conn.commit()
    finally:
        conn.close()


def save_job_posting(
    position: str,
    description: str,
    requirements: str,
    vector: list[float],
    dsn: str = DATABASE_URL,
) -> int:
    conn = _connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO job_postings (position, description, requirements, vector, created_at)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """,
                (position, description, requirements, Json(vector), datetime.now(timezone.utc)),
            )
            row_id = cur.fetchone()[0]
        conn.commit()
        return row_id
    finally:
        conn.close()


def get_job_posting(row_id: int, dsn: str = DATABASE_URL) -> dict | None:
    conn = _connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, position, description, requirements, vector, created_at "
                "FROM job_postings WHERE id = %s",
                (row_id,),
            )
            row = cur.fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "position": row[1],
            "description": row[2],
            "requirements": row[3],
            "vector": row[4],
            "created_at": row[5],
        }
    finally:
        conn.close()


def delete_job_posting(row_id: int, dsn: str = DATABASE_URL) -> None:
    conn = _connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM job_postings WHERE id = %s", (row_id,))
        conn.commit()
    finally:
        conn.close()


def list_job_postings(dsn: str = DATABASE_URL) -> list[dict]:
    conn = _connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, position, description, requirements, created_at "
                "FROM job_postings ORDER BY created_at DESC"
            )
            rows = cur.fetchall()
        return [
            {
                "id": row[0],
                "position": row[1],
                "description": row[2],
                "requirements": row[3],
                "created_at": row[4],
            }
            for row in rows
        ]
    finally:
        conn.close()


def save_candidate(
    file_name: str,
    cv_data: dict,
    source_text: str,
    vector: list[float],
    dsn: str = DATABASE_URL,
) -> int:
    iletisim = cv_data.get("iletisim") or {}
    conn = _connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO candidates (
                    file_name, ad_soyad, e_posta, telefon, deneyim_yili,
                    yetenekler, egitim, ozet, source_text, vector, created_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    file_name,
                    cv_data.get("ad_soyad"),
                    iletisim.get("e_posta"),
                    iletisim.get("telefon"),
                    cv_data.get("deneyim_yili"),
                    Json(cv_data.get("yetenekler") or []),
                    Json(cv_data.get("egitim") or []),
                    cv_data.get("ozet"),
                    source_text,
                    Json(vector),
                    datetime.now(timezone.utc),
                ),
            )
            row_id = cur.fetchone()[0]
        conn.commit()
        return row_id
    finally:
        conn.close()


def get_candidate(row_id: int, dsn: str = DATABASE_URL) -> dict | None:
    conn = _connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, file_name, ad_soyad, e_posta, telefon, deneyim_yili, "
                "yetenekler, egitim, ozet, source_text, vector, created_at "
                "FROM candidates WHERE id = %s",
                (row_id,),
            )
            row = cur.fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "file_name": row[1],
            "ad_soyad": row[2],
            "e_posta": row[3],
            "telefon": row[4],
            "deneyim_yili": row[5],
            "yetenekler": row[6],
            "egitim": row[7],
            "ozet": row[8],
            "source_text": row[9],
            "vector": row[10],
            "created_at": row[11],
        }
    finally:
        conn.close()


def delete_candidate(row_id: int, dsn: str = DATABASE_URL) -> None:
    conn = _connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM candidates WHERE id = %s", (row_id,))
        conn.commit()
    finally:
        conn.close()


def list_candidates(dsn: str = DATABASE_URL) -> list[dict]:
    conn = _connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, file_name, ad_soyad, e_posta, telefon, deneyim_yili, "
                "yetenekler, egitim, ozet, created_at "
                "FROM candidates ORDER BY created_at DESC"
            )
            rows = cur.fetchall()
        return [
            {
                "id": row[0],
                "file_name": row[1],
                "ad_soyad": row[2],
                "e_posta": row[3],
                "telefon": row[4],
                "deneyim_yili": row[5],
                "yetenekler": row[6],
                "egitim": row[7],
                "ozet": row[8],
                "created_at": row[9],
            }
            for row in rows
        ]
    finally:
        conn.close()
