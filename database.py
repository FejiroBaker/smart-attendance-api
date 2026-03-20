# python_backend/database.py

import sqlite3
import json
import os
from typing import Optional, List
from models import StoredEmbedding

DB_PATH = "face_embeddings.db"


def init_db():
    """Initialize the database and create tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS face_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL UNIQUE,
            student_name TEXT NOT NULL,
            embedding TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.commit()
    conn.close()
    print("Database initialized successfully")


def save_embedding(student_id: int, student_name: str, embedding: List[float]) -> bool:
    """Save or update a face embedding for a student."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        embedding_json = json.dumps(embedding)

        # Insert or replace (update if student_id already exists)
        cursor.execute("""
            INSERT INTO face_embeddings (student_id, student_name, embedding, updated_at)
            VALUES (?, ?, ?, datetime('now'))
            ON CONFLICT(student_id) DO UPDATE SET
                student_name = excluded.student_name,
                embedding = excluded.embedding,
                updated_at = datetime('now')
        """, (student_id, student_name, embedding_json))

        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error saving embedding: {e}")
        return False


def get_embedding(student_id: int) -> Optional[StoredEmbedding]:
    """Get face embedding for a specific student."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT student_id, student_name, embedding FROM face_embeddings WHERE student_id = ?",
            (student_id,)
        )
        row = cursor.fetchone()
        conn.close()

        if row:
            return StoredEmbedding(
                student_id=row[0],
                student_name=row[1],
                embedding=json.loads(row[2])
            )
        return None
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None


def get_all_embeddings() -> List[StoredEmbedding]:
    """Get all stored face embeddings."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT student_id, student_name, embedding FROM face_embeddings"
        )
        rows = cursor.fetchall()
        conn.close()

        return [
            StoredEmbedding(
                student_id=row[0],
                student_name=row[1],
                embedding=json.loads(row[2])
            )
            for row in rows
        ]
    except Exception as e:
        print(f"Error getting all embeddings: {e}")
        return []


def delete_embedding(student_id: int) -> bool:
    """Delete a student's face embedding."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM face_embeddings WHERE student_id = ?",
            (student_id,)
        )
        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return deleted
    except Exception as e:
        print(f"Error deleting embedding: {e}")
        return False


def get_registered_count() -> int:
    """Get total number of registered faces."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM face_embeddings")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    except Exception as e:
        print(f"Error getting count: {e}")
        return 0
