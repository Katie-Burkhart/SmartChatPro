"""
Database operations for chat application
"""

import sqlite3
from datetime import datetime
from typing import List, Dict, Optional
import json

class ChatDatabase:
    """Manages SQLite database for chat sessions and messages"""

    def __init__(self, db_path: str = "chat_app.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Create tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                session_name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                total_tokens INTEGER DEFAULT 0
            )
        """)

        # Messages table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                tokens INTEGER DEFAULT 0,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)

        # Users table (for future authentication)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                email TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()

    def create_session(self, session_id: str, user_id: str, session_name: str):
        """Create a new chat session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO sessions (session_id, user_id, session_name)
            VALUES (?, ?, ?)
        """, (session_id, user_id, session_name))

        conn.commit()
        conn.close()

    def get_user_sessions(self, user_id: str) -> List[Dict]:
        """Get all sessions for a user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT session_id, session_name, created_at, updated_at, total_tokens
            FROM sessions
            WHERE user_id = ?
            ORDER BY updated_at DESC
        """, (user_id,))

        sessions = []
        for row in cursor.fetchall():
            sessions.append({
                "session_id": row[0],
                "session_name": row[1],
                "created_at": row[2],
                "updated_at": row[3],
                "total_tokens": row[4]
            })

        conn.close()
        return sessions

    def add_message(self, session_id: str, role: str, content: str, tokens: int = 0):
        """Add a message to a session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO messages (session_id, role, content, tokens)
            VALUES (?, ?, ?, ?)
        """, (session_id, role, content, tokens))

        # Update session's updated_at and total_tokens
        cursor.execute("""
            UPDATE sessions
            SET updated_at = CURRENT_TIMESTAMP,
                total_tokens = total_tokens + ?
            WHERE session_id = ?
        """, (tokens, session_id))

        conn.commit()
        conn.close()

    def get_session_messages(self, session_id: str) -> List[Dict]:
        """Get all messages for a session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT role, content, timestamp, tokens
            FROM messages
            WHERE session_id = ?
            ORDER BY timestamp ASC
        """, (session_id,))

        messages = []
        for row in cursor.fetchall():
            messages.append({
                "role": row[0],
                "content": row[1],
                "timestamp": row[2],
                "tokens": row[3]
            })

        conn.close()
        return messages

    def delete_session(self, session_id: str):
        """Delete a session and all its messages"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))

        conn.commit()
        conn.close()

    def rename_session(self, session_id: str, new_name: str):
        """Rename a session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE sessions
            SET session_name = ?, updated_at = CURRENT_TIMESTAMP
            WHERE session_id = ?
        """, (new_name, session_id))

        conn.commit()
        conn.close()