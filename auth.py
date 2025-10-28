"""
Authentication module
"""

import bcrypt
import sqlite3
import uuid
from typing import Optional, Tuple

class AuthManager:
    """Handles user authentication"""

    def __init__(self, db_path: str = "chat_app.db"):
        self.db_path = db_path

    def hash_password(self, password: str) -> str:
        """Hash a password using bcrypt"""
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')

    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify a password against its hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

    def create_user(self, username: str, password: str, email: Optional[str] = None) -> Tuple[bool, str]:
        """
        Create a new user
        Returns: (success, message or user_id)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check if username already exists
        cursor.execute("SELECT user_id FROM users WHERE username = ?", (username,))
        if cursor.fetchone():
            conn.close()
            return False, "Username already exists"

        # Create user
        user_id = str(uuid.uuid4())
        password_hash = self.hash_password(password)

        try:
            cursor.execute("""
                INSERT INTO users (user_id, username, password_hash, email)
                VALUES (?, ?, ?, ?)
            """, (user_id, username, password_hash, email))
            conn.commit()
            conn.close()
            return True, user_id
        except Exception as e:
            conn.close()
            return False, str(e)

    def authenticate_user(self, username: str, password: str) -> Tuple[bool, Optional[str]]:
        """
        Authenticate a user
        Returns: (success, user_id or None)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT user_id, password_hash
            FROM users
            WHERE username = ?
        """, (username,))

        result = cursor.fetchone()
        conn.close()

        if not result:
            return False, None

        user_id, password_hash = result

        if self.verify_password(password, password_hash):
            return True, user_id
        else:
            return False, None

    def get_username(self, user_id: str) -> Optional[str]:
        """Get username from user_id"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT username FROM users WHERE user_id = ?", (user_id,))
        result = cursor.fetchone()
        conn.close()

        return result[0] if result else None