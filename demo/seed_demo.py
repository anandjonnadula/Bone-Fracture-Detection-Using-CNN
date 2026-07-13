"""Demo-mode account seeding.

Creates (or re-keys) demo-patient / demo-doctor / demo-admin with fresh
random passwords on every boot. The plaintext passwords are returned so the
login page can display them — it's a public demo; the periodic wipe plus
per-boot rotation keeps stale credentials worthless. Clinical registration
keys are disabled while DEMO_MODE=1, so these are the only privileged
accounts that can exist.
"""

import secrets

from werkzeug.security import generate_password_hash

from db import get_db_connection

DEMO_USERS = [
    ("demo-patient", "patient"),
    ("demo-doctor", "doctor"),
    ("demo-admin", "admin"),
]


def seed_demo_accounts():
    accounts = []
    with get_db_connection() as conn:
        for username, role in DEMO_USERS:
            password = secrets.token_urlsafe(9)
            hashed = generate_password_hash(password)
            existing = conn.execute(
                "SELECT id FROM users WHERE username = ?", (username,)
            ).fetchone()
            if existing:
                conn.execute(
                    "UPDATE users SET password = ?, role = ? WHERE id = ?",
                    (hashed, role, existing["id"]),
                )
            else:
                conn.execute(
                    "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                    (username, hashed, role),
                )
            accounts.append({"username": username, "password": password, "role": role})
        conn.commit()
    return accounts
