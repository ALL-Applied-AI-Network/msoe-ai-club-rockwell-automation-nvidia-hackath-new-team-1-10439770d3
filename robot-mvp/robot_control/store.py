"""SQLite-backed plan handoff between MCP, the human CLI, and the policy process."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import uuid

from .safety import validate_path


def default_db_path() -> Path:
    configured = os.environ.get("ATEAM_ROBOT_DB")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[1] / "runtime" / "robot_plans.sqlite3"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class PlanStore:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else default_db_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS plans (
                    id TEXT PRIMARY KEY,
                    label TEXT NOT NULL,
                    waypoints_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    execution_mode TEXT,
                    approved_by TEXT,
                    created_at TEXT NOT NULL,
                    approved_at TEXT,
                    started_at TEXT,
                    completed_at TEXT,
                    note TEXT
                )
                """
            )

    @staticmethod
    def _row(row: sqlite3.Row | None) -> dict | None:
        if row is None:
            return None
        item = dict(row)
        item["waypoints"] = json.loads(item.pop("waypoints_json"))
        return item

    def create(self, label: str, waypoints: list[dict]) -> dict:
        clean = validate_path(waypoints)
        plan_id = uuid.uuid4().hex[:12]
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO plans (id, label, waypoints_json, status, created_at) VALUES (?, ?, ?, 'pending', ?)",
                (plan_id, label.strip() or "point-to-point move", json.dumps(clean), utc_now()),
            )
        return self.get(plan_id)

    def get(self, plan_id: str) -> dict:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM plans WHERE id = ?", (plan_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown plan {plan_id}")
        return self._row(row)

    def list(self, limit: int = 20) -> list[dict]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM plans ORDER BY created_at DESC LIMIT ?", (max(1, min(limit, 100)),)
            ).fetchall()
        return [self._row(row) for row in rows]

    def approve(self, plan_id: str, execution_mode: str, approved_by: str) -> dict:
        if execution_mode not in {"mock", "physical"}:
            raise ValueError("execution_mode must be mock or physical")
        with self.connect() as connection:
            plan = connection.execute("SELECT status FROM plans WHERE id = ?", (plan_id,)).fetchone()
            if plan is None:
                raise KeyError(f"unknown plan {plan_id}")
            if plan["status"] != "pending":
                raise ValueError(f"plan {plan_id} is {plan['status']}, not pending")
            connection.execute(
                "UPDATE plans SET status='approved', execution_mode=?, approved_by=?, approved_at=? WHERE id=?",
                (execution_mode, approved_by.strip() or "local operator", utc_now(), plan_id),
            )
        return self.get(plan_id)

    def claim_next(self, execution_mode: str) -> dict | None:
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM plans WHERE status='approved' AND execution_mode=? "
                "ORDER BY approved_at ASC LIMIT 1",
                (execution_mode,),
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                "UPDATE plans SET status='executing', started_at=? WHERE id=?",
                (utc_now(), row["id"]),
            )
        return self.get(row["id"])

    def complete(self, plan_id: str, note: str = "") -> dict:
        with self.connect() as connection:
            connection.execute(
                "UPDATE plans SET status='completed', completed_at=?, note=? "
                "WHERE id=? AND status='executing'",
                (utc_now(), note, plan_id),
            )
        return self.get(plan_id)

    def cancel(self, plan_id: str, note: str = "cancelled before execution") -> dict:
        with self.connect() as connection:
            changed = connection.execute(
                "UPDATE plans SET status='cancelled', completed_at=?, note=? "
                "WHERE id=? AND status IN ('pending', 'approved')",
                (utc_now(), note, plan_id),
            ).rowcount
        if not changed:
            current = self.get(plan_id)
            raise ValueError(f"plan {plan_id} cannot be cancelled while {current['status']}")
        return self.get(plan_id)

