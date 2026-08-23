"""Provider CRUD HTTP routes (Settings -> AI Providers).

- GET    /api/providers          → list all providers
- POST   /api/providers          → create provider
- GET    /api/providers/{id}     → get provider by ID
- PUT    /api/providers/{id}     → update provider
- DELETE /api/providers/{id}     → delete provider
- POST   /api/providers/{id}/test     → test connectivity
- POST   /api/providers/{id}/enable   → enable/disable
- POST   /api/providers/{id}/default  → set as default
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from src.core.db import _get_conn, _utcnow

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/providers", tags=["providers"])


class ProviderOut(BaseModel):
    id: str
    name: str
    provider_type: str
    provider_kind: str
    enabled: bool = True
    base_url: str | None = None
    model: str | None = None
    config: dict[str, Any] = {}
    capabilities: list[str] = []
    last_test_status: str | None = None
    last_test_at: str | None = None
    created_at: str = ""
    updated_at: str = ""
    needs_key: bool = False
    api_key_configured: bool = False


class ProviderCreateInput(BaseModel):
    name: str = Field(min_length=1)
    provider_type: str = "translation"
    provider_kind: str = "free"
    capabilities: list[str] = ["translation"]
    base_url: str | None = None
    model: str | None = None
    config: dict[str, Any] = {}


class DefaultsOut(BaseModel):
    providers: list[ProviderOut]
    defaults: dict[str, str] = {}


def _row_to_provider(row: Any) -> dict[str, Any]:
    import json
    config = {}
    if row.get("config_json"):
        try:
            config = json.loads(row["config_json"])
        except Exception:
            pass
    caps = []
    if row.get("capabilities"):
        try:
            caps = json.loads(row["capabilities"]) if isinstance(row["capabilities"], str) else row["capabilities"]
        except Exception:
            caps = []
    return {
        "id": row["id"],
        "name": row["name"],
        "provider_type": row.get("provider_type", "translation"),
        "provider_kind": row.get("provider_kind", "free"),
        "enabled": bool(row.get("enabled", 1)),
        "base_url": row.get("base_url"),
        "model": row.get("model"),
        "config": config,
        "capabilities": caps,
        "last_test_status": row.get("last_test_status"),
        "last_test_at": row.get("last_test_at"),
        "created_at": row.get("created_at", ""),
        "updated_at": row.get("updated_at", ""),
        "needs_key": row.get("provider_kind") in ("gemini", "openai"),
        "api_key_configured": False,
    }


def _ensure_table():
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS providers (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            provider_type TEXT NOT NULL DEFAULT 'translation',
            provider_kind TEXT NOT NULL DEFAULT 'free',
            enabled INTEGER NOT NULL DEFAULT 1,
            base_url TEXT,
            model TEXT,
            config_json TEXT DEFAULT '{}',
            capabilities TEXT DEFAULT '[]',
            last_test_status TEXT,
            last_test_at TEXT,
            is_default INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    conn.commit()
    # Migration: add is_default if missing (Rust schema doesn't have it)
    existing_cols = {r[1] for r in conn.execute('PRAGMA table_info(providers)').fetchall()}
    if 'is_default' not in existing_cols:
        conn.execute('ALTER TABLE providers ADD COLUMN is_default INTEGER DEFAULT 0')
        conn.commit()
    # Seed built-in providers if table is empty
    count = conn.execute("SELECT COUNT(*) FROM providers").fetchone()[0]
    if count == 0:
        now = _utcnow()
        for p in [
            ("free", "FREE", "translation", "free", "http://127.0.0.1:8080", None, '["translation","stt"]'),
            ("gemini", "Gemini (cloud)", "translation", "gemini", None, "gemini-flash-lite-latest", '["translation"]'),
            ("local", "Local LLM", "translation", "local", "http://127.0.0.1:8080", None, '["translation"]'),
        ]:
            conn.execute(
                "INSERT OR IGNORE INTO providers (id, name, provider_type, provider_kind, base_url, model, capabilities, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (*p, now, now),
            )
        conn.commit()


@router.get("")
def list_providers() -> dict:
    _ensure_table()
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM providers ORDER BY is_default DESC, name").fetchall()
    providers = [_row_to_provider(dict(r)) for r in rows]
    defaults: dict[str, str] = {}
    for r in rows:
        d = dict(r)
        if d.get("is_default"):
            for cap in d.get("capabilities", "[]"):
                pass
    # Compute defaults from is_default flag
    for r in rows:
        d = dict(r)
        if d.get("is_default"):
            caps = []
            try:
                import json
                caps = json.loads(d["capabilities"]) if isinstance(d["capabilities"], str) else d["capabilities"]
            except Exception:
                pass
            for cap in caps:
                defaults[cap] = d["id"]
    return {"providers": providers, "defaults": defaults}


@router.post("", status_code=status.HTTP_201_CREATED)
def create_provider(input: ProviderCreateInput) -> dict:
    import uuid, json
    _ensure_table()
    conn = _get_conn()
    now = _utcnow()
    pid = str(uuid.uuid4())[:8]
    conn.execute(
        "INSERT INTO providers (id, name, provider_type, provider_kind, base_url, model, config_json, capabilities, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (pid, input.name, input.provider_type, input.provider_kind, input.base_url, input.model,
         json.dumps(input.config), json.dumps(input.capabilities), now, now),
    )
    conn.commit()
    row = conn.execute("SELECT * FROM providers WHERE id = ?", (pid,)).fetchone()
    return _row_to_provider(dict(row))


@router.get("/{provider_id}")
def get_provider(provider_id: str) -> dict:
    _ensure_table()
    conn = _get_conn()
    row = conn.execute("SELECT * FROM providers WHERE id = ?", (provider_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"Provider {provider_id!r} not found")
    return _row_to_provider(dict(row))


@router.put("/{provider_id}")
def update_provider(provider_id: str, input: ProviderCreateInput) -> dict:
    import json
    _ensure_table()
    conn = _get_conn()
    now = _utcnow()
    conn.execute(
        "UPDATE providers SET name=?, provider_type=?, provider_kind=?, base_url=?, model=?, config_json=?, capabilities=?, updated_at=? WHERE id=?",
        (input.name, input.provider_type, input.provider_kind, input.base_url, input.model,
         json.dumps(input.config), json.dumps(input.capabilities), now, provider_id),
    )
    conn.commit()
    row = conn.execute("SELECT * FROM providers WHERE id = ?", (provider_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"Provider {provider_id!r} not found")
    return _row_to_provider(dict(row))


@router.delete("/{provider_id}")
def delete_provider(provider_id: str) -> dict:
    _ensure_table()
    conn = _get_conn()
    conn.execute("DELETE FROM providers WHERE id = ?", (provider_id,))
    conn.commit()
    return {"deleted": True, "id": provider_id}


@router.post("/{provider_id}/test")
def test_provider(provider_id: str) -> dict:
    return {"ok": True, "latency_ms": 0, "detail": "Test endpoint placeholder"}
