"""SQLite persistence + atomic config write-back (HUB_SPEC §4, §6).

The DDL is copied from HUB_SPEC §6 unchanged. Everything that mutates
configuration goes through :meth:`Storage.put_rules` / :meth:`Storage.put_hub_config`
so the three guarantees in §4 hold on every path:

1. write on every PUT, no delayed batching;
2. atomic file replace (``<file>.tmp`` + ``fsync`` + ``rename``, plus an fsync of
   the containing directory so the rename itself is durable);
3. ``rev`` bump + a ``config_versions`` row in the same SQLite transaction.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS alerts (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  event_id      TEXT NOT NULL UNIQUE,
  ts_ms         INTEGER NOT NULL,
  received_ms   INTEGER NOT NULL,
  device_id     TEXT NOT NULL,
  stream_id     TEXT NOT NULL,
  event_type    TEXT NOT NULL CHECK(event_type IN ('zone_enter','loitering','line_cross')),
  rule_name     TEXT NOT NULL,
  track_id      INTEGER NOT NULL,
  score         REAL,
  bbox          TEXT NOT NULL,
  direction     TEXT,
  dwell_s       REAL,
  state         TEXT NOT NULL DEFAULT 'new' CHECK(state IN ('new','acked','dismissed')),
  acted_by      TEXT,
  acted_at      INTEGER,
  snapshot_state TEXT NOT NULL DEFAULT 'none' CHECK(snapshot_state IN ('pending','received','timeout','none')),
  snapshot_path TEXT,
  meta          TEXT
);
CREATE INDEX IF NOT EXISTS idx_alerts_ts ON alerts(ts_ms DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_state ON alerts(state, ts_ms DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_scope ON alerts(device_id, stream_id, ts_ms DESC);

CREATE TABLE IF NOT EXISTS devices (
  device_id    TEXT PRIMARY KEY,
  online       INTEGER NOT NULL DEFAULT 0,
  last_seen_ms INTEGER,
  mode         TEXT NOT NULL DEFAULT 'hub' CHECK(mode IN ('hub','single_box')),
  info         TEXT
);

CREATE TABLE IF NOT EXISTS rules (
  device_id  TEXT NOT NULL,
  stream_id  TEXT NOT NULL,
  rev        INTEGER NOT NULL,
  body       TEXT NOT NULL,
  updated_ms INTEGER NOT NULL,
  PRIMARY KEY (device_id, stream_id)
);

CREATE TABLE IF NOT EXISTS config_versions (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  scope      TEXT NOT NULL,
  rev        INTEGER NOT NULL,
  body       TEXT NOT NULL,
  created_ms INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS auth (
  username      TEXT PRIMARY KEY,
  password_hash TEXT NOT NULL,
  must_change   INTEGER NOT NULL DEFAULT 1
);
"""

#: HUB_SPEC §4: config_versions keeps the most recent 50 revisions per scope.
CONFIG_VERSION_KEEP = 50

#: HUB_SPEC §3 alert state machine. `new` is never a destination.
LEGAL_TRANSITIONS = {
    "acked": {"new", "dismissed"},
    "dismissed": {"new", "acked"},
}

ALERT_COLUMNS = (
    "id, event_id, ts_ms, received_ms, device_id, stream_id, event_type, rule_name, "
    "track_id, score, bbox, direction, dwell_s, state, acted_by, acted_at, "
    "snapshot_state, snapshot_path, meta"
)


def atomic_write_json(path: Path, payload: Any) -> None:
    """Write JSON to ``path`` atomically: tmp + fsync + rename + dir fsync."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    data = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    dir_fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def alert_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    """Serialize an alerts row for REST / WS (FRONTEND_SPEC §7 shape).

    The device timestamp is exposed under one name only: ``ts_ms``, the column
    name from the §6 DDL. A second ``ts`` alias used to sit here, and it forced
    every consumer to accept either spelling — the frontend carried an
    ``a.ts || a.ts_ms`` branch for exactly that reason.
    """
    meta = json.loads(row["meta"]) if row["meta"] else {}
    return {
        "id": row["id"],
        "event_id": row["event_id"],
        "ts_ms": row["ts_ms"],
        "received_ms": row["received_ms"],
        "device_id": row["device_id"],
        "stream_id": row["stream_id"],
        "event_type": row["event_type"],
        "rule_name": row["rule_name"],
        "track_id": row["track_id"],
        "score": row["score"],
        "bbox": json.loads(row["bbox"]),
        "direction": row["direction"],
        "dwell_s": row["dwell_s"],
        "state": row["state"],
        "acted_by": row["acted_by"],
        "acted_at": row["acted_at"],
        "snapshot_state": row["snapshot_state"],
        "snapshot_url": (
            f"/api/alerts/{row['id']}/snapshot.jpg" if row["snapshot_path"] else None
        ),
        "simulated": bool(meta.get("simulated")),
        "meta": meta,
    }


class Storage:
    """Owns the SQLite file, the snapshot directory and the config.json mirror."""

    def __init__(self, data_dir: str | os.PathLike[str]) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "hub.db"
        self.config_path = self.data_dir / "config.json"
        self.snapshot_dir = self.data_dir / "snapshots"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(SCHEMA_SQL)

    def close(self) -> None:
        self.conn.close()

    # -- alerts ----------------------------------------------------------
    def insert_alert(self, **fields: Any) -> dict[str, Any]:
        cur = self.conn.execute(
            """INSERT INTO alerts (event_id, ts_ms, received_ms, device_id, stream_id,
                    event_type, rule_name, track_id, score, bbox, direction, dwell_s,
                    snapshot_state, meta)
               VALUES (:event_id, :ts_ms, :received_ms, :device_id, :stream_id,
                    :event_type, :rule_name, :track_id, :score, :bbox, :direction,
                    :dwell_s, :snapshot_state, :meta)""",
            {
                "event_id": fields["event_id"],
                "ts_ms": fields["ts_ms"],
                "received_ms": fields["received_ms"],
                "device_id": fields["device_id"],
                "stream_id": fields["stream_id"],
                "event_type": fields["event_type"],
                "rule_name": fields["rule_name"],
                "track_id": fields["track_id"],
                "score": fields.get("score"),
                "bbox": json.dumps(fields.get("bbox") or []),
                "direction": fields.get("direction"),
                "dwell_s": fields.get("dwell_s"),
                "snapshot_state": fields.get("snapshot_state", "none"),
                "meta": json.dumps(fields.get("meta") or {}),
            },
        )
        return self.get_alert(int(cur.lastrowid))  # type: ignore[arg-type]

    def get_alert(self, alert_id: int) -> dict[str, Any] | None:
        row = self.conn.execute(
            f"SELECT {ALERT_COLUMNS} FROM alerts WHERE id = ?", (alert_id,)
        ).fetchone()
        return alert_row_to_dict(row) if row else None

    def get_alert_by_event_id(self, event_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            f"SELECT {ALERT_COLUMNS} FROM alerts WHERE event_id = ?", (event_id,)
        ).fetchone()
        return alert_row_to_dict(row) if row else None

    def alert_snapshot_path(self, alert_id: int) -> str | None:
        row = self.conn.execute(
            "SELECT snapshot_path FROM alerts WHERE id = ?", (alert_id,)
        ).fetchone()
        return row["snapshot_path"] if row else None

    def query_alerts(
        self,
        state: str | None = None,
        device_id: str | None = None,
        stream_id: str | None = None,
        event_type: str | None = None,
        rule_name: str | None = None,
        date_from: int | None = None,
        date_to: int | None = None,
        after_id: int | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """HUB_SPEC §4 /alerts.

        Default order is ``ts_ms DESC``. With ``after_id`` the order flips to
        ``id ASC`` over rows with a larger id, which is what the WS reconnect
        gap-fill in §3/§5 relies on.
        """
        where: list[str] = []
        args: list[Any] = []
        if state:
            where.append("state = ?")
            args.append(state)
        if device_id:
            where.append("device_id = ?")
            args.append(device_id)
        if stream_id:
            where.append("stream_id = ?")
            args.append(stream_id)
        if event_type:
            where.append("event_type = ?")
            args.append(event_type)
        if rule_name:
            where.append("rule_name = ?")
            args.append(rule_name)
        if date_from is not None:
            where.append("ts_ms >= ?")
            args.append(date_from)
        if date_to is not None:
            where.append("ts_ms <= ?")
            args.append(date_to)
        if after_id is not None:
            where.append("id > ?")
            args.append(after_id)
        order = "id ASC" if after_id is not None else "ts_ms DESC, id DESC"
        sql = f"SELECT {ALERT_COLUMNS} FROM alerts"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += f" ORDER BY {order} LIMIT ? OFFSET ?"
        args += [int(limit), int(offset)]
        return [alert_row_to_dict(r) for r in self.conn.execute(sql, args)]

    def count_alerts(self, **filters: Any) -> int:
        rows = self.query_alerts(limit=1_000_000, **filters)
        return len(rows)

    def transition_alert(
        self, alert_id: int, target: str, actor: str, now_ms: int
    ) -> tuple[str, dict[str, Any] | None]:
        """Apply the §3 state machine.

        Returns ``("ok"|"conflict"|"not_found", alert_or_None)``. An idempotent
        repeat of the current state is a conflict: the only legal sources are
        listed in :data:`LEGAL_TRANSITIONS`, and ``acked -> acked`` is not one.
        """
        row = self.conn.execute(
            "SELECT state FROM alerts WHERE id = ?", (alert_id,)
        ).fetchone()
        if row is None:
            return "not_found", None
        if row["state"] not in LEGAL_TRANSITIONS.get(target, set()):
            return "conflict", self.get_alert(alert_id)
        self.conn.execute(
            "UPDATE alerts SET state = ?, acted_by = ?, acted_at = ? WHERE id = ?",
            (target, actor, now_ms, alert_id),
        )
        return "ok", self.get_alert(alert_id)

    def set_snapshot(self, alert_id: int, path: str, state: str = "received") -> dict[str, Any] | None:
        self.conn.execute(
            "UPDATE alerts SET snapshot_path = ?, snapshot_state = ? WHERE id = ?",
            (path, state, alert_id),
        )
        return self.get_alert(alert_id)

    def set_snapshot_state(self, alert_id: int, state: str) -> dict[str, Any] | None:
        self.conn.execute(
            "UPDATE alerts SET snapshot_state = ? WHERE id = ?", (state, alert_id)
        )
        return self.get_alert(alert_id)

    def pending_snapshot_alerts(self) -> list[dict[str, Any]]:
        """Rows still awaiting a snapshot — re-armed on restart (§3)."""
        return [
            alert_row_to_dict(r)
            for r in self.conn.execute(
                f"SELECT {ALERT_COLUMNS} FROM alerts WHERE snapshot_state = 'pending'"
                " ORDER BY id ASC"
            )
        ]

    def max_event_seq(self, device_id: str, stream_id: str, session_id: str) -> int:
        """Highest ``<device>-<stream>-<session>-<seq>`` suffix already stored.

        ``event_id`` is UNIQUE, and its sequence counter lives in memory in the
        alert manager. A hub restart while a device stays connected leaves the
        device's ``session_id`` unchanged, so a counter that restarted at 1 would
        regenerate event_ids that are already in the table and every insert would
        fail. Recovering the high-water mark from storage keeps the sequence
        monotonic across restarts.
        """
        prefix = f"{device_id}-{stream_id}-{session_id}-"
        highest = 0
        for row in self.conn.execute(
            "SELECT event_id FROM alerts WHERE event_id LIKE ? || '%' ESCAPE '\\'",
            (prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_"),),
        ):
            tail = str(row["event_id"])[len(prefix):]
            if tail.isdigit():
                highest = max(highest, int(tail))
        return highest

    def dismissed_by_rule(self) -> list[dict[str, Any]]:
        """False-positive statistics aggregated by rule_name (§3)."""
        return [
            {"rule_name": r["rule_name"], "dismissed": r["n"], "total": r["total"]}
            for r in self.conn.execute(
                "SELECT rule_name,"
                " SUM(CASE WHEN state = 'dismissed' THEN 1 ELSE 0 END) AS n,"
                " COUNT(*) AS total FROM alerts GROUP BY rule_name"
            )
        ]

    # -- devices ---------------------------------------------------------
    def upsert_device(
        self,
        device_id: str,
        online: bool,
        last_seen_ms: int,
        mode: str,
        info: dict[str, Any],
    ) -> None:
        self.conn.execute(
            """INSERT INTO devices (device_id, online, last_seen_ms, mode, info)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(device_id) DO UPDATE SET
                 online = excluded.online,
                 last_seen_ms = excluded.last_seen_ms,
                 mode = excluded.mode,
                 info = excluded.info""",
            (device_id, int(online), last_seen_ms, mode, json.dumps(info)),
        )

    def list_devices(self) -> list[dict[str, Any]]:
        return [
            {
                "device_id": r["device_id"],
                "online": bool(r["online"]),
                "last_seen_ms": r["last_seen_ms"],
                "mode": r["mode"],
                **(json.loads(r["info"]) if r["info"] else {}),
            }
            for r in self.conn.execute("SELECT * FROM devices ORDER BY device_id")
        ]

    def device_mode(self, device_id: str) -> str:
        row = self.conn.execute(
            "SELECT mode FROM devices WHERE device_id = ?", (device_id,)
        ).fetchone()
        return row["mode"] if row else "hub"

    # -- rules -----------------------------------------------------------
    def get_rules(self, device_id: str, stream_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT rev, body, updated_ms FROM rules WHERE device_id = ? AND stream_id = ?",
            (device_id, stream_id),
        ).fetchone()
        if row is None:
            return None
        return {
            "rev": row["rev"],
            "updated_ms": row["updated_ms"],
            "body": json.loads(row["body"]),
        }

    def rules_body(self, device_id: str, stream_id: str) -> dict[str, Any] | None:
        entry = self.get_rules(device_id, stream_id)
        return entry["body"] if entry else None

    def all_rules(self) -> dict[str, dict[str, Any]]:
        """Two-level index tree: ``{device_id: {stream_id: {rev, body, ...}}}``."""
        tree: dict[str, dict[str, Any]] = {}
        for r in self.conn.execute(
            "SELECT device_id, stream_id, rev, body, updated_ms FROM rules"
            " ORDER BY device_id, stream_id"
        ):
            tree.setdefault(r["device_id"], {})[r["stream_id"]] = {
                "rev": r["rev"],
                "updated_ms": r["updated_ms"],
                "body": json.loads(r["body"]),
            }
        return tree

    def put_rules(
        self, device_id: str, stream_id: str, body: dict[str, Any], now_ms: int
    ) -> dict[str, Any]:
        """Replace one stream's rules; bump rev, version it, mirror to disk."""
        return self._put_rules_many([(device_id, stream_id, body)], now_ms)[0]

    def put_device_rules(
        self, device_id: str, streams: dict[str, Any], now_ms: int
    ) -> list[dict[str, Any]]:
        """Restore a whole device's config (PUT /devices/{id}/config)."""
        items = [(device_id, sid, body) for sid, body in streams.items()]
        return self._put_rules_many(items, now_ms)

    def _put_rules_many(
        self, items: Iterable[tuple[str, str, dict[str, Any]]], now_ms: int
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            for device_id, stream_id, body in items:
                row = self.conn.execute(
                    "SELECT rev FROM rules WHERE device_id = ? AND stream_id = ?",
                    (device_id, stream_id),
                ).fetchone()
                rev = (row["rev"] + 1) if row else 1
                blob = json.dumps(body, ensure_ascii=False, sort_keys=True)
                self.conn.execute(
                    """INSERT INTO rules (device_id, stream_id, rev, body, updated_ms)
                       VALUES (?, ?, ?, ?, ?)
                       ON CONFLICT(device_id, stream_id) DO UPDATE SET
                         rev = excluded.rev, body = excluded.body,
                         updated_ms = excluded.updated_ms""",
                    (device_id, stream_id, rev, blob, now_ms),
                )
                scope = f"rules:{device_id}/{stream_id}"
                self._insert_version(scope, rev, blob, now_ms)
                results.append(
                    {
                        "device_id": device_id,
                        "stream_id": stream_id,
                        "rev": rev,
                        "persisted_ms": now_ms,
                    }
                )
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise
        self._mirror_config()
        return results

    def _insert_version(self, scope: str, rev: int, blob: str, now_ms: int) -> None:
        self.conn.execute(
            "INSERT INTO config_versions (scope, rev, body, created_ms) VALUES (?, ?, ?, ?)",
            (scope, rev, blob, now_ms),
        )
        self.conn.execute(
            """DELETE FROM config_versions WHERE scope = ? AND id NOT IN (
                   SELECT id FROM config_versions WHERE scope = ?
                   ORDER BY id DESC LIMIT ?)""",
            (scope, scope, CONFIG_VERSION_KEEP),
        )

    def config_versions(self, scope: str) -> list[dict[str, Any]]:
        return [
            {"id": r["id"], "rev": r["rev"], "created_ms": r["created_ms"]}
            for r in self.conn.execute(
                "SELECT id, rev, created_ms FROM config_versions WHERE scope = ?"
                " ORDER BY id DESC",
                (scope,),
            )
        ]

    # -- hub config ------------------------------------------------------
    def get_hub_config(self) -> dict[str, Any] | None:
        """Hub-scope config lives in ``config_versions`` under scope ``hub``.

        The DDL in §6 has no dedicated table for it and the pruning window keeps
        the newest revision, so the latest ``hub`` version row *is* the current
        value.
        """
        row = self.conn.execute(
            "SELECT rev, body FROM config_versions WHERE scope = 'hub'"
            " ORDER BY rev DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        return {"rev": row["rev"], "body": json.loads(row["body"])}

    def put_hub_config(self, body: dict[str, Any], now_ms: int) -> dict[str, Any]:
        """Persist hub config with the same rev / version / atomic-write rules."""
        current = self.get_hub_config()
        rev = (current["rev"] + 1) if current else 1
        blob = json.dumps(body, ensure_ascii=False, sort_keys=True)
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            self._insert_version("hub", rev, blob, now_ms)
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise
        self._mirror_config()
        return {"rev": rev, "persisted_ms": now_ms}

    def _mirror_config(self) -> None:
        """Atomically rewrite the single config.json mirror (HUB_SPEC §4)."""
        hub = self.get_hub_config()
        atomic_write_json(
            self.config_path,
            {
                "hub": (hub or {}).get("body", {}),
                "hub_rev": (hub or {}).get("rev", 0),
                "rules": self.all_rules(),
            },
        )

    # -- auth ------------------------------------------------------------
    def get_auth(self, username: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT username, password_hash, must_change FROM auth WHERE username = ?",
            (username,),
        ).fetchone()
        if row is None:
            return None
        return {
            "username": row["username"],
            "password_hash": row["password_hash"],
            "must_change": bool(row["must_change"]),
        }

    def any_auth(self) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT username FROM auth LIMIT 1").fetchone()
        return self.get_auth(row["username"]) if row else None

    def set_auth(self, username: str, password_hash: str, must_change: bool) -> None:
        self.conn.execute(
            """INSERT INTO auth (username, password_hash, must_change) VALUES (?, ?, ?)
               ON CONFLICT(username) DO UPDATE SET
                 password_hash = excluded.password_hash,
                 must_change = excluded.must_change""",
            (username, password_hash, int(must_change)),
        )

    # -- retention -------------------------------------------------------
    def purge_older_than(self, cutoff_ms: int) -> int:
        """Delete alerts (and their snapshot files) older than ``cutoff_ms``."""
        rows = self.conn.execute(
            "SELECT id, snapshot_path FROM alerts WHERE received_ms < ?", (cutoff_ms,)
        ).fetchall()
        for row in rows:
            if row["snapshot_path"]:
                try:
                    Path(row["snapshot_path"]).unlink(missing_ok=True)
                except OSError:
                    pass
        self.conn.execute("DELETE FROM alerts WHERE received_ms < ?", (cutoff_ms,))
        for day_dir in sorted(self.snapshot_dir.glob("*")):
            if day_dir.is_dir() and not any(day_dir.iterdir()):
                day_dir.rmdir()
        return len(rows)

    def snapshot_target(self, event_id: str, now_ms: int | None = None) -> Path:
        """``/data/snapshots/<YYYYMMDD>/<event_id>.jpg`` (HUB_SPEC §6)."""
        stamp = time.strftime("%Y%m%d", time.localtime((now_ms or int(time.time() * 1000)) / 1000))
        safe = event_id.replace("/", "_")
        target = self.snapshot_dir / stamp / f"{safe}.jpg"
        target.parent.mkdir(parents=True, exist_ok=True)
        return target
