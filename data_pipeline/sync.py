
from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
FEATURES_DIR = DATA_DIR / "features"
DB_PATH = DATA_DIR / "menu.db"

MAX_REMOVED_RATIO = 0.30
MIN_ACTIVE_FOR_GUARD = 10

SCALAR_FIELDS = ("name", "primary_category", "price", "calories", "description", "image", "popular")
JSON_FIELDS = ("categories", "tags", "ingredients", "characteristics", "modifier_groups")
TRACKED_FIELDS = SCALAR_FIELDS + JSON_FIELDS

SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    id               TEXT PRIMARY KEY,
    name             TEXT NOT NULL,
    primary_category TEXT NOT NULL,
    price            REAL NOT NULL,
    calories         INTEGER,
    description      TEXT,
    image            TEXT,
    popular          INTEGER NOT NULL DEFAULT 0,
    categories       TEXT NOT NULL,   -- JSON list
    tags             TEXT NOT NULL,   -- JSON list
    ingredients      TEXT NOT NULL,   -- JSON list
    characteristics  TEXT NOT NULL,   -- JSON list
    modifier_groups  TEXT NOT NULL,   -- JSON list
    active           INTEGER NOT NULL DEFAULT 1,
    first_seen       TEXT NOT NULL,
    last_seen        TEXT NOT NULL,
    updated_at       TEXT NOT NULL,
    removed_at       TEXT
);

CREATE TABLE IF NOT EXISTS item_changes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER NOT NULL,
    item_id     TEXT NOT NULL,
    change_type TEXT NOT NULL,        -- added | updated | removed | restored
    field       TEXT,
    old_value   TEXT,
    new_value   TEXT,
    changed_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sync_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    source_file TEXT,
    status      TEXT NOT NULL,        -- running | success | failed
    added       INTEGER DEFAULT 0,
    updated     INTEGER DEFAULT 0,
    removed     INTEGER DEFAULT 0,
    restored    INTEGER DEFAULT 0,
    unchanged   INTEGER DEFAULT 0,
    error       TEXT
);

CREATE INDEX IF NOT EXISTS idx_items_active ON items(active);
CREATE INDEX IF NOT EXISTS idx_changes_item ON item_changes(item_id);
"""


class SyncError(Exception):
    """The sync cannot continue safely."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------- turning a feature item into a DB row ----------

def to_row(item: dict) -> dict:
    """Normalize one item so it can be compared with what is stored in the DB."""
    row = {
        "name": item["name"],
        "primary_category": item["primary_category"],
        "price": round(float(item["price"]), 2),
        "calories": item.get("calories"),
        "description": item.get("description_clean") or item.get("description") or "",
        "image": item.get("image"),
        "popular": int(bool(item.get("popular"))),
    }
    for field in JSON_FIELDS:
        # sort_keys makes the text identical whenever the content is identical
        row[field] = json.dumps(item.get(field) or [], ensure_ascii=False, sort_keys=True)
    return row


def load_incoming(path: Path) -> dict[str, dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SyncError(f"Cannot read {path}: {exc}") from exc

    items = data.get("items") if isinstance(data, dict) else None
    if not isinstance(items, list) or not items:
        raise SyncError("Features file has no items.")

    incoming: dict[str, dict] = {}
    for index, item in enumerate(items):
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not item["id"].strip():
            raise SyncError(f"Item at index {index} has no valid id.")
        if item["id"] in incoming:
            raise SyncError(f"Duplicate id in features file: {item['id']}")
        try:
            incoming[item["id"]] = to_row(item)
        except (KeyError, TypeError, ValueError) as exc:
            raise SyncError(f"Item {item['id']} is incomplete: {exc!r}") from exc
    return incoming


def load_existing(conn: sqlite3.Connection) -> dict[str, dict]:
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM items").fetchall()
    return {r["id"]: dict(r) for r in rows}


# ---------- comparing ----------

def changed_fields(old: dict, new: dict) -> list[tuple[str, object, object]]:
    return [(f, old[f], new[f]) for f in TRACKED_FIELDS if old[f] != new[f]]


def compute_diff(incoming: dict[str, dict], existing: dict[str, dict]) -> dict:
    added, updated, restored, unchanged = [], [], [], []

    for item_id, new in incoming.items():
        old = existing.get(item_id)
        if old is None:
            added.append(item_id)
            continue
        diffs = changed_fields(old, new)
        if not old["active"]:
            restored.append((item_id, diffs))
        elif diffs:
            updated.append((item_id, diffs))
        else:
            unchanged.append(item_id)

    removed = sorted(i for i, old in existing.items() if old["active"] and i not in incoming)
    return {"added": added, "updated": updated, "restored": restored,
            "removed": removed, "unchanged": unchanged}


def check_safety(diff: dict, existing: dict[str, dict]) -> None:
    active = sum(1 for o in existing.values() if o["active"])
    if active >= MIN_ACTIVE_FOR_GUARD:
        ratio = len(diff["removed"]) / active
        if ratio > MAX_REMOVED_RATIO:
            raise SyncError(
                f"{len(diff['removed'])} of {active} active items ({ratio:.0%}) would be removed, "
                f"above the {MAX_REMOVED_RATIO:.0%} limit. Probably a scraping problem. "
                f"Nothing was changed."
            )


# ---------- writing ----------

def log_change(conn, run_id, item_id, change_type, field=None, old=None, new=None):
    conn.execute(
        "INSERT INTO item_changes (run_id, item_id, change_type, field, old_value, new_value, changed_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (run_id, item_id, change_type, field,
         None if old is None else str(old), None if new is None else str(new), now()),
    )


def apply_diff(conn: sqlite3.Connection, run_id: int, diff: dict,
               incoming: dict[str, dict]) -> None:
    ts = now()
    cols = list(TRACKED_FIELDS)

    for item_id in diff["added"]:
        row = incoming[item_id]
        conn.execute(
            f"INSERT INTO items (id, {', '.join(cols)}, active, first_seen, last_seen, updated_at) "
            f"VALUES (?, {', '.join('?' for _ in cols)}, 1, ?, ?, ?)",
            (item_id, *[row[c] for c in cols], ts, ts, ts),
        )
        log_change(conn, run_id, item_id, "added")

    for item_id, diffs in diff["updated"]:
        row = incoming[item_id]
        conn.execute(
            f"UPDATE items SET {', '.join(f'{c} = ?' for c in cols)}, last_seen = ?, updated_at = ? "
            f"WHERE id = ?",
            (*[row[c] for c in cols], ts, ts, item_id),
        )
        for field, old, new in diffs:
            log_change(conn, run_id, item_id, "updated", field, old, new)

    for item_id, diffs in diff["restored"]:
        row = incoming[item_id]
        conn.execute(
            f"UPDATE items SET {', '.join(f'{c} = ?' for c in cols)}, active = 1, removed_at = NULL, "
            f"last_seen = ?, updated_at = ? WHERE id = ?",
            (*[row[c] for c in cols], ts, ts, item_id),
        )
        log_change(conn, run_id, item_id, "restored")

    for item_id in diff["removed"]:
        conn.execute("UPDATE items SET active = 0, removed_at = ?, updated_at = ? WHERE id = ?",
                     (ts, ts, item_id))
        log_change(conn, run_id, item_id, "removed")

    # Everything that was seen this run gets a fresh last_seen
    conn.executemany("UPDATE items SET last_seen = ? WHERE id = ?",
                     [(ts, i) for i in diff["unchanged"]])


def record_failure(run_id: int | None, source: str, error: str) -> None:
    """Best effort: store the failed run after the main transaction rolled back."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.executescript(SCHEMA)
            conn.execute(
                "INSERT INTO sync_runs (started_at, finished_at, source_file, status, error) "
                "VALUES (?, ?, ?, 'failed', ?)", (now(), now(), source, error[:2000]))
    except sqlite3.Error:
        pass


# ---------- main ----------

def latest_features() -> Path | None:
    files = sorted(FEATURES_DIR.glob("*_features.json"))
    return files[-1] if files else None


def main(argv: list[str]) -> int:
    path = Path(argv[1]) if len(argv) > 1 else latest_features()
    if path is None or not path.exists():
        print(f"No input file found. Pass one: python -m data_pipeline.sync path/to/x_features.json "
              f"(looked in {FEATURES_DIR})")
        return 2

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    run_id = None
    try:
        conn.executescript(SCHEMA)
        incoming = load_incoming(path)
        existing = load_existing(conn)
        diff = compute_diff(incoming, existing)
        check_safety(diff, existing)

        with conn:  # one transaction: commits on success, rolls back on any exception
            cur = conn.execute(
                "INSERT INTO sync_runs (started_at, source_file, status) VALUES (?, ?, 'running')",
                (now(), path.name))
            run_id = cur.lastrowid
            apply_diff(conn, run_id, diff, incoming)
            conn.execute(
                "UPDATE sync_runs SET finished_at = ?, status = 'success', added = ?, updated = ?, "
                "removed = ?, restored = ?, unchanged = ? WHERE id = ?",
                (now(), len(diff["added"]), len(diff["updated"]), len(diff["removed"]),
                 len(diff["restored"]), len(diff["unchanged"]), run_id))
    except (SyncError, sqlite3.Error) as exc:
        conn.close()
        print(f"FAILED: {exc}")
        record_failure(run_id, path.name, str(exc))
        return 1
    except Exception as exc:  # unexpected bug: still keep the DB safe and report it
        conn.close()
        print(f"FAILED (unexpected): {exc!r}")
        record_failure(run_id, path.name, repr(exc))
        return 1

    conn.close()
    print(f"Input    : {path}")
    print(f"Database : {DB_PATH}")
    print(f"Result   : {len(diff['added'])} added, {len(diff['updated'])} updated, "
          f"{len(diff['removed'])} removed, {len(diff['restored'])} restored, "
          f"{len(diff['unchanged'])} unchanged")
    for item_id in diff["added"][:10]:
        print(f"  + added    {item_id}")
    for item_id, diffs in diff["updated"][:10]:
        print(f"  ~ updated  {item_id}: {', '.join(f for f, _, _ in diffs)}")
    for item_id in diff["removed"][:10]:
        print(f"  - removed  {item_id}")
    for item_id, _ in diff["restored"][:10]:
        print(f"  ^ restored {item_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))