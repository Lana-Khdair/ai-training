from __future__ import annotations

import contextlib
import io
import json
import logging
import time
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler

from config.settings import LOG_DIR

log = logging.getLogger("pipeline")


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S")

    file_handler = RotatingFileHandler(
        LOG_DIR / "pipeline.log", maxBytes=1_000_000, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)

    console = logging.StreamHandler()
    console.setFormatter(fmt)

    log.setLevel(logging.INFO)
    log.handlers.clear()
    log.addHandler(file_handler)
    log.addHandler(console)


def run_stage(name: str, func, attempts: int, wait: int) -> dict:
    result = {"stage": name, "ok": False, "attempts": 0, "seconds": 0.0, "return_code": None}
    started = time.monotonic()

    for attempt in range(1, attempts + 1):
        result["attempts"] = attempt
        log.info("[%s] starting (attempt %d/%d)", name, attempt, attempts)
        buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(buffer):   # capture the stage's print() output
                code = func()
        except SystemExit as exc:                      # e.g. missing spaCy model
            code = 1
            log.error("[%s] stopped: %s", name, exc)
        except Exception:
            code = 1
            log.exception("[%s] crashed", name)

        for line in buffer.getvalue().splitlines():
            if line.strip():
                log.info("[%s] %s", name, line)

        result["return_code"] = code
        if code == 0:
            result["ok"] = True
            break
        log.error("[%s] failed with code %s", name, code)
        if attempt < attempts:
            log.warning("[%s] retrying in %ds", name, wait)
            time.sleep(wait)

    result["seconds"] = round(time.monotonic() - started, 1)
    return result


def run_pipeline(stages: list[tuple]) -> int:
    """stages: list of (name, function, max_attempts, wait_seconds)."""
    setup_logging()
    started_at = datetime.now(timezone.utc)
    log.info("=" * 60)
    log.info("Pipeline run started")

    results: list[dict] = []
    failed_stage = None

    for name, func, attempts, wait in stages:
        res = run_stage(name, func, attempts, wait)
        results.append(res)
        if not res["ok"]:
            failed_stage = name
            break
        log.info("[%s] finished OK in %.1fs", name, res["seconds"])

    status = "success" if failed_stage is None else "failed"
    if failed_stage:
        log.error("Pipeline FAILED at stage '%s'. Later stages were skipped, "
                  "so the database was not changed by this run.", failed_stage)
    else:
        log.info("Pipeline finished successfully")

    (LOG_DIR / "last_run.json").write_text(json.dumps({
        "status": status,
        "failed_stage": failed_stage,
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "stages": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    return 0 if status == "success" else 1