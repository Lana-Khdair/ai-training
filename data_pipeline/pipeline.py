
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
LOG_DIR = PACKAGE_DIR / "data" / "logs"

TASK_NAME = "FoodMenuPipeline"
DEFAULT_DAY = "MON"
DEFAULT_TIME = "03:00"

# (name, module, max attempts, seconds to wait between attempts)
# Only the scraper is retried: network hiccups are common there, while a
# failing cleaner/features/sync run is a real problem a retry won't fix.
STAGES = [
    ("scrape", "data_pipeline.scraper", 3, 30),
    ("clean", "data_pipeline.cleaner", 1, 0),
    ("features", "data_pipeline.features", 1, 0),
    ("sync", "data_pipeline.sync", 1, 0),
]
STAGE_TIMEOUT = 15 * 60  # seconds; a stuck stage must not block the schedule forever

log = logging.getLogger("pipeline")


# ---------------------------------------------------------------- logging

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


# ---------------------------------------------------------------- running the stages

def run_stage(name: str, module: str, attempts: int, wait: int) -> dict:
    """Run one stage as `python -m <module>`. Returns a result dict."""
    # Force UTF-8 in the child, otherwise Windows may crash when it prints
    # characters like 'é' or '–' into a pipe.
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    result = {"stage": name, "ok": False, "attempts": 0, "seconds": 0.0, "return_code": None}
    started = time.monotonic()

    for attempt in range(1, attempts + 1):
        result["attempts"] = attempt
        log.info("[%s] starting (attempt %d/%d)", name, attempt, attempts)
        try:
            proc = subprocess.run(
                [sys.executable, "-m", module],
                cwd=PROJECT_ROOT,
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=STAGE_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            log.error("[%s] timed out after %ds", name, STAGE_TIMEOUT)
            result["return_code"] = "timeout"
        else:
            result["return_code"] = proc.returncode
            for line in proc.stdout.splitlines():
                if line.strip():
                    log.info("[%s] %s", name, line)
            for line in proc.stderr.splitlines():
                if line.strip():
                    log.error("[%s] %s", name, line)
            if proc.returncode == 0:
                result["ok"] = True
                break
            log.error("[%s] exited with code %s", name, proc.returncode)

        if attempt < attempts:
            log.warning("[%s] retrying in %ds", name, wait)
            time.sleep(wait)

    result["seconds"] = round(time.monotonic() - started, 1)
    return result


def write_summary(summary: dict) -> None:
    (LOG_DIR / "last_run.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run_pipeline() -> int:
    setup_logging()
    started_at = datetime.now(timezone.utc)
    log.info("=" * 60)
    log.info("Pipeline run started")

    results: list[dict] = []
    failed_stage = None

    for name, module, attempts, wait in STAGES:
        res = run_stage(name, module, attempts, wait)
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

    write_summary({
        "status": status,
        "failed_stage": failed_stage,
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "stages": results,
    })
    return 0 if status == "success" else 1


# ---------------------------------------------------------------- weekly schedule (Windows)

def _require_windows() -> bool:
    if os.name != "nt":
        print("Scheduling is built for Windows Task Scheduler. On Linux/macOS use cron, e.g.:\n"
              f'  0 3 * * 1  cd "{PROJECT_ROOT}" && "{sys.executable}" "{Path(__file__).resolve()}"')
        return False
    return True


def install_schedule(day: str, at: str) -> int:
    """Register (or replace) a weekly task that runs this very file."""
    if not _require_windows():
        return 1

    # We run the file by its path instead of `-m`, so the task does not depend
    # on the "start in" folder. The pipeline sets the right folder itself.
    command = f'"{sys.executable}" "{Path(__file__).resolve()}"'
    proc = subprocess.run(
        ["schtasks", "/Create", "/F", "/TN", TASK_NAME,
         "/SC", "WEEKLY", "/D", day.upper(), "/ST", at, "/TR", command],
        capture_output=True, text=True,
    )
    print(proc.stdout.strip() or proc.stderr.strip())
    if proc.returncode != 0:
        print("Could not create the task.")
        return 1

    # Best effort: if the PC was off at the scheduled time, run as soon as it is on again.
    ps = (f"Set-ScheduledTask -TaskName '{TASK_NAME}' "
          "-Settings (New-ScheduledTaskSettingsSet -StartWhenAvailable "
          "-AllowStartIfOnBatteries -DontStopIfGoingOnBatteries) | Out-Null")
    extra = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                           capture_output=True, text=True)
    if extra.returncode != 0:
        print("Note: could not enable 'run missed schedule when PC is back on'. "
              "Tick it manually in Task Scheduler > task > Settings.")

    print(f"\nScheduled: every {day.upper()} at {at}, using Python:\n  {sys.executable}")
    print(f"Test it now without waiting:  schtasks /Run /TN {TASK_NAME}")
    return 0


def remove_schedule() -> int:
    if not _require_windows():
        return 1
    proc = subprocess.run(["schtasks", "/Delete", "/F", "/TN", TASK_NAME],
                          capture_output=True, text=True)
    print(proc.stdout.strip() or proc.stderr.strip())
    return proc.returncode


def schedule_status() -> int:
    if not _require_windows():
        return 1
    proc = subprocess.run(["schtasks", "/Query", "/TN", TASK_NAME, "/V", "/FO", "LIST"],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        print("No schedule found. Create one with: python -m data_pipeline.pipeline --install-schedule")
        return 1
    wanted = ("TaskName", "Next Run Time", "Last Run Time", "Last Result", "Status", "Schedule Type", "Days")
    for line in proc.stdout.splitlines():
        if line.split(":", 1)[0].strip() in wanted:
            print(line)
    last = LOG_DIR / "last_run.json"
    if last.exists():
        info = json.loads(last.read_text(encoding="utf-8"))
        print(f"\nLast pipeline run: {info['status']} (started {info['started_at']})")
    return 0


# ---------------------------------------------------------------- entry point

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Menu pipeline: scrape -> clean -> features -> sync")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--install-schedule", action="store_true",
                       help="create/replace the weekly Windows Task Scheduler job")
    group.add_argument("--remove-schedule", action="store_true", help="delete the weekly job")
    group.add_argument("--schedule-status", action="store_true", help="show the weekly job and last run")
    parser.add_argument("--day", default=DEFAULT_DAY,
                        choices=["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"],
                        help=f"day of the week for --install-schedule (default {DEFAULT_DAY})")
    parser.add_argument("--time", default=DEFAULT_TIME,
                        help=f"HH:MM (24h) for --install-schedule (default {DEFAULT_TIME})")
    args = parser.parse_args(argv)

    if args.install_schedule:
        return install_schedule(args.day, args.time)
    if args.remove_schedule:
        return remove_schedule()
    if args.schedule_status:
        return schedule_status()
    return run_pipeline()


if __name__ == "__main__":
    sys.exit(main())