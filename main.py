from __future__ import annotations

import sys

import scraper
import cleaner
import features
import sync
from runner import run_pipeline
from config.settings import SCRAPE_ATTEMPTS, SCRAPE_RETRY_WAIT

# (name, function to call, max attempts, seconds between attempts)
# Only the scraper is retried: network problems are common there, while a
# failing cleaner/features/sync is a real problem a retry won't fix.
STAGES = [
    ("scrape", scraper.run, SCRAPE_ATTEMPTS, SCRAPE_RETRY_WAIT),
    ("clean", cleaner.run, 1, 0),
    ("features", features.run, 1, 0),
    ("sync", sync.run, 1, 0),
]


def main() -> int:
    return run_pipeline(STAGES)


if __name__ == "__main__":
    sys.exit(main())