
from __future__ import annotations
import json
import re
from datetime import datetime, timezone
from pathlib import Path
import requests

BASE_URL = "https://foodnevercome.com"
ORDER_URL = f"{BASE_URL}/order"
HEADERS = {"User-Agent": "FoodMenuPipeline/1.0 (student project;)"}
TIMEOUT = 20
RAW_DIR = Path(__file__).resolve().parent / "data" / "raw"

CHUNK_RE = re.compile(r"/_next/static/chunks/[\w\-]+\.js")
FOODS_RE = re.compile(r'"foods"\s*,\s*0\s*,\s*\[')
CATEGORIES_RE = re.compile(r'"categories"\s*,\s*0\s*,\s*\[')
STRING_RE = re.compile(r'"(?:[^"\\]|\\.)*"')


class ScrapeError(Exception):
    """Raised when the site structure is not what we expect."""


def fetch_text(url: str) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    resp.raise_for_status()
    resp.encoding = "utf-8"
    return resp.text


def find_chunk_urls(html: str) -> list[str]:
    """All unique JS chunk URLs mentioned anywhere in the page"""
    unique = list(dict.fromkeys(CHUNK_RE.findall(html)))
    return [BASE_URL + path for path in unique]


def download_menu_chunk(chunk_urls: list[str]) -> tuple[str, str]:
    """Return (url, js_text) of the chunk that contains the menu data."""
    for url in chunk_urls:
        js = fetch_text(url)
        if FOODS_RE.search(js) and CATEGORIES_RE.search(js):
            return url, js
    raise ScrapeError(
        f"Menu data not found in any of {len(chunk_urls)} chunks - site structure may have changed."
    )


def extract_array(js: str, marker: re.Pattern) -> str:
    """Return the text of the [...] that follows `marker`, matching brackets
    while ignoring any brackets that appear inside strings."""
    match = marker.search(js)
    if not match:
        raise ScrapeError(f"Marker {marker.pattern!r} not found.")
    start = match.end() - 1  # position of the opening '['
    depth, in_str, i = 0, False, start
    while i < len(js):
        ch = js[i]
        if in_str:
            if ch == "\\":
                i += 2
                continue
            if ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return js[start : i + 1]
        i += 1
    raise ScrapeError("Unbalanced brackets while reading the array.")


def _fix_js_segment(seg: str) -> str:
    seg = seg.replace("!0", "true").replace("!1", "false")        # minified booleans
    seg = re.sub(r"(?<![\w.])\.(\d)", r"0.\1", seg)               # .99 -> 0.99
    seg = re.sub(r"([{,]\s*)([A-Za-z_$][\w$]*)(\s*:)", r'\1"\2"\3', seg)  # quote keys
    return seg


def js_literal_to_json(text: str):
    """Convert a JS array/object literal to Python data.
    Only the parts OUTSIDE string literals are rewritten, so descriptions
    are never touched."""
    out, pos = [], 0
    for m in STRING_RE.finditer(text):
        out.append(_fix_js_segment(text[pos : m.start()]))
        out.append(m.group())
        pos = m.end()
    out.append(_fix_js_segment(text[pos:]))
    return json.loads("".join(out))


def scrape_menu() -> dict:
    html = fetch_text(ORDER_URL)
    chunk_urls = find_chunk_urls(html)
    if not chunk_urls:
        raise ScrapeError("No JS chunks found on /order - site structure may have changed.")
    chunk_url, js = download_menu_chunk(chunk_urls)

    categories = js_literal_to_json(extract_array(js, CATEGORIES_RE))
    foods = js_literal_to_json(extract_array(js, FOODS_RE))

    if not isinstance(foods, list) or not foods or not all(isinstance(f, dict) and "id" in f for f in foods):
        raise ScrapeError("Extracted menu does not look like a list of items with ids.")

    return {
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "source_url": ORDER_URL,
        "chunk_url": chunk_url,
        "categories": categories,
        "foods": foods,
    }


def save_snapshot(payload: dict) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.fromisoformat(payload["scraped_at"]).strftime("%Y%m%d_%H%M%S")
    path = RAW_DIR / f"menu_{stamp}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    data = scrape_menu()
    path = save_snapshot(data)
    print(f"Scraped {len(data['foods'])} items in {len(data['categories'])} categories")
    print(f"Source chunk: {data['chunk_url']}")
    print(f"Saved to: {path}")