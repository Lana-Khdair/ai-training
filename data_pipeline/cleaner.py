from __future__ import annotations

import json
import math
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "clean"

REQUIRED_FIELDS = ("id", "name", "price", "category")
MAX_REJECT_RATIO = 0.20  # more than 20% rejected -> something is wrong with the source

SPELLING_FIXES = {"sharable": "shareable", "favourite": "favorite", "flavour": "flavor"}
SPELLING_RE = re.compile(r"\b(" + "|".join(SPELLING_FIXES) + r")\b")
DASHES_RE = re.compile(r"\s*[\u2013\u2014]\s*")
# Joke words that say nothing about the food itself.
FILLER_RE = re.compile(
    r"\b(?:fake[- ]?order|fake|pretend|imaginary|fictional|hypothetical|delivery[- ]app|never[- ]arriving)\b",
    re.IGNORECASE,
)


class CleanError(Exception):
    """The snapshot as a whole is unusable."""

def collapse_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def to_number(value) -> float | None:
    """Accept 16.99, 16, "16.99", "$16.99". Return None if it is not a finite number."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, str):
        try:
            number = float(value.strip().replace("$", "").replace(",", ""))
        except ValueError:
            return None
        return number if math.isfinite(number) else None
    return None


def clean_description(text: str) -> str:
    """Lowercase, accent-free, spelling-normalized text with joke words removed."""
    t = DASHES_RE.sub(", ", text)  # "a — b"      -> "a, b"
    t = strip_accents(t).lower()  # "Café"       -> "cafe"
    t = SPELLING_RE.sub(lambda m: SPELLING_FIXES[m.group(1)], t)  # "flavour" -> "flavor"
    t = FILLER_RE.sub("", t)  # remove joke words
    t = re.sub(r"\s+", " ", t)  # fix double spaces left by removals
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)  # "bomb ," -> "bomb,"
    t = re.sub(r"(?:,\s*){2,}", ", ", t)  # ", , " -> ", "
    return t.strip(" ,")  # no leading/trailing commas or spaces


def clean_tags(tags) -> list[str]:
    if not isinstance(tags, list):
        return []
    out: list[str] = []
    for tag in tags:
        if not isinstance(tag, str):
            continue
        t = re.sub(r"[\s_]+", "-", tag.strip().lower())
        t = SPELLING_FIXES.get(t, t)
        if t and t != "popular" and t not in out:  # "popular" is already a separate flag
            out.append(t)
    return out


# ---------- one item ----------

def clean_item(raw, known_categories: list[str]):
    """Return (clean_item_or_None, reject_reasons, warnings)."""
    if not isinstance(raw, dict):
        return None, ["item is not an object"], []

    reasons: list[str] = []
    warnings: list[str] = []

    missing = [f for f in REQUIRED_FIELDS
               if raw.get(f) is None or (isinstance(raw.get(f), str) and not raw[f].strip())]
    reasons += [f"missing required field: {f}" for f in missing]

    price = None
    if "price" not in missing:
        price = to_number(raw["price"])
        if price is None or price < 0:
            reasons.append(f"invalid price: {raw['price']!r}")
        elif price == 0:
            warnings.append("price is 0")

    calories = None
    if raw.get("calories") is None:
        warnings.append("calories missing")
    else:
        calories = to_number(raw["calories"])
        if calories is None or calories < 0:
            reasons.append(f"invalid calories: {raw['calories']!r}")

    # Categories: one list, main category first.
    canonical = {c.lower(): c for c in known_categories}
    categories: list[str] = []
    if "category" not in missing:
        primary = canonical.get(str(raw["category"]).strip().lower())
        if primary is None:
            reasons.append(f"unknown category: {raw['category']!r}")
        else:
            categories.append(primary)
    extras = raw.get("categories")
    if extras is not None and not isinstance(extras, list):
        warnings.append("'categories' is not a list - ignored")
        extras = []
    for extra in extras or []:
        found = canonical.get(str(extra).strip().lower())
        if found is None:
            warnings.append(f"unknown extra category dropped: {extra!r}")
        elif found not in categories:
            categories.append(found)

    if reasons:
        return None, reasons, warnings

    description = collapse_spaces(raw["description"]) if isinstance(raw.get("description"), str) else ""
    if not description:
        warnings.append("description missing")
    image = raw["image"].strip() if isinstance(raw.get("image"), str) and raw["image"].strip() else None
    if image is None:
        warnings.append("image missing")
    if raw.get("slug") and raw["slug"] != raw["id"]:
        warnings.append("slug differs from id")

    tags = clean_tags(raw.get("tags"))
    had_popular_tag = any(isinstance(t, str) and t.strip().lower() == "popular" for t in raw.get("tags") or [])

    item = {
        "id": str(raw["id"]).strip(),
        "name": collapse_spaces(str(raw["name"])),
        "primary_category": categories[0],
        "categories": categories,
        "price": round(price, 2),
        "calories": int(round(calories)) if calories is not None else None,
        "description": description,
        "description_clean": clean_description(description),
        "tags": tags,
        "popular": bool(raw.get("popular")) or had_popular_tag,
        "image": image,
    }
    return item, [], warnings


# ---------- whole snapshot ----------

def clean_menu(snapshot: dict) -> dict:
    foods = snapshot.get("foods")
    categories = snapshot.get("categories")
    if not isinstance(foods, list) or not foods:
        raise CleanError("Snapshot has no 'foods' list.")
    if not isinstance(categories, list) or not categories:
        raise CleanError("Snapshot has no 'categories' list.")

    items, rejected, warnings = [], [], []
    seen_ids: set[str] = set()
    seen_names: dict[str, str] = {}

    for raw in foods:
        item, reasons, item_warnings = clean_item(raw, categories)
        label = raw.get("id") if isinstance(raw, dict) else None

        if item is not None and item["id"] in seen_ids:
            item, reasons = None, ["duplicate id (first one kept)"]

        if item is None:
            rejected.append({"id": label, "reasons": reasons, "raw": raw})
            continue

        seen_ids.add(item["id"])
        name_key = strip_accents(item["name"]).lower()
        if name_key in seen_names:
            item_warnings.append(f"same name as item {seen_names[name_key]!r}")
        seen_names[name_key] = item["id"]

        warnings += [f"{item['id']}: {w}" for w in item_warnings]
        items.append(item)

    total = len(foods)
    ratio = len(rejected) / total
    ok = bool(items) and ratio <= MAX_REJECT_RATIO
    return {
        "ok": ok,
        "items": items,
        "rejected": rejected,
        "warnings": warnings,
        "stats": {"raw_items": total, "kept": len(items), "rejected": len(rejected),
                  "reject_ratio": round(ratio, 3), "warnings": len(warnings)},
    }


# ---------- files / command line ----------

def latest_raw() -> Path | None:
    files = sorted(RAW_DIR.glob("menu_*.json"))
    return files[-1] if files else None


def main(argv: list[str]) -> int:
    path = Path(argv[1]) if len(argv) > 1 else latest_raw()
    if path is None or not path.exists():
        print(f"No input file found. Pass one: python cleaner.py path/to/raw.json  (looked in {RAW_DIR})")
        return 2

    snapshot = json.loads(path.read_text(encoding="utf-8"))
    try:
        result = clean_menu(snapshot)
    except CleanError as exc:
        print(f"FAILED: {exc}")
        return 1

    stats = result["stats"]
    print(f"Input : {path}")
    print(f"Items : {stats['raw_items']} raw -> {stats['kept']} kept, {stats['rejected']} rejected, "
          f"{stats['warnings']} warnings")
    for entry in result["rejected"]:
        print(f"  REJECTED {entry['id']}: {'; '.join(entry['reasons'])}")
    for warning in result["warnings"][:20]:
        print(f"  warning  {warning}")
    if len(result["warnings"]) > 20:
        print(f"  ... and {len(result['warnings']) - 20} more warnings")

    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    stem = path.stem
    if result["rejected"]:
        (CLEAN_DIR / f"{stem}_rejected.json").write_text(
            json.dumps(result["rejected"], ensure_ascii=False, indent=2), encoding="utf-8")

    if not result["ok"]:
        print(f"FAILED: reject ratio {stats['reject_ratio']:.0%} is above {MAX_REJECT_RATIO:.0%} "
              f"(or nothing was kept). No clean file written.")
        return 1

    out = CLEAN_DIR / f"{stem}_clean.json"
    out.write_text(json.dumps({
        "cleaned_at": datetime.now(timezone.utc).isoformat(),
        "source_file": path.name,
        "scraped_at": snapshot.get("scraped_at"),
        "categories": snapshot["categories"],
        "items": result["items"],
        "stats": stats,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved : {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))