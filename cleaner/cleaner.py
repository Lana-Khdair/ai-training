from __future__ import annotations

import json
import math
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from collections import defaultdict
from config.settings import RAW_DIR, CLEAN_DIR, MAX_REJECT_RATIO, MIN_SAMPLES

DATA_DIR = Path(__file__).resolve().parent / "data"
REQUIRED_FIELDS = ("id", "name", "category")
SPELLING_FIXES = {"sharable": "shareable", "favourite": "favorite", "flavour": "flavor"}
SPELLING_RE = re.compile(r"\b(" + "|".join(SPELLING_FIXES) + r")\b")
DASHES_RE = re.compile(r"\s*[\u2013\u2014]\s*")


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
    """Normalize whitespace without changing the original meaning."""
    return collapse_spaces(text)


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

def clean_modifier_groups(groups) -> list[dict]:
    """Preserve valid structured customization groups."""
    if not isinstance(groups, list):
        return []

    cleaned = []

    for group in groups:
        if not isinstance(group, dict):
            continue

        group_id = group.get("id")
        options = group.get("options")

        if not isinstance(group_id, str) or not group_id.strip():
            continue
        if not isinstance(options, list):
            continue

        valid_options = []
        for option in options:
            if not isinstance(option, dict):
                continue

            option_id = option.get("id")
            option_name = option.get("name")

            if not isinstance(option_id, str) or not option_id.strip():
                continue
            if not isinstance(option_name, str) or not option_name.strip():
                continue

            cleaned_option = {
                "id": option_id.strip(),
                "name": collapse_spaces(option_name),
            }

            if "priceDelta" in option:
                delta = to_number(option["priceDelta"])
                if delta is None:
                    continue
                cleaned_option["priceDelta"] = round(delta, 2)

            if "default" in option:
                cleaned_option["default"] = bool(option["default"])

            valid_options.append(cleaned_option)

        if not valid_options:
            continue

        cleaned_group = {
            "id": group_id.strip(),
            "options": valid_options,
        }

        for field in ("name", "required", "minSelections", "maxSelections"):
            if field in group:
                cleaned_group[field] = group[field]

        cleaned.append(cleaned_group)

    return cleaned


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
    if raw.get("price") is None or (isinstance(raw.get("price"), str) and not raw["price"].strip()):
        warnings.append("price missing")
    else:
        price = to_number(raw["price"])
        if price is None or price < 0:
            reasons.append(f"invalid price: {raw['price']!r}")   # present but garbage -> still rejected
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

    item_id = str(raw["id"]).strip()
    item_name = collapse_spaces(str(raw["name"]))

    if not item_id:
        reasons.append("id is empty after normalization")

    if not item_name:
        reasons.append("name is empty after normalization")

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
        "id": item_id,
        "name": item_name,
        "primary_category": categories[0],
        "categories": categories,
        "price": round(price, 2) if price is not None else None,
        "price_imputed": False,
        "calories": int(round(calories)) if calories is not None else None,
        "calories_imputed": False,
        "description": description,
        "description_clean": clean_description(description),
        "tags": tags,
        "popular": bool(raw.get("popular")) or had_popular_tag,
        "image": image,
        "modifier_groups": clean_modifier_groups(
            raw.get("modifier_groups", [])
        ),
    }
    return item, [], warnings

def _fill_with_median(items: list[dict], field: str, flag: str, convert, usable=lambda v: True) -> int:
    by_cat = defaultdict(list)
    for it in items:
        v = it[field]
        if v is not None and usable(v):
            by_cat[it["primary_category"]].append(v)

    all_values = [v for vals in by_cat.values() for v in vals]
    global_median = median(all_values) if all_values else None

    filled = 0
    for it in items:
        if it[field] is not None:
            continue
        vals = by_cat.get(it["primary_category"], [])
        guess = median(vals) if len(vals) >= MIN_SAMPLES else global_median
        if guess is not None:
            it[field] = convert(guess)
            it[flag] = True
            filled += 1
    return filled

def impute_calories(items: list[dict]) -> int:
    return _fill_with_median(items, "calories", "calories_imputed", lambda v: int(round(v)))


def impute_prices(items: list[dict]) -> int:
    return _fill_with_median(items, "price", "price_imputed", lambda v: round(v, 2), usable=lambda v: v > 0)

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

    imputed = impute_calories(items)
    prices_imputed = impute_prices(items)

    ok = bool(items) and ratio <= MAX_REJECT_RATIO
    return {
        "ok": ok,
        "items": items,
        "rejected": rejected,
        "warnings": warnings,
        "stats": {"raw_items": total, "kept": len(items), "rejected": len(rejected),
                  "reject_ratio": round(ratio, 3), "warnings": len(warnings),
                  "calories_imputed": imputed, "prices_imputed": prices_imputed},
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
    print(f"Imputed calories : {stats['calories_imputed']}")
    print(f"Imputed prices   : {stats['prices_imputed']}")
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