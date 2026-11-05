
import json
import logging
import sqlite3
from contextlib import closing
from dataclasses import dataclass, field
from functools import lru_cache

from config.settings import DB_PATH

log = logging.getLogger(__name__)


# the result object
@dataclass
class ParsedQuery:
    semantic_text: str = ""
    category: str | None = None
    price_min: float | None = None
    price_max: float | None = None
    calories_min: int | None = None
    calories_max: int | None = None
    include_ingredients: list[str] = field(default_factory=list)   # must contain ALL
    exclude_ingredients: list[str] = field(default_factory=list)   # must contain NONE
    popular_only: bool = False
    order_by: str | None = None
    order_dir: str = "asc"


# read from your DB
@lru_cache(maxsize=1)
def load_menu_vocab():
    """(categories, ingredients) exactly as stored in the items table.
    Call load_menu_vocab.cache_clear() after the menu changes."""
    with closing(sqlite3.connect(DB_PATH)) as conn:
        rows = conn.execute("SELECT primary_category, ingredients FROM items WHERE active = 1").fetchall()
    cats, ings = set(), set()
    for cat, raw in rows:
        if cat:
            cats.add(cat)
        try:
            ings.update(str(x).strip() for x in json.loads(raw or "[]") if str(x).strip())
        except (json.JSONDecodeError, TypeError):
            pass
    return sorted(cats), sorted(ings)


# system prompt
SYSTEM = """You turn a customer's message into a structured search query for a restaurant menu.
The message may be English, Arabic (any dialect, e.g. Levantine), Arabizi, or a mix, and may contain typos
or Arabic-Indic digits. The menu is in English and prices are in USD.

Fields:
- semantic_text: English words for the dish / taste / texture / cuisine wanted ("spicy cheesy", "sushi", "crunchy").
  Translate Arabic, fix typos. Do NOT include numbers, limits, excluded ingredients, or anything already
  captured by another field. Empty string if nothing is left.
- category: only if the customer asks for a whole menu category. Choose from the allowed list, else null.
- price_min / price_max: "under 15", "اقل من 15 دولار" -> price_max 15. "over 20" -> price_min 20.
- calories_min / calories_max: same, for numbers with calories / cal / kcal / سعرات / كالوري.
- A number with no clear unit, or in a non-USD currency (شيكل, shekel, ₪, JD): do NOT guess, leave the limit null.
- include_ingredients: ingredients the dish must contain ("with mozzarella", "بالجبنة").
  Choose ONLY from the allowed ingredient list.
- exclude_ingredients: things to avoid ("no nuts", "بدون مكسرات", "sugar free", "non-alcoholic").
  Choose ONLY from the allowed ingredient list, and for a group (nuts, dairy, alcohol, sugar, ...) list EVERY
  matching ingredient from that list.
- popular_only: true for popular / best sellers / الأكثر طلبا.
- order_by: "cheapest" / الأرخص -> price, asc. "low calorie" / "lightest" / خفيف -> calories, asc.
  If the customer gives ONLY an upper limit (e.g. "under $15") and nothing else, order_by that field with
  order_dir desc. Otherwise order_by null.

Never invent constraints the customer did not state. When unsure, leave a field empty and keep the words in
semantic_text. If the message is not about food, leave everything empty."""


def build_schema() -> dict:
    """Plain JSON Schema of the answer we expect. Whatever LLM you use, give it this
    (as tool/function parameters or as the response format)."""
    cats, ings = load_menu_vocab()
    num = {"type": ["number", "null"]}
    return {
        "type": "object",
        "properties": {
            "semantic_text": {"type": "string"},
            "category": {"type": ["string", "null"], "enum": cats + [None]},
            "price_min": num, "price_max": num,
            "calories_min": num, "calories_max": num,
            "include_ingredients": {"type": "array", "items": {"type": "string", "enum": ings}},
            "exclude_ingredients": {"type": "array", "items": {"type": "string", "enum": ings}},
            "popular_only": {"type": "boolean"},
            "order_by": {"type": ["string", "null"], "enum": ["price", "calories", None]},
            "order_dir": {"type": "string", "enum": ["asc", "desc"]},
        },
        "required": ["semantic_text"],
    }


# turn the LLM's answer into a ParsedQuery
def _num(x):
    try:
        return float(x) if x is not None else None
    except (TypeError, ValueError):
        return None

def apply_default_order(q: ParsedQuery) -> ParsedQuery:
    """Only a limit, no text, no order asked -> closest to the limit first."""
    if not q.semantic_text and q.order_by is None:
        for field, lo, hi in (("price", q.price_min, q.price_max),
                              ("calories", q.calories_min, q.calories_max)):
            if lo is None and hi is None:
                continue
            q.order_by = field
            q.order_dir = "desc" if (hi is not None and lo is None) else "asc"   # only an upper limit -> descending
            break
    return q

def build_query(raw: dict) -> ParsedQuery:
    """`raw` = the LLM's answer, a dict shaped like build_schema()."""
    cats, ings = load_menu_vocab()               # cheap safety check, not parsing: keep only DB values
    allowed_ings = set(ings)
    exclude = [i for i in dict.fromkeys(raw.get("exclude_ingredients") or []) if i in allowed_ings]
    include = [i for i in dict.fromkeys(raw.get("include_ingredients") or [])
               if i in allowed_ings and i not in exclude]
    cmin, cmax = _num(raw.get("calories_min")), _num(raw.get("calories_max"))

    q = ParsedQuery(                              # <-- was: return ParsedQuery(
        semantic_text=(raw.get("semantic_text") or "").strip(),
        category=raw.get("category") if raw.get("category") in cats else None,
        price_min=_num(raw.get("price_min")),
        price_max=_num(raw.get("price_max")),
        calories_min=int(cmin) if cmin is not None else None,
        calories_max=int(cmax) if cmax is not None else None,
        include_ingredients=include,
        exclude_ingredients=exclude,
        popular_only=bool(raw.get("popular_only")),
        order_by=raw.get("order_by") if raw.get("order_by") in ("price", "calories") else None,
        order_dir="desc" if raw.get("order_dir") == "desc" else "asc",
    )

    return apply_default_order(q)

def parse_query(text: str) -> ParsedQuery:
    text = (text or "").strip()
    if not text:
        return ParsedQuery()
    # TODO (LLM step): raw = <ask the LLM with SYSTEM + build_schema()>; return build_query(raw)
    return ParsedQuery(semantic_text=text)       # for now: search the raw text, no filters