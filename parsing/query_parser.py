import json
import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass, field
from difflib import get_close_matches
from functools import lru_cache
from nltk.corpus import stopwords
import nltk
from config.settings import DB_PATH


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
    order_by: str | None = None        # "price" or "calories" (filters-only queries)
    order_dir: str = "asc"


# ---------------- patterns & word lists ----------------
COMPARATORS = {
    "max": ["under", "below", "less than", "lower than", "fewer than", "up to",
            "at most", "no more than", "cheaper than", "max", "maximum"],
    "min": ["over", "above", "more than", "at least", "higher than", "min", "minimum"],
}
_CMP_KIND = {w: kind for kind, words in COMPARATORS.items() for w in words}
_NUM = r"\d+(?:\.\d+)?"
_UNIT = r"(?P<unit>\$|[a-z]+)?"

NUM_RE = re.compile(
    r"\b(?P<cmp>" + "|".join(sorted(map(re.escape, _CMP_KIND), key=len, reverse=True)) + r")"
    rf"\s*(?P<cur>\$)?\s*(?P<num>{_NUM})\s*{_UNIT}"
)
BETWEEN_RE = re.compile(
    rf"\bbetween\s+(?P<cur>\$)?\s*(?P<a>{_NUM})\s*(?:and|to|-)\s*\$?\s*(?P<b>{_NUM})\s*{_UNIT}"
)
POPULAR_RE = re.compile(r"\b(?:popular|best[\s-]?sellers?|most ordered)\b")
CHEAPEST_RE = re.compile(r"\b(?:cheapest|lowest price)\b")
LIGHTEST_RE = re.compile(r"\b(?:low[\s-]cal(?:orie)?s?|lowest calories?|lightest)\b")
EXCLUDE_RE = re.compile(r"\b(?:no|without|non|excluding)[\s-]+(?P<term>[a-z]+)")
FREE_RE    = re.compile(r"\b(?P<term>[a-z]+)[\s-]+free\b")      # "nut-free"
INCLUDE_TRIGGERS = {"with", "contain", "contains", "containing", "has", "have",
                    "having", "including", "includes"}
CAL_UNITS = ["calories", "calorie", "cal", "cals", "kcal"]
MONEY_UNITS = ["dollars", "dollar", "usd", "bucks", "buck"]
EXCLUDE_GROUPS = {
    "alcohol": ["beer", "wine", "vodka", "whiskey", "whisky", "rum", "gin", "tequila",
                "brandy", "champagne", "bourbon", "liqueur", "sake", "cocktail"],
    "nuts": ["almond", "peanut", "walnut", "cashew", "pistachio", "hazelnut", "pecan"],
    "dairy": ["milk", "cheese", "butter", "cream", "yogurt", "mozzarella", "parmesan"],
    "sugar": ["sugar", "syrup", "honey", "molasses"],
}

try:
    _EN = set(stopwords.words("english"))
except LookupError:                      # first run only: downloads the list
    nltk.download("stopwords")
    _EN = set(stopwords.words("english"))

# words specific to ordering food (NLTK doesn't know them)
DOMAIN_STOPWORDS = {"im", "i'm", "wanna", "something", "anything", "show", "find",
                    "get", "want", "need", "give", "looking", "please", "like",
                    "dishes", "menu", "food"}

STOPWORDS = _EN | DOMAIN_STOPWORDS
for _w in _CMP_KIND:                      # "over", "above", "least", "than"... are never semantic
    STOPWORDS.update(_w.split())


# ---------------- vocabulary from your DB ----------------
#turns a JSON (ingredients, tags) into a clean list
def _as_list(v):
    try:
        return [str(x).strip() for x in json.loads(v or "[]") if str(x).strip()]
    except (json.JSONDecodeError, TypeError):
        return []


def _singular(w):
    return w[:-1] if len(w) > 3 and w.endswith("s") else w

#lowercases a phrase and singularizes each word
def _norm(phrase):
    return " ".join(_singular(w) for w in re.findall(r"[a-z]+", phrase.lower()))

#for recognizing and mapping
@dataclass
class Vocab:
    terms: dict    # "category" | "ingredient" | "tag" -> {normalized text -> exact stored value}
    words: set     # every known word, for spell-correction


def build_vocab(rows) -> Vocab:
    terms = {"category": {}, "ingredient": {}, "tag": {}}
    words = set()
    for r in rows:
        cat = r["primary_category"]
        ings = _as_list(r["ingredients"])
        tags = _as_list(r["tags"])

        # categories are matched word by word ("drinks" -> "Hot Drinks"); None = ambiguous
        if cat:
            for w in re.findall(r"[a-z]+", cat.lower()):
                if len(w) >= 4:
                    key = _singular(w)
                    cur = terms["category"].get(key, cat)
                    terms["category"][key] = cat if cur == cat else None

        # ingredients and tags are matched as whole phrases
        for kind, vals in (("ingredient", ings), ("tag", tags)):
            for val in vals:
                key = _norm(val)
                if key:
                    terms[kind][key] = val

        # every word seen anywhere is the spell-check dictionary
        for text in (cat, r["name"], r["description"], *ings, *tags):
            words.update(re.findall(r"[a-z]+", (text or "").lower()))

    terms["category"] = {k: v for k, v in terms["category"].items() if v}
    return Vocab(terms, words)

#Fetches name, category, tags, and ingredients for every active item
@lru_cache(maxsize=1)
def load_vocab() -> Vocab:
    """Cached. Call load_vocab.cache_clear() after the menu changes."""
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT name, primary_category, tags, ingredients, description FROM items WHERE active = 1"
        ).fetchall()
    return build_vocab(rows)


# ---------------- helpers ----------------
def classify_unit(unit):
    if not unit:
        return None
    if unit == "$":
        return "price"
    if len(unit) < 3:
        return None
    if unit in CAL_UNITS or get_close_matches(unit, CAL_UNITS, n=1, cutoff=0.8):
        return "calories"
    if unit in MONEY_UNITS or get_close_matches(unit, MONEY_UNITS, n=1, cutoff=0.8):
        return "price"
    return None


def _set_limit(q, kind, lo=None, hi=None):
    if kind == "price":
        if lo is not None: q.price_min = lo
        if hi is not None: q.price_max = hi
    else:
        if lo is not None: q.calories_min = int(lo)
        if hi is not None: q.calories_max = int(hi)


def resolve_exclusion(term, v):
    ings = v.terms["ingredient"]
    if len(term) >= 3:
        for group, members in EXCLUDE_GROUPS.items():
            if term.startswith(group) or group.startswith(term):
                return [ing for key, ing in ings.items()
                        if any(m in key.split() for m in members)]
    key = _singular(term)
    if key in ings:
        return [ings[key]]
    hit = get_close_matches(key, list(ings), n=1, cutoff=0.85)
    return [ings[hit[0]]] if hit else []


def match_ingredient(toks, i, v):
    ings = v.terms["ingredient"]
    for n in (3, 2, 1):
        if i + n > len(toks):
            continue
        key = " ".join(_singular(x) for x in toks[i:i + n])
        if key in ings:
            return ings[key], n
    tok = _singular(toks[i])
    if len(tok) >= 4:
        hit = get_close_matches(tok, list(ings), n=1, cutoff=0.85)
        if hit:
            return ings[hit[0]], 1
    return None
# ---------------- the parser ----------------
def parse_query(text: str) -> ParsedQuery:
    v = load_vocab()
    q = ParsedQuery()
    t = text.lower()

    # 1a) ranges: "between 10 and 20 dollars"
    def _between(m):
        uk = classify_unit(m.group("unit"))
        kind = "price" if m.group("cur") else uk
        if kind is None:
            return m.group(0)
        lo, hi = sorted((float(m.group("a")), float(m.group("b"))))
        _set_limit(q, kind, lo, hi)
        return f" {'' if uk else (m.group('unit') or '')} "      # keep a word that wasn't a unit
    t = BETWEEN_RE.sub(_between, t)

    # 1b) limits: "less than 20 doller", "under 300 calori", "more than $20"
    def _num(m):
        uk = classify_unit(m.group("unit"))
        kind = "price" if m.group("cur") else uk
        if kind is None:
            return m.group(0)                          # number without a clear unit: ignore
        num = float(m.group("num"))
        if _CMP_KIND[m.group("cmp")] == "max":
            _set_limit(q, kind, hi=num)
        else:
            _set_limit(q, kind, lo=num)
        return f" {'' if uk else (m.group('unit') or '')} "
    t = NUM_RE.sub(_num, t)

    # 2) exclusions: "no alcohol", "without nuts", "non-alcoholic"
    def _excl(m):
        term = m.group("term")
        names = resolve_exclusion(term, v)
        is_group = len(term) >= 3 and any(term.startswith(g) or g.startswith(term) for g in EXCLUDE_GROUPS)
        if not names and not is_group:
            return m.group(0)
        q.exclude_ingredients.extend(n for n in names if n not in q.exclude_ingredients)
        return " "

    t = EXCLUDE_RE.sub(_excl, t)
    t = FREE_RE.sub(_excl, t)

    # 3) flags and sort words
    if POPULAR_RE.search(t):
        q.popular_only = True
        t = POPULAR_RE.sub(" ", t)
    if CHEAPEST_RE.search(t):
        q.order_by = "price"
        t = CHEAPEST_RE.sub(" ", t)
    if LIGHTEST_RE.search(t):
        q.order_by = "calories"
        t = LIGHTEST_RE.sub(" ", t)

    # 4) walk the remaining words: includes, category, spell-fix, semantic text
    toks = re.findall(r"[a-z']+", t)
    rest, i, including = [], 0, False
    while i < len(toks):
        tok = toks[i]

        if tok in INCLUDE_TRIGGERS:                    # "with mozzarella and basil"
            including = True
            i += 1
            continue
        if including:
            hit = match_ingredient(toks, i, v)
            if hit:
                name, n = hit
                if name not in q.include_ingredients and name not in q.exclude_ingredients:
                    q.include_ingredients.append(name)
                i += n
                continue
            if tok in ("and", "or"):
                i += 1
                continue
            including = False

        if tok in STOPWORDS:
            i += 1
            continue

        if len(tok) >= 4:                              # category (fuzzy only for unknown words)
            cats = v.terms["category"]
            key = _singular(tok)
            cat_key = key if key in cats else None
            if cat_key is None and tok not in v.words:
                cat_key = (get_close_matches(key, list(cats), n=1, cutoff=0.75) or [None])[0]
            if cat_key:
                cat = cats[cat_key]
                if q.category is None or q.category == cat:
                    q.category = cat
                    i += 1
                    continue

        if tok not in v.words and len(tok) >= 4:
            cutoff = 0.9 if len(tok) <= 5 else 0.8     # stricter for short words
            hit = get_close_matches(tok, list(v.words), n=1, cutoff=cutoff)
            tok = hit[0] if hit else tok
        rest.append(tok)
        i += 1
    q.semantic_text = " ".join(rest)


    # 5) filters-only queries
    if not q.semantic_text and q.order_by is None:
        if q.price_max is not None or q.calories_max is not None:
            q.order_by, q.order_dir = ("price" if q.price_max is not None else "calories"), "desc"
    return q