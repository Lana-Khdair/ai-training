from dataclasses import asdict, replace
from functools import lru_cache
import torch
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition, Filter, MatchAny, MatchValue, OrderBy, Range,
)
from sentence_transformers import SentenceTransformer
from parsing.query_llm import ParsedQuery, parse_query
from config.settings import (
    EMBEDDING_DIM, EMBEDDING_MODEL, QDRANT_API_KEY, QDRANT_COLLECTION, QDRANT_URL,
)

SORTABLE = {"price", "calories"}
MARGINS = (0.10, 0.25, 0.50)   # how far past the user's price/calorie limits we may go, step by step
REL_CUTOFF = 0.08              # drop semantic hits scoring this far below the best hit (None = off). Tune it!
POOL = 50                      # candidates ranked by the "closest to your limits" step

# ---------- shared resources ----------
#connects to Qdrant
@lru_cache(maxsize=1)
def get_client():
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=120)

#loads the embedding model
@lru_cache(maxsize=1)
def get_model():
    return SentenceTransformer(
        EMBEDDING_MODEL,
        config_kwargs={"vision_config": None, "audio_config": None},
        model_kwargs={"torch_dtype": torch.float32},
    )

#turns text into a vector, using the “query” mode
@lru_cache(maxsize=256)
def embed_query(text):
    return get_model().encode([text], prompt_name="query", truncate_dim=EMBEDDING_DIM,
                              normalize_embeddings=True)[0].tolist()


# Turns the parsed query into a Qdrant filter
def build_qdrant_filter(q: ParsedQuery) -> Filter | None:
    must, must_not = [], []
    if q.category:
        must.append(FieldCondition(key="primary_category", match=MatchValue(value=q.category)))
    if q.price_min is not None or q.price_max is not None:
        must.append(FieldCondition(key="price", range=Range(gte=q.price_min, lte=q.price_max)))
    if q.calories_min is not None or q.calories_max is not None:
        must.append(FieldCondition(key="calories", range=Range(gte=q.calories_min, lte=q.calories_max)))
    if q.popular_only:
        must.append(FieldCondition(key="popular", match=MatchValue(value=True)))
    for ing in q.include_ingredients:                       # one condition each = ALL required
        must.append(FieldCondition(key="ingredients", match=MatchValue(value=ing)))
    if q.exclude_ingredients:
        must_not.append(FieldCondition(key="ingredients", match=MatchAny(any=q.exclude_ingredients)))
    if not must and not must_not:
        return None
    return Filter(must=must or None, must_not=must_not or None)


def _hit(p, score=None):
    return {"id": str(p.id), "score": score, "payload": p.payload}


# drops weak semantic hits that are far below the best one (e.g. "sushi" -> dumplings)
def _trim(hits):
    if REL_CUTOFF is None or not hits:
        return hits
    best = hits[0]["score"]
    return [h for h in hits if h["score"] >= best - REL_CUTOFF]


# semantic search (the filter is applied INSIDE the vector search)
def semantic_search(text, flt, limit, score_threshold=None, trim=True):
    res = get_client().query_points(
        QDRANT_COLLECTION,
        query=embed_query(text),
        query_filter=flt,                 # None = pure semantic
        limit=limit,
        score_threshold=score_threshold,
        with_payload=True,
    )
    hits = [_hit(p, p.score) for p in res.points]
    return _trim(hits) if trim else hits


# filters only
def filter_search(q: ParsedQuery, flt, limit):
    order = OrderBy(key=q.order_by, direction=q.order_dir) if q.order_by in SORTABLE else None
    points, _ = get_client().scroll(
        QDRANT_COLLECTION, scroll_filter=flt, order_by=order, limit=limit, with_payload=True,
    )
    return [_hit(p) for p in points]


# sort key for order_by; items with a missing value go last
def _by_field(hit, key, desc):
    v = hit["payload"].get(key)
    if v is None:
        return float("-inf") if desc else float("inf")
    return v


# one search, four modes
def search(q: ParsedQuery, limit: int = 10, score_threshold: float | None = None) -> dict:
    """mode: "semantic" (text only), "hybrid" (text + filters),
             "filters" (filters only), "none" (nothing to search)."""
    flt = build_qdrant_filter(q)

    if q.semantic_text:
        mode = "hybrid" if flt else "semantic"
        if q.order_by in SORTABLE:
            # text + explicit order ("cheapest spicy burger"): take the relevant hits
            # (REL_CUTOFF applies), then sort THOSE by the field
            hits = semantic_search(q.semantic_text, flt, POOL, score_threshold)
            desc = q.order_dir == "desc"
            hits.sort(key=lambda h: _by_field(h, q.order_by, desc), reverse=desc)
            hits = hits[:limit]
        else:
            hits = semantic_search(q.semantic_text, flt, limit, score_threshold)
    elif flt or q.order_by:
        mode = "filters"
        hits = filter_search(q, flt, limit)
    else:
        mode, hits = "none", []

    return {"mode": mode, "query": asdict(q), "hits": hits}


# ---------- numeric limits (price / calories) are never ignored ----------
def _has_numeric(q: ParsedQuery) -> bool:
    return any(v is not None for v in (q.price_min, q.price_max, q.calories_min, q.calories_max))


def _no_numeric(q: ParsedQuery) -> ParsedQuery:
    return replace(q, price_min=None, price_max=None, calories_min=None, calories_max=None)


def _widen(q: ParsedQuery, m: float) -> ParsedQuery:
    """margin m: upper limits grow by m, lower limits shrink by m (0.10 = 10%)"""
    up = lambda v: None if v is None else v * (1 + m)
    down = lambda v: None if v is None else v * (1 - m)
    return replace(q, price_max=up(q.price_max), price_min=down(q.price_min),
                   calories_max=up(q.calories_max), calories_min=down(q.calories_min))


def _miss(payload, q: ParsedQuery) -> float:
    """how far an item is outside the requested price/calorie range, relative to the limit (0 = inside)"""
    total = 0.0
    for key, lo, hi in (("price", q.price_min, q.price_max), ("calories", q.calories_min, q.calories_max)):
        if lo is None and hi is None:
            continue
        v = payload.get(key)
        if v is None:
            total += 1.0
            continue
        if lo is not None and v < lo:
            total += (lo - v) / max(lo, 1)
        if hi is not None and v > hi:
            total += (v - hi) / max(hi, 1)
    return total


def _relax_numeric(q: ParsedQuery, limit: int):
    """Same query, limits kept: 1) widen them by growing margins, 2) rank by closeness to them."""
    if not _has_numeric(q):
        return None
    for m in MARGINS:
        r = search(_widen(q, m), limit if q.semantic_text else POOL)
        if r["hits"]:
            if not q.semantic_text:  # filter-only: closest to the ORIGINAL limit first
                r["hits"].sort(key=lambda h: _miss(h["payload"], q))
                r["hits"] = r["hits"][:limit]
            return {**r, "margin": m,
                    "note": f"Nothing within your limits, so showing results up to {m:.0%} beyond them."}
    base = _no_numeric(replace(q, order_by=None))           # keep category / ingredients / exclusions / text
    flt = build_qdrant_filter(base)
    if base.semantic_text:
        hits = semantic_search(base.semantic_text, flt, POOL)
    else:
        hits = filter_search(base, flt, POOL)
    if not hits:
        return None
    hits.sort(key=lambda h: _miss(h["payload"], q))         # stable: ties keep the semantic order
    return {"mode": "nearest", "query": asdict(q), "hits": hits[:limit], "margin": None,
            "note": "Nothing matches your limits, so showing the closest options to them."}


# ---------- fallback ----------
DROP_ORDER = [      # least important first. price / calories are NOT here any more
    ("popular",     {"popular_only": False},
        lambda q: q.popular_only),
    ("ingredients", {"include_ingredients": []},
        lambda q: bool(q.include_ingredients)),
    ("category",    {"category": None},
        lambda q: bool(q.category)),
]

# builds the final result: adds `dropped` and one combined `fallback` message
def _finish(r: dict, dropped: list[str]) -> dict:
    parts = []
    if dropped:
        parts.append(f"No exact match. Ignoring: {', '.join(dropped)}.")
    note = r.pop("note", None)
    if note:
        parts.append(note)
    return {**r, "dropped": dropped, "margin": r.get("margin"),
            "fallback": " ".join(parts) or "No exact match. Showing the closest results."}

# search_always: runs the steps in order and returns at the FIRST step that finds hits.
#
#   1. strict      all filters exactly as asked
#   2. no score threshold          (only if a score_threshold was passed)
#   3. margins     price/calorie limits widened 10% -> 25% -> 50%
#   4. nearest     numeric limits removed, candidates ranked by distance to the limits
#   5-7. drop soft filters one by one: popular -> ingredients -> category
#      (dropped words move into semantic_text; after each drop, steps 1, 3 and 4 run again)
#   8. nothing found -> "No results found."
#
# Steps 3-4 run only when the query has a price/calorie limit.
# Never dropped or ignored: excluded ingredients, price/calorie limits, semantic text.

def search_always(text: str, limit: int = 10, score_threshold: float | None = None) -> dict:
    q = parse_query(text)
    strict = search(q, limit, score_threshold)
    if strict["mode"] == "none" or strict["hits"]:          # 1. strict
        return {**strict, "fallback": None, "dropped": [], "margin": None}

    if score_threshold is not None and q.semantic_text:     # 2. same query, no score threshold
        r = search(q, limit)
        if r["hits"]:
            return {**r, "fallback": "No exact match. Showing the closest results.",
                    "dropped": ["similarity threshold"], "margin": None}

    r = _relax_numeric(q, limit)                           # 3. margins, then 4. nearest
    if r:
        return _finish(r, [])

    # runs only when the above found nothing: drop soft filters one by one
    sem = " ".join(filter(None, [q.category, *q.include_ingredients, q.semantic_text])) or text
    loose, dropped = replace(q, semantic_text=sem, order_by=None), []
    for name, reset, used in DROP_ORDER:                  # 5-7. drop popular -> ingredients -> category
        if not used(q):
            continue
        loose = replace(loose, **reset)                    # drop it (and keep it dropped)
        dropped.append(name)
        r = search(loose, limit)                           # steps 1, then 3 + 4 again on the loosened query
        if not r["hits"]:
            r = _relax_numeric(loose, limit) or r
        if r["hits"]:
            return _finish(r, dropped)

    return {**strict, "fallback": "No results found.", "dropped": [], "margin": None}  # 8. nothing found