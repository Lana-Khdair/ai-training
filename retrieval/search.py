from dataclasses import asdict, replace
from functools import lru_cache
import torch
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition, Filter, MatchAny, MatchValue, OrderBy, Range,
)
from sentence_transformers import SentenceTransformer
from parsing.query_parser import ParsedQuery, parse_query
from config.settings import (
    EMBEDDING_DIM, EMBEDDING_MODEL, QDRANT_API_KEY, QDRANT_COLLECTION, QDRANT_URL,
)

SORTABLE = {"price", "calories"}

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


# semantic search (the filter is applied INSIDE the vector search)
def semantic_search(text, flt, limit, score_threshold=None):
    res = get_client().query_points(
        QDRANT_COLLECTION,
        query=embed_query(text),
        query_filter=flt,                 # None = pure semantic
        limit=limit,
        score_threshold=score_threshold,
        with_payload=True,
    )
    return [_hit(p, p.score) for p in res.points]


# filters only
def filter_search(q: ParsedQuery, flt, limit):
    order = OrderBy(key=q.order_by, direction=q.order_dir) if q.order_by in SORTABLE else None
    points, _ = get_client().scroll(
        QDRANT_COLLECTION, scroll_filter=flt, order_by=order, limit=limit, with_payload=True,
    )
    return [_hit(p) for p in points]


# one search, four modes
def search(q: ParsedQuery, limit: int = 10, score_threshold: float | None = None) -> dict:
    """mode: "semantic" (text only), "hybrid" (text + filters),
             "filters" (filters only), "none" (nothing to search)."""
    flt = build_qdrant_filter(q)

    if q.semantic_text:
        mode = "hybrid" if flt else "semantic"
        hits = semantic_search(q.semantic_text, flt, limit, score_threshold)
    elif flt or q.order_by:
        mode = "filters"
        hits = filter_search(q, flt, limit)
    else:
        mode, hits = "none", []

    return {"mode": mode, "query": asdict(q), "hits": hits}

DROP_ORDER = [      # least important first
    ("popular",     {"popular_only": False},
        lambda q: q.popular_only),
    ("ingredients", {"include_ingredients": []},
        lambda q: bool(q.include_ingredients)),
    ("price",       {"price_min": None, "price_max": None},
        lambda q: q.price_min is not None or q.price_max is not None),
    ("calories",    {"calories_min": None, "calories_max": None},
        lambda q: q.calories_min is not None or q.calories_max is not None),
    ("category",    {"category": None},
        lambda q: bool(q.category)),
]

def search_always(text: str, limit: int = 10, score_threshold: float | None = None) -> dict:
    q = parse_query(text)
    r = search(q, limit, score_threshold)
    if r["mode"] == "none" or r["hits"]:
        return {**r, "fallback": None, "dropped": []}

    sem = " ".join(filter(None, [q.category, *q.include_ingredients, q.semantic_text])) or text

    if score_threshold is not None and q.semantic_text:     # same query, no threshold
        r = search(q, limit)
        if r["hits"]:
            return {**r, "fallback": "No exact match. Showing the closest results.",
                    "dropped": ["similarity threshold"]}

    loose, dropped = replace(q, semantic_text=sem, order_by=None), []
    for name, reset, used in DROP_ORDER:
        if not used(q):
            continue
        loose = replace(loose, **reset)                     # exclusions are never reset
        dropped.append(name)
        r = search(loose, limit)
        if r["hits"]:
            return {**r, "dropped": dropped,
                    "fallback": f"No exact match. Showing the closest results, ignoring: {', '.join(dropped)}."}

    return {**r, "fallback": "No results found.", "dropped": []}