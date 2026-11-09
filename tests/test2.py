
import json
import sys
from pathlib import Path

from retrieval.search import search_always

QUERIES_FILE = Path(__file__).resolve().parents[1] / "data" / "eval_queries.json"
LIMIT = 5


def load_queries():
    if len(sys.argv) > 1:
        return sys.argv[1:]
    return [case["query"] for case in json.loads(QUERIES_FILE.read_text(encoding="utf-8"))]


def main():
    for text in load_queries():
        r = search_always(text, limit=LIMIT)
        q = r["query"]
        print(f"\n{text!r} -> {r['mode']}")
        if r.get("fallback"):
            print(f"  fallback: {r['fallback']}")
        print(f"  semantic={q['semantic_text']!r}  category={q['category']}  "
              f"price=({q['price_min']},{q['price_max']})  "
              f"cal=({q['calories_min']},{q['calories_max']})  "
              f"include={q['include_ingredients']}  exclude={(q['exclude_ingredients'])}  "
              f"order={q['order_by']}")
        if not r["hits"]:
            print("  (no results)")
        for i, h in enumerate(r["hits"], 1):
            p = h["payload"]
            score = f"  score={h['score']:.3f}" if h["score"] is not None else ""
            print(f"  {i}. {p['name']:30} ${p['price']}  {p['calories']}cal{score}")


if __name__ == "__main__":
    main()