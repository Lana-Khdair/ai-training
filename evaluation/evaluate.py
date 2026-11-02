"""Retrieval evaluation. Run from the project root:  python -m evaluation.evaluate

Each case in data/eval_queries.json:
  query            the text a customer would type
  type             semantic | hybrid | filters | fallback | none   (only used for the report)
  expected         dish names that are a GOOD answer (any one of them counts as a hit)
  mode             optional: the search mode you expect (semantic / hybrid / filters / none)
  checks           optional rules every returned dish must follow:
                     price_min, price_max, calories_min, calories_max, category,
                     include [ingredients], exclude [ingredients],
                     order_by (price | calories) + order_dir (asc | desc),
                     expect_fallback (true when the exact filters should find nothing)
"""
import json
from collections import defaultdict
from pathlib import Path

from retrieval.search import search_always

QUERIES_FILE = Path(__file__).resolve().parents[1] / "data" / "eval_queries.json"
K = 5


def norm(s):
    return str(s).strip().lower()


def check_constraints(hits, chk):
    """Every returned dish must respect the limits the query asked for."""
    problems = []
    for h in hits:
        p = h["payload"]
        name, price, cal = p["name"], p.get("price"), p.get("calories")
        ings = {norm(i) for i in p.get("ingredients", [])}

        if "price_max" in chk and price is not None and price > chk["price_max"]:
            problems.append(f"{name}: price {price} > {chk['price_max']}")
        if "price_min" in chk and price is not None and price < chk["price_min"]:
            problems.append(f"{name}: price {price} < {chk['price_min']}")
        if "calories_max" in chk and cal is not None and cal > chk["calories_max"]:
            problems.append(f"{name}: {cal} cal > {chk['calories_max']}")
        if "calories_min" in chk and cal is not None and cal < chk["calories_min"]:
            problems.append(f"{name}: {cal} cal < {chk['calories_min']}")
        if "category" in chk and p.get("primary_category") != chk["category"]:
            problems.append(f"{name}: category {p.get('primary_category')} != {chk['category']}")
        for ing in chk.get("include", []):
            if norm(ing) not in ings:
                problems.append(f"{name}: missing ingredient {ing}")
        for ing in chk.get("exclude", []):
            if norm(ing) in ings:
                problems.append(f"{name}: contains excluded {ing}")

    key = chk.get("order_by")
    if key:
        vals = [h["payload"][key] for h in hits if h["payload"].get(key) is not None]
        want = sorted(vals, reverse=chk.get("order_dir", "asc") == "desc")
        if vals != want:
            problems.append(f"not sorted by {key} {chk.get('order_dir', 'asc')}: {vals}")
    return problems


def run_case(case):
    r = search_always(case["query"], limit=K)
    hits, chk = r["hits"], case.get("checks", {})
    names = [norm(h["payload"]["name"]) for h in hits]
    expected = {norm(x) for x in case.get("expected", [])}

    res = {"query": case["query"], "type": case.get("type", "other"),
           "mode": r["mode"], "fallback": r.get("fallback"),
           "top": [h["payload"]["name"] for h in hits],
           "has_expected": bool(expected), "rank": None, "problems": []}

    # 1) did a good dish show up, and how high?
    if expected:
        ranks = [i + 1 for i, n in enumerate(names) if n in expected]
        res["rank"] = min(ranks) if ranks else None
        if res["rank"] is None:
            res["problems"].append(f"none of the expected dishes in top {K}")

        found = {n for n in names if n in expected}           # distinct expected dishes returned
        relevant_returned = sum(1 for n in names if n in expected)
        res["precision"] = relevant_returned / max(len(names), 1)
        res["recall"] = len(found) / len(expected)
        p, rc = res["precision"], res["recall"]
        res["f1"] = 2 * p * rc / (p + rc) if (p + rc) else 0.0

    # 2) did the parser/search choose the right mode?
    if "mode" in case and case["mode"] != r["mode"]:
        res["problems"].append(f"mode {r['mode']} (expected {case['mode']})")

    # 3) fallback should happen only when the case says so
    want_fb = chk.get("expect_fallback", False)
    if want_fb and not r.get("fallback"):
        res["problems"].append("expected a fallback, but the exact search found results")
    if r.get("fallback") and not want_fb:
        res["problems"].append(f"unexpected fallback ({r.get('dropped')})")

    # 4) no results at all (except for "none" cases)
    if not hits and case.get("mode") != "none":
        res["problems"].append("no results")

    # 5) filters must hold on exact (non-fallback) results; exclusions must ALWAYS hold
    if not r.get("fallback"):
        res["problems"] += check_constraints(hits, chk)
    else:
        res["problems"] += check_constraints(hits, {"exclude": chk.get("exclude", [])})
    return res


def pct(x):
    return f"{100 * x:.0f}%"


def summarize(results):
    scored = [r for r in results if r["has_expected"]]
    n = max(len(scored), 1)
    hit = lambda k: sum(1 for r in scored if r["rank"] and r["rank"] <= k) / n
    mrr = sum(1 / r["rank"] for r in scored if r["rank"]) / n
    ok = sum(1 for r in results if not r["problems"]) / max(len(results), 1)
    return {"n": len(results), "ranked": len(scored), "hit1": hit(1), "hit3": hit(3),
            "hit5": hit(5), "mrr": mrr, "ok": ok,
            "precision": sum(r["precision"] for r in scored) / n,
            "recall": sum(r["recall"] for r in scored) / n,
            "f1": sum(r["f1"] for r in scored) / n}

def print_row(label, s):
    print(f"{label:10} n={s['n']:<3} ranked={s['ranked']:<3} "
          f"Hit@1={pct(s['hit1']):>4} Hit@3={pct(s['hit3']):>4} Hit@5={pct(s['hit5']):>4} "
          f"MRR={s['mrr']:.2f}  P@{K}={pct(s['precision']):>4} R@{K}={pct(s['recall']):>4} "
          f"F1={s['f1']:.2f}  checks passed={pct(s['ok'])}")

def main():
    cases = json.loads(QUERIES_FILE.read_text(encoding="utf-8"))
    results = [run_case(c) for c in cases]

    failures = [r for r in results if r["problems"]]
    for r in failures:
        print(f"\nFAIL {r['query']!r}  (type={r['type']}, mode={r['mode']})")
        for p in r["problems"]:
            print(f"   - {p}")
        print(f"   top: {r['top']}")

    print("\n" + "=" * 70)
    by_type = defaultdict(list)
    for r in results:
        by_type[r["type"]].append(r)
    for t, rs in sorted(by_type.items()):
        print_row(t, summarize(rs))
    print("-" * 70)
    print_row("ALL", summarize(results))


if __name__ == "__main__":
    main()