"""Simple test of the SEARCH layer (no LLM, no expectations).

For every case it prints only:
  - the ParsedQuery (all 11 fields and what each is used for)
  - every search step search_always() runs (strict, threshold, margins, nearest, dropped filters)
  - the final result (mode, margin, dropped filters, fallback message, hits)

Each case = the text the customer types + the ParsedQuery the LLM *should* produce for it.
Arabic cases are twins of English ones (same ParsedQuery; semantic_text is always English).
parse_query is replaced by a lookup, so the fields are exactly the ones written here.

Run from the project root:   python tests/test_search_details.py
"""
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    sys.stdout.reconfigure(encoding="utf-8")        # Arabic output on Windows consoles
except Exception:
    pass

import retrieval.search as S
from parsing.query_llm import ParsedQuery as PQ, apply_default_order


def C(lang, text, q, note="", thr=None):
    return dict(lang=lang, text=text, q=q, note=note, thr=thr)


CASES = [
    # ======================================================= A. SEMANTIC
    "A. SEMANTIC  (semantic_text only -> no filter, ranked by meaning)",
    C("en", "something spicy", PQ(semantic_text="spicy"), "text only: embedded and compared with every item"),
    C("ar", "بدي اشي حار", PQ(semantic_text="spicy"), "Arabic twin: the LLM translates to English semantic_text"),
    C("en", "sushi", PQ(semantic_text="sushi"), "REL_CUTOFF removes the weak matches"),
    C("ar", "سوشي", PQ(semantic_text="sushi")),
    C("en", "ramen", PQ(semantic_text="ramen")),
    C("ar", "رامن", PQ(semantic_text="ramen")),
    C("en", "tacos", PQ(semantic_text="tacos"), "many good matches with similar scores: the cutoff keeps them all"),
    C("ar", "تاكو", PQ(semantic_text="tacos")),
    C("en", "something spicy and cheesy", PQ(semantic_text="spicy cheesy"),
      "one embedding cannot enforce BOTH words"),

    # ======================================================= B. FILTERS
    "B. FILTERS  (no semantic_text -> Qdrant just returns the items that pass; order_by sorts them)",
    C("en", "pizza", PQ(category="Pizza"), "category -> must: primary_category == 'Pizza'"),
    C("ar", "بيتزا", PQ(category="Pizza")),
    C("en", "under $15", PQ(price_max=15, order_by="price", order_dir="desc"), "price_max -> must: price <= 15"),
    C("ar", "اقل من 15 دولار", PQ(price_max=15, order_by="price", order_dir="desc")),
    C("en", "over $20", PQ(price_min=20, order_by="price", order_dir="asc"),
      "price_min -> must: price >= 20; sorted from the limit upward (closest to 20 first)"),
    C("ar", "اكثر من 20 دولار", PQ(price_min=20, order_by="price", order_dir="asc")),
    C("en", "between 8 and 14 dollars", PQ(price_min=8, price_max=14, order_by="price", order_dir="asc"),
      "price_min + price_max -> ONE range condition; sorted from the lower bound upward"),
    C("ar", "بين 8 و 14 دولار", PQ(price_min=8, price_max=14, order_by="price", order_dir="asc")),
    C("en", "under 300 calories", PQ(calories_max=300, order_by="calories", order_dir="desc"),
      "calories_max -> must: calories <= 300"),
    C("ar", "اقل من 300 سعرة حرارية", PQ(calories_max=300, order_by="calories", order_dir="desc")),
    C("en", "more than 500 calories", PQ(calories_min=500, order_by="calories", order_dir="asc"),
      "calories_min -> must: calories >= 500; sorted from the limit upward (closest to 500 first)"),
    C("ar", "اكثر من 500 سعرة حرارية", PQ(calories_min=500, order_by="calories", order_dir="asc")),
    C("en", "dessert over 500 calories", PQ(category="Desserts", calories_min=500, order_by="calories", order_dir="asc"),
      "the limit IS met by an item, so no margin and no nearest step run"),
    C("en", "popular", PQ(popular_only=True), "popular_only -> must: popular == True"),
    C("ar", "الاكثر طلبا", PQ(popular_only=True)),
    C("en", "pizza with mozzarella", PQ(category="Pizza", include_ingredients=["mozzarella"]),
      "include_ingredients -> must: ingredients == 'mozzarella'"),
    C("ar", "بيتزا بالموزاريلا", PQ(category="Pizza", include_ingredients=["mozzarella"])),
    C("en", "pizza with mozzarella and pepperoni", PQ(category="Pizza", include_ingredients=["mozzarella", "pepperoni"]),
      "several included ingredients -> one condition each -> the item needs ALL of them"),
    C("en", "pizza without mozzarella", PQ(category="Pizza", exclude_ingredients=["mozzarella"]),
      "exclude_ingredients -> must_not: ingredients any of ['mozzarella']"),
    C("ar", "بيتزا بدون موزاريلا", PQ(category="Pizza", exclude_ingredients=["mozzarella"])),
    C("en", "pizza without mozzarella and pepperoni", PQ(category="Pizza", exclude_ingredients=["mozzarella", "pepperoni"]),
      "several excluded ingredients -> the item must contain NONE of them"),
    C("en", "cheapest", PQ(order_by="price"), "order_by alone still makes it a filter search (sorted by Qdrant)"),
    C("ar", "الارخص", PQ(order_by="price")),
    C("en", "most expensive", PQ(order_by="price", order_dir="desc"), "order_dir='desc' flips the sort"),
    C("ar", "الاغلى", PQ(order_by="price", order_dir="desc")),
    C("en", "low calorie", PQ(order_by="calories"), "the 0-calorie placeholder items come first"),
    C("ar", "قليل السعرات", PQ(order_by="calories")),

    # ======================================================= C. HYBRID
    "C. HYBRID  (filters first, then the survivors are ranked by meaning; with order_by they are re-sorted)",
    C("en", "spicy burger under $15", PQ(semantic_text="spicy", category="Burgers", price_max=15),
      "the filter is applied INSIDE the vector search"),
    C("ar", "برغر حار بأقل من 15 دولار", PQ(semantic_text="spicy", category="Burgers", price_max=15)),
    C("en", "chocolate dessert", PQ(semantic_text="chocolate", category="Desserts")),
    C("ar", "حلوى بالشوكولاتة", PQ(semantic_text="chocolate", category="Desserts")),
    C("en", "crunchy chicken without dairy",
      PQ(semantic_text="crunchy chicken", exclude_ingredients=["cheese", "milk", "butter", "cream"]),
      "the LLM expands the group 'dairy' into every dairy ingredient of the menu"),
    C("ar", "دجاج مقرمش بدون منتجات الحليب",
      PQ(semantic_text="crunchy chicken", exclude_ingredients=["cheese", "milk", "butter", "cream"])),
    C("en", "spicy burger with cheese, no bacon, 5-20 dollars, under 1000 calories, cheapest first",
      PQ(semantic_text="spicy", category="Burgers", price_min=5, price_max=20, calories_max=1000,
         include_ingredients=["cheese"], exclude_ingredients=["bacon"], order_by="price", order_dir="asc"),
      "almost every field at once; the relevant hits are then sorted by price"),
    C("en", "cheapest spicy burger", PQ(semantic_text="spicy", category="Burgers", order_by="price", order_dir="asc"),
      "text + order_by: relevant burgers first (REL_CUTOFF), then sorted by price ascending"),
    C("ar", "ارخص برغر حار", PQ(semantic_text="spicy", category="Burgers", order_by="price", order_dir="asc")),
    C("en", "most expensive spicy burger", PQ(semantic_text="spicy", category="Burgers", order_by="price", order_dir="desc")),

    # ======================================================= D. NONE
    "D. NONE  (nothing to search -> no search runs, no hits)",
    C("en", "show me please", PQ()),
    C("ar", "ممكن تساعدني", PQ()),
    C("en", "hello", PQ()),
    C("ar", "مرحبا", PQ()),

    # ======================================================= E. FALLBACKS
    "E1. MARGINS  (limits widened 10% -> 25% -> 50%; every other filter stays)",
    C("en", "pizza under $14", PQ(category="Pizza", price_max=14), "14 * 1.10 = 15.4"),
    C("ar", "بيتزا بأقل من 14 دولار", PQ(category="Pizza", price_max=14)),
    C("en", "pizza under $13", PQ(category="Pizza", price_max=13), "13 * 1.25 = 16.25"),
    C("en", "pizza under $10", PQ(category="Pizza", price_max=10), "10 * 1.5 = 15"),
    C("en", "pizza under 700 calories", PQ(category="Pizza", calories_max=700),
      "the same margins apply to calories: 700 * 1.10 = 770"),
    C("en", "dessert over 600 calories", PQ(category="Desserts", calories_min=600),
      "a MIN limit shrinks instead: 600 * 0.90 = 540"),
    C("en", "popular pizza under $15", PQ(category="Pizza", price_max=15, popular_only=True),
      "popular is KEPT while the price is relaxed"),

    "E2. NEAREST  (no margin helps: keep the other filters, rank candidates by distance to the limit)",
    C("en", "pizza under $1", PQ(category="Pizza", price_max=1), "cheapest pizzas come first"),
    C("ar", "بيتزا بأقل من دولار", PQ(category="Pizza", price_max=1)),
    C("en", "dessert over 5000 calories", PQ(category="Desserts", calories_min=5000),
      "for a MIN limit, the closest are the highest ones"),
    C("en", "pizza under 100 calories", PQ(category="Pizza", calories_max=100)),
    C("en", "dessert under $3 and under 100 calories", PQ(category="Desserts", price_max=3, calories_max=100),
      "price and calories distances are added"),
    C("en", "pizza under $1 without mozzarella", PQ(category="Pizza", price_max=1, exclude_ingredients=["mozzarella"]),
      "exclusions survive EVERY fallback step"),
    C("en", "spicy burger under $3", PQ(semantic_text="spicy", category="Burgers", price_max=3),
      "the text is kept: only burgers close enough to 'spicy' are ranked"),

    "E3. SOFT FILTERS DROPPED  (only when margins and nearest find nothing; the dropped words move into the text)",
    C("en", "dessert with cheese", PQ(category="Desserts", include_ingredients=["cheese"]),
      "ingredients dropped, 'Desserts cheese' becomes the semantic text"),
    C("ar", "حلوى بالجبنة", PQ(category="Desserts", include_ingredients=["cheese"])),
    C("en", "dessert with cheese under $1", PQ(category="Desserts", price_max=1, include_ingredients=["cheese"]),
      "two fallbacks in a row: ingredients dropped, then the price handled by 'nearest'"),

    "E4. SIMILARITY THRESHOLD",
    C("en", "sushi (similarity threshold 0.95)", PQ(semantic_text="sushi"),
      "nothing scores >= 0.95, so the same query is repeated without the threshold", thr=0.95),
]


# ---------------------------------------------------------------- explaining a ParsedQuery
ROLE = {
    "semantic_text":       "embedded -> vector search, ranks by meaning",
    "category":            "filter (must): primary_category ==",
    "price_min":           "filter (must): price >=",
    "price_max":           "filter (must): price <=",
    "calories_min":        "filter (must): calories >=",
    "calories_max":        "filter (must): calories <=",
    "include_ingredients": "filter (must): contains EACH of them",
    "exclude_ingredients": "filter (must_not): contains NONE of them",
    "popular_only":        "filter (must): popular == True",
    "order_by":            "sort key (alone: Qdrant sorts; with semantic_text: the relevant hits are re-sorted)",
    "order_dir":           "sort direction (with order_by)",
}
DEFAULTS = asdict(PQ())


def fmt(v):
    if isinstance(v, float):
        return f"{v:.2f}".rstrip("0").rstrip(".")
    return repr(v)


def show_fields(q):
    d = asdict(q)
    print("  QUERY FIELDS")
    for k, role in ROLE.items():
        used = (d["order_by"] is not None) if k == "order_dir" else d[k] != DEFAULTS[k]
        print(f"    {k:20} {fmt(d[k]):34} {role if used else '(unused)'}")


def diff(orig, q):
    a, b = asdict(orig), asdict(q)
    out = [f"{k}: {fmt(a[k])} -> {fmt(b[k])}" for k in a if a[k] != b[k]]
    return "; ".join(out) or "same query as the original"


# ---------------------------------------------------------------- trace every search step
TRACE, DEPTH = [], [0]
_search, _semantic, _filter = S.search, S.semantic_search, S.filter_search


def t_search(q, limit=10, score_threshold=None):
    DEPTH[0] += 1
    try:
        r = _search(q, limit, score_threshold)
    finally:
        DEPTH[0] -= 1
    TRACE.append(("search", q, r["mode"], len(r["hits"]), score_threshold))
    return r


def t_semantic(text, flt, limit, score_threshold=None, trim=True):
    h = _semantic(text, flt, limit, score_threshold, trim)
    if DEPTH[0] == 0:                      # called directly = the 'nearest' step collecting candidates
        TRACE.append(("pool-semantic", text, None, len(h), None))
    return h


def t_filter(q, flt, limit):
    h = _filter(q, flt, limit)
    if DEPTH[0] == 0:
        TRACE.append(("pool-filter", q, None, len(h), None))
    return h


S.search, S.semantic_search, S.filter_search = t_search, t_semantic, t_filter


def show_trace(orig):
    print("  SEARCH STEPS")
    for i, (kind, q, mode, n, thr) in enumerate(TRACE, 1):
        if kind == "search":
            extra = f" (score_threshold={thr})" if thr is not None else ""
            print(f"    {i}. search [{mode}]{extra} -> {n} hits | changes: {diff(orig, q)}")
        elif kind == "pool-semantic":
            print(f"    {i}. nearest: top {S.POOL} candidates by meaning of {q!r} (category/ingredients/exclusions kept, "
                  f"numeric limits removed) -> {n}, then sorted by distance to the limits")
        else:
            print(f"    {i}. nearest: top {S.POOL} candidates by filters only (numeric limits removed) -> {n}, "
                  f"then sorted by distance to the limits")


def main():
    cases = [c for c in CASES if isinstance(c, dict)]
    for c in cases:
        c["q"] = apply_default_order(c["q"])            # the same rule build_query applies in the real flow
    queries = {c["text"]: c["q"] for c in cases}
    S.parse_query = lambda text: queries[text]

    print("mode rule:  semantic_text + filters/order_by -> hybrid | semantic_text only -> semantic |"
          " filters/order_by only -> filters | nothing -> none")
    print(f"settings:   MARGINS={S.MARGINS}  REL_CUTOFF={S.REL_CUTOFF}  POOL={S.POOL}")

    n = 0
    for c in CASES:
        if isinstance(c, str):
            print("\n\n" + "#" * 110 + f"\n# {c}\n" + "#" * 110)
            continue
        n += 1
        TRACE.clear()
        print(f"\n{'=' * 110}\n[{c['lang']}] case {n}: {c['text']!r}")
        if c["note"]:
            print(f"  note: {c['note']}")
        show_fields(c["q"])
        print()
        r = S.search_always(c["text"], limit=5, score_threshold=c["thr"])
        show_trace(c["q"])
        print(f"  RESULT mode={r['mode']} margin={r['margin']} dropped={r['dropped']}")
        if r["fallback"]:
            print(f"    fallback: {r['fallback']}")
        for i, h in enumerate(r["hits"], 1):
            p, s = h["payload"], h["score"]
            score = f"  score={s:.3f}" if s is not None else ""
            print(f"    {i}. {p.get('name', '?'):30} ${p.get('price', 0):.2f}  {p.get('calories', 0)}cal{score}")
        if not r["hits"]:
            print("    (no results)")


if __name__ == "__main__":
    main()