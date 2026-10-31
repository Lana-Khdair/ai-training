from retrieval.search import search_always

queries = [
    # --- semantic only (no filters) ---
    "something spicy and cheesy",
    "crispy and crunchy",
    "light and fresh",
    "chessy",                                   # typo, should be fixed to "cheesy"

    # --- price ---
    "anything under $15",
    "less than 10 dollars",
    "between 8 and 14 dollars",
    "over $20",
    "cheapest",                                 # filters only, sorted by price

    # --- calories ---
    "under 300 calories",
    "low calorie",                              # sorted by calories
    "more than 500 calories",

    # --- category (use your real categories) ---
    "pizza",
    "burgers",
    "pizzaa",                                   # typo in category
    "desserts under $8",

    # --- include / exclude ingredients ---
    "pizza with mozzarella",
    "with chicken and cheese",
    "no nuts",
    "without alcohol",
    "dessert without sugar",
    "sugar free dessert",
    "gluten free",                              # known weak spot, see what happens
    "mushroom free pizza",

    # --- mixed ---
    "cheesy pizza under 300 calories no nuts",
    "popular pizza under $15",
    "spicy burger with cheese between 10 and 15 dollars",
    "vegan salad",

    # --- should trigger the fallback ---
    "pizza under $1",
    "dessert over 5000 calories",
    "vegan pizza with chicken",                 # contradiction

    # --- edge cases ---
    "hello",
    "show me please",
    "",
    "!!!",
]

def check(r):
    """Cheap sanity check: does every hit respect the parsed filters?"""
    q, problems = r["query"], []
    for h in r["hits"]:
        p = h["payload"]
        if not r.get("fallback"):               # filters were not loosened
            if q["price_max"] is not None and p["price"] is not None and p["price"] > q["price_max"]:
                problems.append(f"{p['name']}: price {p['price']} > {q['price_max']}")
            if q["category"] and p["primary_category"] != q["category"]:
                problems.append(f"{p['name']}: category {p['primary_category']}")
        bad = set(q["exclude_ingredients"]) & set(p.get("ingredients", []))
        if bad:                                  # exclusions must ALWAYS hold
            problems.append(f"{p['name']}: contains excluded {bad}")
    return problems

for text in queries:
    r = search_always(text, limit=5)
    print(f"\n{text!r} -> {r['mode']}  fallback={r.get('fallback')}  dropped={r.get('dropped')}")
    print(f"  semantic={r['query']['semantic_text']!r}  category={r['query']['category']}  "
          f"price=({r['query']['price_min']},{r['query']['price_max']})  "
          f"cal=({r['query']['calories_min']},{r['query']['calories_max']})  "
          f"inc={r['query']['include_ingredients']}  exc={len(r['query']['exclude_ingredients'])} items  "
          f"order={r['query']['order_by']}")
    for h in r["hits"]:
        p = h["payload"]
        print(f"  {p['name']:30} ${p['price']}  {p['calories']}cal  score={h['score']}")
    for problem in check(r):
        print(f"  !! {problem}")