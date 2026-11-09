
import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import retrieval.search as S
from parsing.query_llm import ParsedQuery as PQ, apply_default_order

LIMIT = 10
CAP_RECALL = True          # recall@k = hits / min(|expected|, k)


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower().replace("&", "and")).strip("-")


def nm(x):
    return x["name"] if isinstance(x, dict) else x


def dcg(gains):
    return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(gains))


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def fmt(x):
    return "-" if x is None else f"{x:.3f}"


def pearson(a, b):
    if len(a) < 3:
        return None
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return num / den if den else None


def evaluate(c, ks):
    r = S.search_always(c["query"], limit=LIMIT, score_threshold=c.get("threshold"))
    hits = r["hits"]
    ids = [slug(h["payload"].get("name", "")) for h in hits]
    scores = [h["score"] for h in hits]
    exp = c.get("expected", [])
    exp_ids = [slug(nm(x)) for x in exp]
    rel = set(exp_ids)
    cos = {slug(x["name"]): x["cosine"] for x in exp if isinstance(x, dict) and x.get("cosine") is not None}
    for x in c.get("top_exact", []):
        cos.setdefault(slug(x["name"]), x["cosine"])

    out = {"query": c["query"], "type": c["type"], "lang": c.get("lang", "en"), "mode": r["mode"], "ids": ids,
           "fallback": bool(r.get("fallback") or r.get("margin") or r.get("dropped")), "pairs": []}

    if rel:
        for k in ks:
            top = ids[:k]
            tp = sum(1 for i in top if i in rel)
            p = tp / len(top) if top else 0.0
            den = min(len(rel), k) if CAP_RECALL else len(rel)
            rc = tp / den
            out[f"P@{k}"], out[f"R@{k}"] = p, rc
            out[f"F1@{k}"] = 2 * p * rc / (p + rc) if p + rc else 0.0
            out[f"Hit@{k}"] = 1.0 if tp else 0.0
            ideal = dcg([1] * min(len(rel), k))
            out[f"nDCG@{k}"] = dcg([1 if i in rel else 0 for i in top]) / ideal
            if cos:
                ic = dcg(sorted(cos.values(), reverse=True)[:k])
                out[f"cosnDCG@{k}"] = dcg([cos.get(i, 0.0) for i in top]) / ic if ic else 0.0
        out["MRR"] = next((1 / (n + 1) for n, i in enumerate(ids) if i in rel), 0.0)
        h, ap = 0, 0.0
        for n, i in enumerate(ids):
            if i in rel:
                h += 1
                ap += h / (n + 1)
        out["MAP"] = ap / min(len(rel), LIMIT)
        # set-level metrics over EVERYTHING the engine returned (no k)
        tp_all = sum(1 for i in ids if i in rel)
        out["P"] = tp_all / len(ids) if ids else 0.0
        out["R"] = min(1.0, tp_all / min(len(rel), LIMIT))      # the engine returns at most LIMIT items
        out["F1"] = 2 * out["P"] * out["R"] / (out["P"] + out["R"]) if out["P"] + out["R"] else 0.0
        out["missing"] = [nm(x) for x in exp if slug(nm(x)) not in ids]
        out["extra"] = [h_["payload"].get("name") for h_, i in zip(hits, ids) if i not in rel]
        out["pairs"] = [(s, cos[i]) for i, s in zip(ids, scores) if s is not None and i in cos]
        out["residuals"] = [{"name": h_["payload"].get("name"), "estimated": cos[i], "returned": round(s, 4),
                             "residual": round(s - cos[i], 4)}
                            for h_, i, s in zip(hits, ids, scores) if s is not None and i in cos]
        if out["pairs"]:
            out["score_mae"] = mean([abs(s - c_) for s, c_ in out["pairs"]])
            out["score_bias"] = mean([s - c_ for s, c_ in out["pairs"]])

    ck, chk = c.get("checks", {}), {}
    if c.get("mode"):
        chk["mode"] = r["mode"] == c["mode"]
    if ck.get("order_by") and len(hits) > 1 and not ck.get("expect_fallback"):
        vals = [h_["payload"].get(ck["order_by"]) for h_ in hits]
        asc = ck.get("order_dir", "asc") == "asc"
        chk["order"] = all((a <= b) if asc else (a >= b) for a, b in zip(vals, vals[1:]))
    if "expect_fallback" in ck:
        chk["fallback"] = out["fallback"] == bool(ck["expect_fallback"])
    if ck.get("must_not"):
        bad = {slug(x) for x in ck["must_not"]} & set(ids)
        chk["must_not"] = not bad
        out["violations"] = sorted(bad)
    if ck.get("top1"):
        chk["top1"] = bool(ids) and ids[0] == slug(ck["top1"])
    if ck.get("expect_empty"):
        chk["empty"] = not ids
    out["checks"] = chk
    return out

def simple_report(results):
    rows = []
    for r in results:
        failed = [k for k, v in r["checks"].items() if not v]
        p, rc = r.get("P"), r.get("R")
        if failed or r.get("MRR") == 0:
            status = "FAIL"
        else:
            status = "OK"
        row = {"status": status, "query": r["query"], "type": r["type"], "lang": r["lang"],
               "returned": r["ids"][:5]}
        if p is not None:
            row.update(precision=round(p, 2), recall=round(rc, 2), mrr=round(r["MRR"], 2))
        if failed:
            row["failed_checks"] = failed
        if status != "OK" and r.get("missing"):
            row["missing"] = r["missing"][:5]      # first 5 only
        if status != "OK" and r.get("extra"):
            row["unexpected"] = r["extra"][:5]
        rows.append(row)
    order = {"FAIL": 0, "OK": 1}
    rows.sort(key=lambda x: order[x["status"]])    # problems first
    counts = {s: sum(1 for x in rows if x["status"] == s) for s in order}
    return {"summary": counts, "cases": rows}

def main():
    ap = argparse.ArgumentParser()
    here, root = Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]
    candidates = [here / "eval_queries.json", here / "eval_cases.json", root / "data" / "eval_queries.json",
                  root / "data" / "eval_cases.json", root / "data" / "eval_set.json"]
    default = next((p for p in candidates if p.exists()), candidates[2])
    ap.add_argument("--data", default=str(default))
    ap.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 10])
    ap.add_argument("--llm", action="store_true", help="use the real parse_query (LLM). Default: the 'parsed' object in each case")
    ap.add_argument("--verbose", action="store_true", help="full report (default: short summary)")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "eval_report"))
    a = ap.parse_args()
    ks = sorted(a.k)
    a.stub = not a.llm

    if not Path(a.data).exists():
        sys.exit(f"Dataset not found: {a.data}\nPut eval_queries.json in training/data/ or pass --data <path>")
    cases = json.loads(Path(a.data).read_text(encoding="utf-8"))
    print(f"dataset: {a.data}")
    if a.stub:
        lookup = {c["query"]: apply_default_order(PQ(**c.get("parsed", {}))) for c in cases}
        S.parse_query = lambda text: lookup[text]
    print(f"{len(cases)} cases")

    results = [evaluate(c, ks) for c in cases]

    if not a.verbose:
        k5 = 5 if 5 in ks else ks[-1]
        keys = ["P", "R", "F1", "Hit@1", "MRR", "MAP", f"nDCG@{k5}"]
        print("\nSUMMARY  (P / R / F1 = over everything returned)")
        groups = defaultdict(list)
        for r in results:
            if "MRR" in r:
                groups[r["type"]].append(r)
                groups["ALL"].append(r)
        for g in sorted(groups, key=lambda x: (x == "ALL", x)):
            print(f"  {g:9} n={len(groups[g]):<3}" + "  ".join(f"{c}={fmt(mean([r.get(c) for r in groups[g]]))}" for c in keys))
        agg = defaultdict(lambda: [0, 0])
        for r in results:
            for kk_, v in r["checks"].items():
                agg[kk_][1] += 1
                agg[kk_][0] += int(v)
        res_groups = defaultdict(list)
        for r in results:
            res_groups[r["type"]].extend(r["pairs"])
            res_groups["ALL"].extend(r["pairs"])
        if res_groups["ALL"]:
            print("\nRESIDUAL  (returned score - estimated cosine)")
            for g in sorted(res_groups, key=lambda x: (x == "ALL", x)):
                pr = res_groups[g]
                if pr:
                    print(f"  {g:9} n={len(pr):<4}mean={mean([x - y for x, y in pr]):+.3f}")
        failed = {k_: v for k_, v in agg.items() if v[0] < v[1]}
        if failed:                      # show the behaviour checks only when something breaks
            print("\nCHECKS FAILED")
            for kk_, (pp, tt) in sorted(failed.items()):
                print(f"  {kk_:10} {pp}/{tt}")
            for r in results:
                bad = [k_ for k_, v in r["checks"].items() if not v]
                if bad:
                    print(f"  - {r['query']!r} {bad}")
        Path(a.out + ".json").write_text(
            json.dumps(simple_report(results), ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\nsaved {a.out}.json")
        return

    cols = (["P", "R", "F1"] + [f"P@{k}" for k in ks] + [f"R@{k}" for k in ks] + [f"F1@{k}" for k in ks] + [f"Hit@{k}" for k in ks]
            + ["MRR", "MAP"] + [f"nDCG@{k}" for k in ks] + [f"cosnDCG@{k}" for k in ks])
    for key in ("type", "lang"):
        groups = defaultdict(list)
        for r in results:
            groups[r[key]].append(r)
            groups["ALL"].append(r)
        print(f"\n{'=' * 110}\nMETRICS BY {key.upper()}\n{'=' * 110}")
        for g in sorted(groups, key=lambda x: (x == "ALL", x)):
            rs = [r for r in groups[g] if "MRR" in r]
            vals = {c: mean([r.get(c) for r in rs]) for c in cols}
            print(f"[{g}] n={len(rs)}  " + "  ".join(f"{c}={fmt(v)}" for c, v in vals.items() if v is not None))

    pairs = [p for r in results for p in r["pairs"]]
    if pairs:
        corr = pearson([p[0] for p in pairs], [p[1] for p in pairs])
        print(f"\nreturned score vs dataset cosine: n={len(pairs)}  MAE={fmt(mean([abs(s - c_) for s, c_ in pairs]))}  "
              f"bias={fmt(mean([s - c_ for s, c_ in pairs]))}  pearson={fmt(corr)}")

    print(f"\n{'=' * 110}\nCHECKS (pass/total)\n{'=' * 110}")
    agg = defaultdict(lambda: [0, 0])
    for r in results:
        for k, v in r["checks"].items():
            agg[k][1] += 1
            agg[k][0] += int(v)
    for k, (p, t) in sorted(agg.items()):
        print(f"  {k:10} {p}/{t}")
    for r in results:
        bad = [k for k, v in r["checks"].items() if not v]
        if bad:
            extra = f" violations={r['violations']}" if r.get("violations") else ""
            print(f"  FAIL {r['query']!r} [{r['type']}] {bad} mode={r['mode']} returned={r['ids'][:5]}{extra}")

    kk = ks[min(2, len(ks) - 1)]
    print(f"\n{'=' * 110}\nWORST 10 RANKED CASES (lowest nDCG@{kk})\n{'=' * 110}")
    for r in sorted([r for r in results if "MRR" in r], key=lambda r: (r[f"nDCG@{kk}"], r["MRR"]))[:10]:
        print(f"  {r['query']!r} [{r['type']}] MRR={r['MRR']:.2f} nDCG@{kk}={r[f'nDCG@{kk}']:.2f}")
        print(f"      missing: {r['missing'][:6]}   unexpected: {r['extra'][:6]}")

    Path(a.out + ".json").write_text(
        json.dumps(simple_report(results), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nsaved {a.out}.json")


if __name__ == "__main__":
    main()