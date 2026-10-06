#!/usr/bin/env python3
"""
check_label.py — label json.gz 하나를 specs/ 의 기준으로 채점한다.

vln-annotator(tools/check_label.py, specs/, 2026-10-06)에서 복사. 기준 yaml 은 그대로이고
analyze_path 만 이 레포의 gate2_path.path_analyzer 로 바꿨다.

기준은 코드가 아니라 specs/label_quality.yaml + specs/provenance.yaml 에 있다.
두 레포가 같은 파일을 읽는다 (InternNav 쪽은 얇은 래퍼).

  python3 gate7_eval/check_label.py <label.json.gz> --gt <gt.json.gz>
  python3 gate7_eval/check_label.py --dir <data/vln/mp3d/r2r/v1> --label v218     # 3 split 전부 + G1

G7(공개 System2 zero-shot)은 GPU 가 필요하므로 여기서 하지 않는다. 안내만 출력한다.
"""
import argparse, gzip, json, os, re, sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml
from gate2_path.path_analyzer import analyze_path

SPECS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "specs")


def load_specs():
    with open(os.path.join(SPECS, "label_quality.yaml")) as f: q = yaml.safe_load(f)
    with open(os.path.join(SPECS, "provenance.yaml")) as f: p = yaml.safe_load(f)
    return q, p


def load(path):
    with gzip.open(path, "rt") as f:
        d = json.load(f)
    return {e["episode_id"]: e for e in d["episodes"]}


def text_of(ep):
    return ((ep.get("instruction") or {}).get("instruction_text") or "").strip()


# ── 지표 ─────────────────────────────────────────────────────────────────────
_DIST = re.compile(r"\b\d+(?:\.\d+)?\s*(?:m|meter|meters|metre|metres|ft|feet|foot|cm|step|steps)\b", re.I)
_STOP_GROUNDED = re.compile(r"\b(?:stop|wait)\b[^.]{0,20}?\b(?:near|at|by|in\s+front\s+of|beside|next\s+to)\s+the\b", re.I)
_TURN_DIR = re.compile(r"\bturn\s+(left|right)\b|\bbear\s+(left|right)\b|\b(left|right)\s+turn\b", re.I)


def content_vocab(texts, spec):
    stop = set(spec["content_word"]["stopwords"]); n = spec["content_word"]["min_length"]
    c = Counter()
    for t in texts:
        for w in re.findall(r"[a-z]+", t.lower()):
            if w not in stop and len(w) >= n: c[w] += 1
    return c


def jaccard(a, b):
    A, B = set(re.findall(r"[a-z]+", a.lower())), set(re.findall(r"[a-z]+", b.lower()))
    return len(A & B) / len(A | B) if A | B else 0.0


def geom_turns(ep):
    prims = analyze_path(ep["reference_path"], ep.get("start_rotation")).get("primitives", [])
    return [p["type"].split("_")[0] for p in prims if p["type"] in ("left_turn", "right_turn")]


def text_turns(t):
    return [next(g for g in m.groups() if g).lower() for m in _TURN_DIR.finditer(t)]


def direction_consistency(eps, flip=False):
    """마지막 회전 기준 일치율. flip=True 면 기하 부호를 뒤집는다."""
    ok = tot = 0
    for ep in eps.values():
        g, t = geom_turns(ep), text_turns(text_of(ep))
        if not g or not t: continue
        gl = g[-1]
        if flip: gl = "right" if gl == "left" else "left"
        tot += 1; ok += (gl == t[-1])
    return (ok / tot if tot else 0.0), tot


def classify(stats, pspec):
    for c in pspec["label_classes"]:
        w = c["when"]
        if not w: return c["name"]
        if "gt_jaccard_min" in w and stats["gt_jaccard"] >= w["gt_jaccard_min"]: return c["name"]
        if "unique_pct_max" in w and (stats["unique_pct"] < w["unique_pct_max"]
                                      or stats["content_vocab"] < w["content_vocab_max"]
                                      or stats["max_duplicate"] >= w["max_duplicate_min"]): return c["name"]
    return pspec["label_classes"][-1]["name"]


def measure(path, gt_path, split, q, p):
    eps = load(path); texts = [text_of(e) for e in eps.values() if text_of(e)]
    n = len(texts)
    cv = content_vocab(texts, p)
    gt = load(gt_path) if gt_path else {}
    ids = set(eps) & set(gt)
    gj = sum(jaccard(text_of(gt[i]), text_of(eps[i])) for i in ids) / len(ids) if ids else 0.0
    # 부호 보정: GT 가 더 높게 나오는 쪽을 양의 방향으로 본다
    flip = False
    if gt:
        a, _ = direction_consistency(gt, False); b, _ = direction_consistency(gt, True)
        flip = b > a
    dc, dc_n = direction_consistency(eps, flip)
    gt_dc = direction_consistency(gt, flip)[0] if gt else None
    s = {
        "n": n,
        "unique_pct": 100 * len(set(texts)) / n,
        "max_duplicate": Counter(texts).most_common(1)[0][1],
        "content_vocab": len(cv),
        "gt_jaccard": gj,
        "direction_consistency": dc, "direction_n": dc_n, "gt_direction_consistency": gt_dc,
        "flip": flip,
        "distance_mention_pct": 100 * sum(1 for t in texts if _DIST.search(t)) / n,
        "stop_grounded_pct": 100 * sum(1 for t in texts if _STOP_GROUNDED.search(t)) / n,
        "avg_words": sum(len(t.split()) for t in texts) / n,
    }
    gtv = p["gt_content_vocab"].get(split)
    s["content_vocab_ratio"] = len(cv) / gtv if gtv else None
    s["class"] = classify(s, p)
    return s


def gates(s, q):
    rows = []
    for g in q["gates"]:
        gid, m = g["id"], g.get("metric")
        if gid == "G1_split_consistency" or gid == "G7_zero_shot_system2":
            rows.append((gid, "—", "—", "skip", g.get("blocking", False))); continue
        v = s.get(m)
        if v is None: rows.append((gid, "—", "—", "n/a", g.get("blocking", False))); continue
        lo, hi = g.get("min"), g.get("max")
        ok = (lo is None or v >= lo) and (hi is None or v <= hi)
        bound = (f"{lo}~{hi}" if lo is not None and hi is not None
                 else (f">={lo}" if lo is not None else f"<={hi}"))
        rows.append((gid, f"{v:.3f}" if isinstance(v, float) else str(v), bound,
                     "PASS" if ok else "FAIL", g.get("blocking", False)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?")
    ap.add_argument("--gt")
    ap.add_argument("--split", choices=["train", "val_seen", "val_unseen"])
    ap.add_argument("--dir", help="data/vln/mp3d/r2r/v1 — 3개 split 을 한 번에 (G1 포함)")
    ap.add_argument("--label")
    a = ap.parse_args()
    q, p = load_specs()

    targets = []
    if a.dir and a.label:
        for sp in ("train", "val_seen", "val_unseen"):
            f = os.path.join(a.dir, sp, f"{sp}_{a.label}.json.gz")
            g = os.path.join(a.dir, sp, f"{sp}.json.gz")
            if os.path.exists(f): targets.append((f, g if os.path.exists(g) else None, sp))
            else: print(f"[!] 없음: {f}")
    elif a.path:
        sp = a.split or next((x for x in ("val_unseen", "val_seen", "train") if x in os.path.basename(a.path)), "val_unseen")
        targets.append((a.path, a.gt, sp))
    else:
        ap.error("path 또는 --dir/--label 필요")

    classes = {}
    for path, gtp, sp in targets:
        s = measure(path, gtp, sp, q, p)
        classes[sp] = s["class"]
        print(f"\n=== {os.path.basename(path)}  [{sp}]  n={s['n']}  provenance={s['class']}"
              + (f"  (기하 부호 반전 적용)" if s["flip"] else "") + " ===")
        print(f"  고유 {s['unique_pct']:.1f}%  최대중복 {s['max_duplicate']}  내용어 {s['content_vocab']}"
              f" (GT 대비 {s['content_vocab_ratio']:.2f})  avg_words {s['avg_words']:.1f}"
              f"  GT Jaccard {s['gt_jaccard']:.3f}")
        if s["gt_direction_consistency"] is not None:
            print(f"  방향 일치: label {s['direction_consistency']:.3f} / GT {s['gt_direction_consistency']:.3f}"
                  f"  (비교 가능 에피소드 {s['direction_n']})")
        print(f"  {'gate':<28}{'값':>10}{'기준':>10}  결과")
        for gid, v, b, r, blocking in gates(s, q):
            mark = "" if r == "PASS" else ("  ← BLOCKING" if blocking and r == "FAIL" else "")
            print(f"  {gid:<28}{v:>10}{b:>10}  {r}{mark}")

    if len(classes) == 3:
        same = len(set(classes.values())) == 1
        print(f"\n  G1_split_consistency        {classes}  {'PASS' if same else 'FAIL  ← BLOCKING'}")
    elif a.dir:
        print(f"\n  G1_split_consistency        split {len(classes)}/3 개만 존재  FAIL  ← BLOCKING")
    print("\n  G7_zero_shot_system2        공개 System2 를 이 label 의 val_unseen 으로 평가 (GPU ~1.7h, GT 기준 61.7)")


if __name__ == "__main__":
    main()
