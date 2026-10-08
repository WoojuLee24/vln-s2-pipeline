#!/usr/bin/env python3
"""
Generate the v218_reproduce datasets for ALL splits: train, val_seen, val_unseen.

Difference from generate_v217_all_splits.py — two things:

  1. has_gate3 is True for EVERY split. The originals hardcoded it False for
     train/val_seen because no rendered frames existed for those splits, which is
     why train_v218 has only 27 distinct content words: with perframe={} the
     _has_context branch in metadata_reproducer.py:361 is never taken and the
     vocabulary collapses to _PATH_ONLY_LMS = ["room","hallway","corridor"].

  2. Each split reads its own gate3 cache directory. gate3 writes
     episode_{id:06d}.json with no split qualifier, so sharing one directory
     would let train silently overwrite val_unseen.

Everything else — the generator, the RNG seeding, the vocab source — is unchanged.
Writing into the GT tree (the originals' "deploy" step) is removed; GT is read-only.

Usage:
  python3 generate_v218_all_splits.py                      # all splits
  python3 generate_v218_all_splits.py --splits val_unseen  # one split
  VERSION=v219 python3 generate_v218_all_splits.py         # v219 → *_v219_reproduce
  VERSION picks the frozen generator metadata_reproducer_<VERSION>.py (v218 | v219);
  LABEL defaults to <VERSION>_reproduce.
"""

import argparse
import gzip
import importlib
import json
import os
import re
import sys
import time
from pathlib import Path

PIPELINE_ROOT = Path(__file__).parent
sys.path.insert(0, str(PIPELINE_ROOT))

VERSION = os.environ.get("VERSION", "v218")
assert VERSION in ("v218", "v219"), f"unsupported VERSION={VERSION!r}"
_reproducer = importlib.import_module(f"metadata_reproducer_{VERSION}")
reproduce_instruction, _make_rng = _reproducer.reproduce_instruction, _reproducer._make_rng
from local_paths import GT_PATHS, PERFRAME_DIR, LANDMARK_DIR, DATASETS_DIR, VOCAB_SOURCE

# LABEL is what InternNav sees: data/vln/mp3d/r2r/v1/<split>/<split>_<LABEL>.json.gz
LABEL = os.environ.get("LABEL", f"{VERSION}_reproduce")

OUT_DIR = DATASETS_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = {
    split: {
        "gt_path": GT_PATHS[split],
        "has_gate3": True,                       # ← the whole point of this file
        "perframe_dir": Path(os.environ.get("PERFRAME_ROOT", PERFRAME_DIR)) / split,
        "landmark_dir": LANDMARK_DIR / split,
        "out_name": f"{split}_{LABEL}.json.gz",
    }
    for split in ("train", "val_seen", "val_unseen")
}


def load_vocab():
    """Vocab comes from val_unseen_patched for every split — train/val_seen GT
    files carry an empty instruction_vocab, and habitat's VLNDatasetV1.from_json
    reads word_list unconditionally."""
    with gzip.open(VOCAB_SOURCE, "rt") as f:
        return json.load(f).get("instruction_vocab", {})


def generate_split(split_name: str, cfg: dict) -> list:
    print(f"\n[{split_name}] Loading GT from {cfg['gt_path']} ...")
    t0 = time.time()
    with gzip.open(cfg["gt_path"], "rt") as f:
        gt = json.load(f)
    episodes = gt["episodes"]
    print(f"[{split_name}] {len(episodes)} episodes | gate3 cache {cfg['perframe_dir']}")

    results = []
    missing_gate3 = 0
    initial_turns = 0

    for i, ep in enumerate(episodes):
        eid = ep.get("episode_id", i)

        if cfg["has_gate3"]:
            pf_path = cfg["perframe_dir"] / f"episode_{eid:06d}.json"
            lm_path = cfg["landmark_dir"] / f"episode_{eid:06d}.json"
            perframe = json.loads(pf_path.read_text()) if pf_path.exists() else {}
            landmark = json.loads(lm_path.read_text()) if lm_path.exists() else {}
            if not perframe:
                missing_gate3 += 1
            elif perframe.get("goal") is None:
                # goal frame dropped as black (run_gate3_perframe_v2 + rendered_frames_fix): no goal landmark
                perframe["goal"] = {}
        else:
            perframe = {}
            landmark = {}

        rng = _make_rng(eid)
        instr = reproduce_instruction(
            episode_id=eid,
            reference_path=ep["reference_path"],
            start_rotation=ep.get("start_rotation"),
            perframe=perframe,
            landmark=landmark,
            rng=rng,
        )
        if instr.startswith("Turn"):
            initial_turns += 1

        out_ep = dict(ep)
        out_ep["instruction"] = {"instruction_text": instr}
        results.append(out_ep)

        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{len(episodes)} ({time.time()-t0:.1f}s)")

    elapsed = time.time() - t0
    print(f"[{split_name}] Done: {len(results)} eps, {elapsed:.1f}s, "
          f"init_turns={initial_turns/len(results)*100:.1f}%")
    if missing_gate3:
        pct = missing_gate3 / len(results) * 100
        print(f"[{split_name}] WARNING: {missing_gate3} ({pct:.1f}%) episodes had NO "
              f"gate3 perframe and fell back to the room/hallway vocabulary.")
    return results, missing_gate3


_STOPWORDS = set("""a an the and or but if then of to in into on at by from with for up down over under
walk go turn stop wait move continue head proceed enter exit pass passing past through straight
left right forward around toward towards next near front behind side you your there here is are
it that this until once after before when where while slightly all keep take make""".split())

_VISUAL = set("""red blue green yellow white black brown grey gray orange purple pink beige tan silver
gold golden cream maroon navy wooden wood metal metallic glass marble tile tiled carpet carpeted
leather granite stone brick stainless wicker""".split())


def quality_report(episodes: list, split_name: str) -> dict:
    """The gate that matters: content-word variety. 27 means gate3 never arrived."""
    texts = [e["instruction"]["instruction_text"] for e in episodes]
    n = len(texts)
    content = set()
    visual_hits = 0
    for t in texts:
        words = re.findall(r"[a-z]+", t.lower())
        content |= {w for w in words if w not in _STOPWORDS and len(w) > 2}
        if _VISUAL & set(words):
            visual_hits += 1
    explicit = [len(re.findall(r"\bturn (?:left|right|around|slightly)\b", t, re.I)) for t in texts]
    stats = {
        "n": n,
        "unique_sentences": len(set(texts)),
        "unique_pct": round(100 * len(set(texts)) / n, 1),
        "content_words": len(content),
        "avg_words": round(sum(len(t.split()) for t in texts) / n, 1),
        "visual_pct": round(100 * visual_hits / n, 1),
        "avg_explicit_turns": round(sum(explicit) / n, 2),
    }
    print(f"\n[{split_name}] n={stats['n']}  고유문장 {stats['unique_sentences']} "
          f"({stats['unique_pct']}%)  내용어 {stats['content_words']}  "
          f"avg_words {stats['avg_words']}  시각어휘 {stats['visual_pct']}%  "
          f"explicit_turns {stats['avg_explicit_turns']}")
    if stats["content_words"] < 100:
        print(f"[{split_name}] *** FAIL: 내용어 {stats['content_words']} < 100 — "
              f"gate3 perframe 이 적용되지 않았다 ***")
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="train,val_seen,val_unseen")
    args = ap.parse_args()
    wanted = [s for s in args.splits.split(",") if s]

    print("=" * 70)
    print(f"MetadataReproducer[{_reproducer.__name__}] → {LABEL}  (gate3 enabled for ALL splits)")
    print("=" * 70)

    vocab = load_vocab()
    print(f"vocab: word_list={len(vocab.get('word_list', []))}")
    all_stats = {}

    for split_name in wanted:
        cfg = SPLITS[split_name]
        episodes, missing = generate_split(split_name, cfg)
        all_stats[split_name] = quality_report(episodes, split_name)
        all_stats[split_name]["missing_gate3"] = missing

        out_data = {
            "episodes": episodes,
            "instruction_vocab": vocab,
            "_generation_meta": {
                "label": LABEL,
                "mode": "metadata_reproducer + gate3 perframe (all splits)",
                "model": os.environ.get("VLLM_MODEL", "cyankiwi/gemma-4-31B-it-AWQ-4bit"),
                "split": split_name,
                "n_episodes": len(episodes),
                "n_missing_gate3": missing,
                "perframe_dir": str(cfg["perframe_dir"]),
                **all_stats[split_name],
            },
        }
        out_path = OUT_DIR / cfg["out_name"]
        with gzip.open(out_path, "wt") as f:
            json.dump(out_data, f)
        print(f"  Saved: {out_path} ({out_path.stat().st_size//1024} KB)")

    print("\n" + "=" * 70)
    print(f"SUMMARY — {LABEL}")
    for sn, s in all_stats.items():
        flag = "FAIL" if s["content_words"] < 100 else "ok"
        print(f"  {sn:<12} n={s['n']:>6}  내용어={s['content_words']:>5}  "
              f"고유={s['unique_pct']:>5}%  시각={s['visual_pct']:>5}%  [{flag}]")
    print("\nGT 기준: train 내용어 2228 / 고유 99.9% / 시각 22.8%")


if __name__ == "__main__":
    main()
