#!/usr/bin/env python3
"""
Unpack Kemal's datasets_top_5_annotated.zip and deploy it as InternNav labels.

run_annotator_configured.py's assemble_dataset() prefers `source_dataset` text,
and the val_seen/train configs point source_dataset at the GT files — so the
packed train_v*/seen_v* carry GT text in instruction_text and the Gemma output
only in _generated_instruction.text. This swaps it back in.

  original/  : zip contents as-is (json.gz + manifest; *_meta/ skipped, same data
               is inside each json.gz)
  HABITAT_BASE/{split}/{split}_v{N}.json.gz :
    train, val_seen   instruction_text <- _generated_instruction.text
    val_seen v272     as-is (complete_auto_annotator output, no GT overwrite)
    val_unseen        as-is (ChronoNav GT + sibling subs)
  All get the builder's text normalization ("... . ") and the val_unseen_patched
  vocab, like the v218 labels. instruction_tokens stay [] (GT tokens come from a
  different vocab that gate5_tokenizer does not reproduce; InternVLA-N1 reads text).

Usage: python3 convert_top5_generated.py [--zip PATH]
"""
import argparse
import gzip
import json
import shutil
import zipfile
from pathlib import Path

from local_paths import CACHE_ROOT, HABITAT_BASE, VOCAB_SOURCE

ZIP = "/home/irteam/data-vol2/vln/datasets_top_5_annotated.zip"
ORIG = CACHE_ROOT / "kemal_top5" / "original"
VERSIONS = ["v264", "v272", "v273", "v277", "v278"]
SPLITS = {"train": "train", "seen": "val_seen", "unseen": "val_unseen"}


def normalize(text):  # = InternNav build_relabel_dataset.normalize
    t = text.strip()
    if t and t[-1] not in ".!?":
        t += "."
    return t + " "


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=ZIP)
    args = ap.parse_args()

    ORIG.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip) as z:
        for n in z.namelist():
            name = Path(n).name
            if n.endswith("/") or "_meta/" in n or (ORIG / name).exists():
                continue
            with z.open(n) as src, open(ORIG / name, "wb") as dst:
                shutil.copyfileobj(src, dst)

    vocab = json.load(gzip.open(VOCAB_SOURCE))["instruction_vocab"]
    for short, split in SPLITS.items():
        for v in VERSIONS:
            dest = HABITAT_BASE / split / f"{split}_{v}.json.gz"
            if dest.exists():  # never overwrite anything in the label tree
                print(f"  skip (exists) {dest}")
                continue
            d = json.load(gzip.open(ORIG / f"{short}_{v}.json.gz"))
            n_gen = 0
            for e in d["episodes"]:
                gen = (e.get("_generated_instruction") or {}).get("text")
                if short != "unseen" and gen:
                    e["instruction"]["instruction_text"] = gen
                    n_gen += 1
                e["instruction"]["instruction_text"] = normalize(e["instruction"]["instruction_text"])
                e["instruction"].setdefault("instruction_tokens", [])
            d["instruction_vocab"] = vocab
            with gzip.open(dest, "wt") as f:
                json.dump(d, f)
            print(f"  {dest.name:28} {len(d['episodes']):>6} eps  generated={n_gen}")


if __name__ == "__main__":
    main()
