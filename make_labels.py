#!/usr/bin/env python3
"""configs/labels/ablations.yaml -> configs/labels/<base>_<variant>.yaml (base + overrides). Never overwrites
an existing label yaml with different content (a label name is a promise about what it contains)."""
from pathlib import Path

import yaml

D = Path(__file__).parent / "configs" / "labels"
spec = yaml.safe_load(open(D / "ablations.yaml"))
for name, over in spec["variants"].items():
    assert "." not in name and "_" not in name, name
    label = f"{spec['base']}_{name}"
    text = yaml.safe_dump({"label": label, "base": spec["base"], **over}, sort_keys=False)
    f = D / f"{label}.yaml"
    if f.exists() and f.read_text() != text:
        raise SystemExit(f"{f} exists with different content")
    f.write_text(text)
    print(f.name)
