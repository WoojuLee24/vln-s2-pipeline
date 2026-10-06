"""
Machine-local path configuration.

The original scripts hardcoded the author's machine (/mnt/nvme0/..., /home/kemal/...).
Every value here is overridable by an environment variable; the defaults point at
this machine (internnav-train1-1-0).

Layout notes:
  HABITAT_BASE  mirrors the author's VLN-CE dataset dir. READ-ONLY — never write here.
                Only val_unseen_patched.json.gz carries instruction_vocab (2711 words);
                train/val_seen have an empty vocab, which is why load_vocab() in the
                generators pulls it from val_unseen_patched for every split.
  SCENES_ROOT   Matterport3D meshes. scene_id in the GT is "mp3d/<id>/<id>.glb",
                so this is the directory that *contains* mp3d/.
  CACHE_ROOT    Working cache, deliberately OUTSIDE the repo: .gitignore excludes
                outputs/, which is how the original gate3 cache was lost.
"""
import os
from pathlib import Path

HABITAT_BASE = Path(os.environ.get(
    "VLN_HABITAT_BASE", "/home/irteam/data-vol1/vln/mp3d/r2r/v1"))

SCENES_ROOT = Path(os.environ.get(
    "VLN_SCENES_ROOT", "/home/irteam/data-vol1/InternData-N1/scene_data"))

CACHE_ROOT = Path(os.environ.get(
    "VLN_CACHE_ROOT", "/home/irteam/data-vol1/vln/relabel_cache"))

# Per-split cache subtrees. gate3 writes episode_{id:06d}.json with no split
# qualifier, so the splits MUST stay in separate directories or train silently
# overwrites val_unseen wherever episode_ids collide.
FRAMES_DIR = CACHE_ROOT / "rendered_frames"      # / <split> / episode_XXXXXX /
PERFRAME_DIR = CACHE_ROOT / "gate3_perframe"     # / <split> / episode_XXXXXX.json
LANDMARK_DIR = CACHE_ROOT / "gate3_landmarks"    # / <split> / episode_XXXXXX.json
DATASETS_DIR = CACHE_ROOT / "datasets"

# GT file per split, exactly as the original generators referenced them.
GT_PATHS = {
    "train": HABITAT_BASE / "train" / "train.json.gz",
    "val_seen": HABITAT_BASE / "val_seen" / "val_seen.json.gz",
    "val_unseen": HABITAT_BASE / "val_unseen" / "val_unseen_patched.json.gz",
}
VOCAB_SOURCE = GT_PATHS["val_unseen"]
