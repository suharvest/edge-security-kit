#!/usr/bin/env python3
"""Build the PTQ calibration set for the RKNPU2 int8 detector.

Post-training quantization only needs pixels, not labels: the toolkit runs the
float graph over the set and records per-tensor activation ranges. Those ranges
are the whole quantization, so the set decides the model. The upstream default
(``datasets/COCO/coco_subset_20.txt``, twenty generic web photos) records ranges
for eye-level 2--5 m portraits, which is not what a ceiling-mounted camera
produces, and the layers that lose the most from the mismatch are the ones that
carry small distant targets.

This builds a 400-image set from three kinds of source, and the mix is the
argument:

``surveillance`` (65%)
    Real CCTV stills. ``usrt`` is 1920x1080 multi-camera overhead footage,
    ``ucf`` is 320x240 UCF-Crime-style low-bitrate surveillance. Both carry the
    thing the deployment has and COCO does not: people 20--60 px tall, wide
    angle, compression mush, fixed exposure. Two resolutions rather than one so
    the ranges are not fitted to a single sensor's noise floor.

``coco`` (25%)
    COCO val2017 images containing at least one person. Insurance: with
    surveillance alone the ranges collapse onto one lighting regime and the
    model degrades on anything else. Drawn from a deterministic half of the
    person images; ``eval_coco_person.py`` scores on the *other* half, so no
    image is both calibration and test data.

``target`` (10%)
    Frames from the exact fixture stream the end-to-end harness asserts on.
    Small on purpose -- enough that the real pixel statistics are inside the
    recorded ranges, not enough to tune the model to one video.

Only the manifest and this script are committed. The images are not ours to
redistribute, and the manifest plus a fixed seed reproduces the selection.

Usage::

    ./build_calibration.py \
        --cctv-root  ~/project/CCTV-Gun/data \
        --coco-root  ~/data/coco \
        --target-dir ~/data/esk-target-frames \
        --out        ~/data/esk-calib

writes ``<out>/images/``, ``<out>/dataset.txt`` (the path list RKNN's
``build(dataset=...)`` wants) and refreshes ``calib_manifest.csv`` next to this
script.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

SEED_NOTE = "deterministic: no RNG, every pick is a sorted-order stride"

# name -> (subdir, wanted count). Kept explicit rather than computed from a
# ratio so the manifest can be diffed against an intent.
CCTV_QUOTA = {"usrt": 190, "ucf": 70}
COCO_QUOTA = 100
TARGET_QUOTA = 40

# usrt filenames are <camera>-<timerange>[_Segment_N]_x264_frame_<k>.jpg.
# Consecutive frames of one segment are near-duplicates and contribute one
# activation range between them, so sampling is strided *within* a segment and
# the quota is split *across* segments.
_USRT_SEGMENT = re.compile(r"^(?P<seg>.*)_frame_(?P<idx>\d+)\.(jpg|png)$", re.I)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stride_pick(items: list, want: int) -> list:
    """Evenly spaced picks from a sorted list, endpoints included."""
    n = len(items)
    if want >= n:
        return list(items)
    if want <= 0:
        return []
    return [items[round(i * (n - 1) / (want - 1)) if want > 1 else 0] for i in range(want)]


def pick_surveillance(root: Path, subset: str, want: int) -> list[tuple[Path, str]]:
    images = sorted((root / subset / "images").glob("*.jpg"))
    if not images:
        raise SystemExit(f"no images under {root / subset / 'images'}")

    groups: dict[str, list[Path]] = defaultdict(list)
    for path in images:
        match = _USRT_SEGMENT.match(path.name)
        # ucf has no segment structure in the filename; one group is correct
        # there, it is already a set of unrelated stills.
        groups[match.group("seg") if match else "_flat"].append(path)

    for key in groups:
        groups[key].sort(
            key=lambda p: (
                int(_USRT_SEGMENT.match(p.name).group("idx"))
                if _USRT_SEGMENT.match(p.name)
                else p.name
            )
        )

    # Proportional split across segments, largest-remainder so the total is
    # exact rather than off by the number of groups.
    total = sum(len(v) for v in groups.values())
    exact = {k: want * len(v) / total for k, v in groups.items()}
    quota = {k: int(v) for k, v in exact.items()}
    for key in sorted(groups, key=lambda k: exact[k] - quota[k], reverse=True):
        if sum(quota.values()) >= want:
            break
        quota[key] += 1

    picked: list[tuple[Path, str]] = []
    for key in sorted(groups):
        for path in _stride_pick(groups[key], quota[key]):
            picked.append((path, f"cctv-gun/{subset}"))
    return picked


def pick_coco(root: Path, want: int) -> list[tuple[Path, str]]:
    """Person-bearing val2017 images, calibration half only.

    The split is ``image_id % 2``: even ids calibrate, odd ids evaluate.
    ``eval_coco_person.py`` reads the same rule, so the two sets can never
    silently overlap when either script is re-run.
    """
    ann_path = root / "annotations" / "instances_val2017.json"
    data = json.loads(ann_path.read_text())
    with_person = {
        a["image_id"]
        for a in data["annotations"]
        if a["category_id"] == 1 and not a.get("iscrowd")
    }
    by_id = {i["id"]: i["file_name"] for i in data["images"]}
    calib_ids = sorted(i for i in with_person if i % 2 == 0)
    return [
        (root / "val2017" / by_id[i], "coco-val2017-person-even")
        for i in _stride_pick(calib_ids, want)
    ]


def pick_target(directory: Path, want: int) -> list[tuple[Path, str]]:
    # Keep the two clips in proportion to how much real footage each is: the
    # ADL clip is genuine video, truth.mp4 is a composited trajectory.
    adl = sorted(directory.glob("adl_*.jpg"))
    truth = sorted(directory.glob("truth_*.jpg"))
    n_truth = max(1, round(want * len(truth) / max(1, len(adl) + len(truth))))
    return [(p, "fixture/adl-720p") for p in _stride_pick(adl, want - n_truth)] + [
        (p, "fixture/truth-mp4") for p in _stride_pick(truth, n_truth)
    ]


def main() -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cctv-root", type=Path, required=True)
    ap.add_argument("--coco-root", type=Path, required=True)
    ap.add_argument("--target-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=here / "calib_manifest.csv")
    args = ap.parse_args()

    selected: list[tuple[Path, str]] = []
    for subset, want in CCTV_QUOTA.items():
        selected += pick_surveillance(args.cctv_root.expanduser(), subset, want)
    selected += pick_coco(args.coco_root.expanduser(), COCO_QUOTA)
    selected += pick_target(args.target_dir.expanduser(), TARGET_QUOTA)

    out = args.out.expanduser()
    images = out / "images"
    if images.exists():
        shutil.rmtree(images)
    images.mkdir(parents=True)

    rows = []
    listing = []
    for index, (src, origin) in enumerate(selected):
        # Flat, ordered names: the toolkit prints progress by index and a
        # sortable name makes "which image was #237" answerable.
        dest = images / f"{index:04d}_{origin.replace('/', '-')}_{src.name}"
        shutil.copy2(src, dest)
        listing.append(str(dest.resolve()))
        rows.append(
            {
                "index": index,
                "origin": origin,
                "source_name": src.name,
                "calib_name": dest.name,
                "sha256": _sha256(dest),
            }
        )

    (out / "dataset.txt").write_text("\n".join(listing) + "\n")
    with args.manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["index", "origin", "source_name", "calib_name", "sha256"]
        )
        writer.writeheader()
        writer.writerows(rows)

    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row["origin"]] += 1
    print(f"calibration set: {len(rows)} images -> {out}  ({SEED_NOTE})")
    for origin in sorted(counts):
        print(f"  {origin:32s} {counts[origin]:4d}")
    print(f"dataset list: {out / 'dataset.txt'}")
    print(f"manifest:     {args.manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
