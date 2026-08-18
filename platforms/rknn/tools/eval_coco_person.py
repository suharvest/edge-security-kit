#!/usr/bin/env python3
"""COCO person-class AP for the RKNPU2 .rknn models.

Three subcommands, because the three steps run on three different machines and
nothing but small files crosses between them:

``subset``   (workstation)  choose the evaluation images and copy them out
``infer``    (on the board)  run one .rknn over them, write COCO detections
``score``    (workstation)  pycocotools, overall and by GT box area

Why bother instead of eyeballing a demo video: int8 does not fail loudly. It
shifts the score distribution, and the first thing that goes is the smallest,
lowest-contrast targets -- which for a security camera is the far end of the
corridor, the part of the frame the product exists for. Overall AP hides that,
so ``score`` always prints the COCO area split (small < 32^2 px, medium
< 96^2 px, large above) and the small column is the one to read.

**The evaluation half is disjoint from the calibration half.** ``subset`` takes
person images with an odd ``image_id``; ``../calibration/build_calibration.py``
takes even ones. Same rule in both files, so a PTQ set can never leak into its
own test set.

Thresholds here are not the deployed thresholds. AP needs the full
precision/recall curve, so detection runs at ``--conf 0.001 --iou 0.65``; the
detector ships at 0.35/0.45. A model that only looks good at 0.35 is a model
whose margin has already gone.

    # workstation
    ./eval_coco_person.py subset --coco-root ~/data/coco --count 500 --out ~/data/esk-eval
    # on the board
    ./eval_coco_person.py infer --model models/yolov8n_zoo_int8.rk3588.rknn \
        --eval-dir ~/esk-eval --out ~/esk-eval/dets-C.json
    # workstation
    ./eval_coco_person.py score --coco-root ~/data/coco --eval-dir ~/data/esk-eval \
        --dets dets-A.json dets-B.json dets-C.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

PERSON_CATEGORY_ID = 1


# --------------------------------------------------------------------------- subset
def cmd_subset(args: argparse.Namespace) -> int:
    root = args.coco_root.expanduser()
    data = json.loads((root / "annotations" / "instances_val2017.json").read_text())
    with_person = {
        a["image_id"]
        for a in data["annotations"]
        if a["category_id"] == PERSON_CATEGORY_ID and not a.get("iscrowd")
    }
    meta = {i["id"]: i for i in data["images"]}
    # Odd ids evaluate, even ids calibrate. Sorted then strided, so --count can
    # be raised later and the smaller set stays a subset of the larger.
    eval_ids = sorted(i for i in with_person if i % 2 == 1)
    n = min(args.count, len(eval_ids))
    picked = [eval_ids[round(k * (len(eval_ids) - 1) / (n - 1))] for k in range(n)]

    out = args.out.expanduser()
    images = out / "images"
    if images.exists():
        shutil.rmtree(images)
    images.mkdir(parents=True)
    index = []
    for image_id in picked:
        info = meta[image_id]
        shutil.copy2(root / "val2017" / info["file_name"], images / info["file_name"])
        index.append(
            {
                "id": image_id,
                "file_name": info["file_name"],
                "width": info["width"],
                "height": info["height"],
            }
        )
    (out / "index.json").write_text(json.dumps(index, indent=1))
    print(f"eval subset: {len(index)} images (odd image_id, person-bearing) -> {out}")
    print(f"  population: {len(eval_ids)} of {len(with_person)} person images in val2017")
    print(f"  index: {out / 'index.json'}")
    return 0


# ---------------------------------------------------------------------------- infer
def cmd_infer(args: argparse.Namespace) -> int:
    import cv2
    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from esk_rknn.letterbox import LetterboxTransform, pad_into_canvas
    from esk_rknn.rknn_yolo import RKNNPersonDetector

    eval_dir = args.eval_dir.expanduser()
    index = json.loads((eval_dir / "index.json").read_text())
    model = RKNNPersonDetector(
        str(args.model),
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        input_size=args.input_size,
    )

    results = []
    times: list[float] = []
    for k, info in enumerate(index):
        image = cv2.imread(str(eval_dir / "images" / info["file_name"]))
        if image is None:
            raise SystemExit(f"unreadable: {info['file_name']}")
        height, width = image.shape[:2]
        tf = LetterboxTransform.for_source(width, height, args.input_size)
        scaled = cv2.resize(
            image, (tf.scaled_w, tf.scaled_h), interpolation=cv2.INTER_LINEAR
        )
        canvas = pad_into_canvas(
            cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB), args.input_size
        )
        detections = model.detect(canvas, tf)
        times.append(model.last_inference_ms)
        # COCO wants absolute xywh in the ORIGINAL image; Detection.box is
        # normalized xyxy against that same image, so this is only a scale.
        for det in sorted(detections, key=lambda d: -d.score)[:100]:
            x1, y1, x2, y2 = det.box
            results.append(
                {
                    "image_id": info["id"],
                    "category_id": PERSON_CATEGORY_ID,
                    "bbox": [
                        round(x1 * width, 2),
                        round(y1 * height, 2),
                        round((x2 - x1) * width, 2),
                        round((y2 - y1) * height, 2),
                    ],
                    "score": round(det.score, 5),
                }
            )
        if args.progress and (k + 1) % args.progress == 0:
            print(f"  {k + 1}/{len(index)} images, {len(results)} detections", flush=True)

    model.close()
    args.out.expanduser().write_text(json.dumps(results))
    times.sort()
    p = lambda q: times[min(len(times) - 1, int(q * len(times)))]  # noqa: E731
    print(
        json.dumps(
            {
                "model": str(args.model),
                "head": model.head,
                "images": len(index),
                "detections": len(results),
                "conf": args.conf,
                "iou": args.iou,
                "inference_ms_p50": round(p(0.50), 3),
                "inference_ms_p95": round(p(0.95), 3),
                "out": str(args.out),
            },
            indent=1,
        )
    )
    return 0


# ---------------------------------------------------------------------------- score
def cmd_score(args: argparse.Namespace) -> int:
    import contextlib
    import io

    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    eval_dir = args.eval_dir.expanduser()
    index = json.loads((eval_dir / "index.json").read_text())
    image_ids = sorted(i["id"] for i in index)

    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str((args.coco_root.expanduser() / "annotations"
                       / "instances_val2017.json")))

    # Ground-truth area census, so the per-bucket AP can be read against how
    # many boxes it is actually averaging over. A bucket with 40 instances is
    # noise, and the reader has to be able to tell.
    census = {"small": 0, "medium": 0, "large": 0}
    for ann in gt.loadAnns(gt.getAnnIds(imgIds=image_ids, catIds=[PERSON_CATEGORY_ID],
                                        iscrowd=False)):
        area = ann["area"]
        census["small" if area < 32 ** 2 else "medium" if area < 96 ** 2 else "large"] += 1

    rows = []
    for path in args.dets:
        path = Path(path)
        detections = json.loads((path if path.is_absolute() or path.exists()
                                 else eval_dir / path).read_text())
        if not detections:
            rows.append((path.name, [float("nan")] * 5))
            continue
        with contextlib.redirect_stdout(io.StringIO()):
            dt = gt.loadRes(detections)
            ev = COCOeval(gt, dt, "bbox")
            ev.params.imgIds = image_ids
            ev.params.catIds = [PERSON_CATEGORY_ID]
            ev.evaluate()
            ev.accumulate()
            ev.summarize()
        s = ev.stats
        # stats: 0 AP@.5:.95, 1 AP@.5, 2 AP@.75, 3 AP small, 4 AP medium, 5 AP large
        rows.append((path.name, [s[1], s[0], s[3], s[4], s[5]]))

    width = max(len(r[0]) for r in rows) + 2
    print(f"COCO val2017, person only, {len(image_ids)} images "
          f"(odd image_id half; calibration used the even half)")
    print(f"GT person instances by area: small={census['small']} "
          f"medium={census['medium']} large={census['large']}")
    print()
    print(f"{'detections file':<{width}}{'AP@.5':>9}{'AP@.5:.95':>11}"
          f"{'AP small':>10}{'AP med':>9}{'AP large':>10}")
    print("-" * (width + 49))
    for name, stats in rows:
        cells = "".join(f"{v:>{w}.4f}" for v, w in zip(stats, (9, 11, 10, 9, 10)))
        print(f"{name:<{width}}{cells}")
    return 0


# ------------------------------------------------------------------- domain subset
def cmd_subset_domain(args: argparse.Namespace) -> int:
    """Build an *unlabelled* surveillance holdout, disjoint from calibration.

    COCO cannot answer whether a surveillance-weighted calibration set helped,
    because COCO is the other set's home turf: an int8 model calibrated on 20
    generic COCO photos scores on COCO exactly as well as one calibrated on
    CCTV. The question only has meaning on the deployment domain, and the
    deployment domain has no boxes drawn on it.

    So the domain test is a fidelity test rather than an accuracy test: run the
    fp16 model over held-out surveillance frames, treat what it finds as the
    reference, and ask how much of it each int8 build reproduces. It measures
    the thing quantization is supposed to preserve, and it needs no labels.
    """
    import csv

    used = set()
    if args.exclude_manifest and args.exclude_manifest.exists():
        with args.exclude_manifest.open() as handle:
            used = {row["source_name"] for row in csv.DictReader(handle)}

    pool: list[Path] = []
    for directory in args.images_dir:
        pool += sorted(p for p in directory.expanduser().glob("*.jpg")
                       if p.name not in used)
    if not pool:
        raise SystemExit("no held-out images: everything is in the calibration set")
    n = min(args.count, len(pool))
    picked = [pool[round(k * (len(pool) - 1) / (n - 1))] for k in range(n)]

    out = args.out.expanduser()
    images = out / "images"
    if images.exists():
        shutil.rmtree(images)
    images.mkdir(parents=True)
    index = []
    for k, src in enumerate(picked):
        from PIL import Image

        with Image.open(src) as im:
            width, height = im.size
        dest = images / f"{k:04d}_{src.name}"
        shutil.copy2(src, dest)
        index.append({"id": k + 1, "file_name": dest.name,
                      "width": width, "height": height})
    (out / "index.json").write_text(json.dumps(index, indent=1))
    print(f"domain holdout: {len(index)} surveillance frames -> {out}")
    print(f"  pool after excluding {len(used)} calibration images: {len(pool)}")
    return 0


# -------------------------------------------------------------------- score-pseudo
def cmd_score_pseudo(args: argparse.Namespace) -> int:
    """Score detection files against another detection file taken as truth."""
    import contextlib
    import io

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    eval_dir = args.eval_dir.expanduser()
    index = json.loads((eval_dir / "index.json").read_text())
    reference = json.loads((eval_dir / args.reference).read_text())

    # The reference is thresholded at the *deployed* confidence: the question
    # is whether int8 still finds what the shipped fp16 detector would act on,
    # not whether it reproduces its 0.001 tail.
    kept = [d for d in reference if d["score"] >= args.reference_conf]
    gt_doc = {
        "info": {}, "licenses": [],
        "images": index,
        "categories": [{"id": PERSON_CATEGORY_ID, "name": "person"}],
        "annotations": [
            {
                "id": k + 1,
                "image_id": d["image_id"],
                "category_id": PERSON_CATEGORY_ID,
                "bbox": d["bbox"],
                "area": d["bbox"][2] * d["bbox"][3],
                "iscrowd": 0,
            }
            for k, d in enumerate(kept)
        ],
    }
    tmp = eval_dir / "_pseudo_gt.json"
    tmp.write_text(json.dumps(gt_doc))

    census = {"small": 0, "medium": 0, "large": 0}
    for ann in gt_doc["annotations"]:
        a = ann["area"]
        census["small" if a < 32 ** 2 else "medium" if a < 96 ** 2 else "large"] += 1

    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(tmp))
    image_ids = sorted(i["id"] for i in index)

    rows = []
    for name in args.dets:
        detections = json.loads((eval_dir / name).read_text())
        with contextlib.redirect_stdout(io.StringIO()):
            ev = COCOeval(gt, gt.loadRes(detections), "bbox")
            ev.params.imgIds = image_ids
            ev.params.catIds = [PERSON_CATEGORY_ID]
            ev.evaluate(); ev.accumulate(); ev.summarize()
        s = ev.stats
        rows.append((name, [s[1], s[0], s[3], s[4], s[5]]))

    width = max(len(r[0]) for r in rows) + 2
    print(f"agreement with {args.reference} @score>={args.reference_conf}, "
          f"{len(index)} held-out surveillance frames")
    print(f"reference boxes by area: small={census['small']} "
          f"medium={census['medium']} large={census['large']}")
    print()
    print(f"{'detections file':<{width}}{'AP@.5':>9}{'AP@.5:.95':>11}"
          f"{'AP small':>10}{'AP med':>9}{'AP large':>10}")
    print("-" * (width + 49))
    for name, stats in rows:
        cells = "".join(f"{v:>{w}.4f}" for v, w in zip(stats, (9, 11, 10, 9, 10)))
        print(f"{name:<{width}}{cells}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("subset", help="pick the evaluation images (workstation)")
    s.add_argument("--coco-root", type=Path, required=True)
    s.add_argument("--count", type=int, default=500)
    s.add_argument("--out", type=Path, required=True)
    s.set_defaults(func=cmd_subset)

    i = sub.add_parser("infer", help="run one .rknn over the subset (board)")
    i.add_argument("--model", required=True)
    i.add_argument("--eval-dir", type=Path, required=True)
    i.add_argument("--out", type=Path, required=True)
    i.add_argument("--conf", type=float, default=0.001)
    i.add_argument("--iou", type=float, default=0.65)
    i.add_argument("--input-size", type=int, default=640)
    i.add_argument("--progress", type=int, default=100)
    i.set_defaults(func=cmd_infer)

    c = sub.add_parser("score", help="pycocotools AP table (workstation)")
    c.add_argument("--coco-root", type=Path, required=True)
    c.add_argument("--eval-dir", type=Path, required=True)
    c.add_argument("--dets", nargs="+", required=True)
    c.set_defaults(func=cmd_score)

    d = sub.add_parser("subset-domain", help="unlabelled surveillance holdout")
    d.add_argument("--images-dir", type=Path, nargs="+", required=True)
    d.add_argument("--exclude-manifest", type=Path,
                   default=Path(__file__).resolve().parents[1]
                   / "calibration" / "calib_manifest.csv")
    d.add_argument("--count", type=int, default=300)
    d.add_argument("--out", type=Path, required=True)
    d.set_defaults(func=cmd_subset_domain)

    ps = sub.add_parser("score-pseudo", help="agreement against a reference model")
    ps.add_argument("--eval-dir", type=Path, required=True)
    ps.add_argument("--reference", required=True)
    ps.add_argument("--reference-conf", type=float, default=0.35)
    ps.add_argument("--dets", nargs="+", required=True)
    ps.set_defaults(func=cmd_score_pseudo)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
