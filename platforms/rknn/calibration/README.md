# PTQ calibration set

Post-training quantization needs pixels, not labels. The toolkit runs the float
graph over this set, records each tensor's activation range, and those ranges
*are* the int8 model. Get them wrong and nothing fails — the model just loses
things.

`build_calibration.py` assembles 400 images from three kinds of source. Only
this script and `calib_manifest.csv` are committed; the images belong to their
datasets and the manifest plus the fixed selection rule reproduces the set.

```bash
./build_calibration.py \
  --cctv-root  ~/project/CCTV-Gun/data \
  --coco-root  ~/data/coco \
  --target-dir ~/data/esk-target-frames \
  --out        ~/data/esk-calib
```

| origin | n | what it is |
|---|---:|---|
| `cctv-gun/usrt` | 190 | 1920×1080 real CCTV, several fixed cameras |
| `cctv-gun/ucf` | 70 | 320×240 low-bitrate surveillance stills |
| `coco-val2017-person-even` | 100 | person-bearing val2017, even `image_id` only |
| `fixture/adl-720p` | 30 | frames from the ADL clip the harness streams |
| `fixture/truth-mp4` | 10 | frames from the synthetic truth trajectory |

Selection is deterministic — no RNG, every pick is a stride through a sorted
list, and within `usrt` the quota is split across camera segments first so 190
consecutive frames of one corridor cannot stand in for 190 scenes.

`mgd/` is excluded: those are 512×512 web photographs, not camera output.

## The eval half is disjoint from the calibration half

COCO images are split on `image_id % 2`. Even ids calibrate here; odd ids are
what `../tools/eval_coco_person.py subset` scores on. The same rule is written
in both files, so re-running either cannot leak one set into the other.

The fixture frames are a deliberate exception and the one caveat in the
numbers: the end-to-end harness asserts on the same clip these 40 frames came
from. That is the right thing for a deployed model — calibrate on the scene you
will run in — but it means the harness result is not evidence about unseen
scenes, and the COCO and surveillance-holdout numbers are.

## What measuring it actually showed

The premise of this set was that a surveillance-weighted calibration would beat
the upstream default (20 generic COCO photos). A control model built from a
20-image generic COCO set was measured against it, and **there is no
difference worth having**:

| | AP@.5 | AP@.5:.95 | AP small | AP med | AP large |
|---|---:|---:|---:|---:|---:|
| int8, this 400-image set | 0.7784 | 0.5322 | 0.2726 | 0.6299 | 0.7708 |
| int8, 20 generic COCO images | 0.7785 | 0.5339 | **0.2795** | 0.6313 | 0.7708 |

The control is very slightly *ahead* on small targets. On a 300-frame
surveillance holdout scored against the fp16 model's own output, the two are
also level (mAP 0.9775 vs 0.9756).

The reason is visible in the data rather than in the argument. Of the boxes the
fp16 model finds on that surveillance holdout, **4 are small, 83 medium and 533
large**: the CCTV-Gun footage is close-range corridor cameras, not the distant
wide-angle regime the set was assembled to represent. See
`../fixtures/int8-vs-fp16-surveillance.jpg` — two people at three metres,
filling a third of the frame height.

So the honest reading is: this calibration set is more representative of
*deployment* than 20 web photos, and it costs nothing to keep, but it is not
what earned the accuracy. YOLOv8n's int8 loss on this graph is small
regardless of which of these two sets is used, and the small-target claim these
images were chosen to support is unproven — the source footage does not
contain small targets. A set that would test it has to come from cameras with
people at 15–40 m, which is not in any dataset currently on hand.
