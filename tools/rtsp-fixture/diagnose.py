#!/usr/bin/env python3
"""Why did the track churn? Inter-frame gaps, fps and track lifetimes."""
import json
import statistics
import os
from pathlib import Path

FIX = Path(os.environ.get("ESK_FIXTURE_DIR", Path.home() / "edge-security-fixture"))
rows = json.loads((FIX / "e2e-out/detections-tap.json").read_text())

for key, v in rows.items():
    ts = [r["ts_ms"] for r in v]
    gaps = [b - a for a, b in zip(ts, ts[1:])]
    gs = sorted(gaps)
    print(f"\n=== {key}: {len(v)} msgs over {(ts[-1]-ts[0])/1000:.1f}s "
          f"= {len(v)/((ts[-1]-ts[0])/1000):.2f} msg/s")
    print(f"  inter-message gap ms: p50={statistics.median(gs):.0f} "
          f"p90={gs[int(.9*len(gs))]:.0f} p99={gs[int(.99*len(gs))]:.0f} max={gs[-1]:.0f}")
    print(f"  gaps > 750ms (track_max_lost_s): {sum(1 for g in gaps if g > 750)}")
    inf = [r["inference_time_ms"] for r in v if r.get("inference_time_ms")]
    pipe = [r["pipeline_ms"] for r in v if r.get("pipeline_ms")]
    if inf:
        i = sorted(inf)
        print(f"  inference_time_ms p50={statistics.median(i):.1f} p95={i[int(.95*len(i))]:.1f}")
    if pipe:
        p = sorted(pipe)
        print(f"  pipeline_ms      p50={statistics.median(p):.1f} p95={p[int(.95*len(p))]:.1f}")
    # track lifetimes
    tracks = {}
    for r in v:
        if r["track_id"]:
            tracks.setdefault(r["track_id"], []).append(r["ts_ms"])
    print(f"  distinct track_ids: {len(tracks)} -> "
          + ", ".join(f"{t}:{(x[-1]-x[0])/1000:.1f}s/{len(x)}f" for t, x in sorted(tracks.items())))
    # where the gaps land relative to a track change
    prev = None
    for a, b in zip(v, v[1:]):
        if a["track_id"] != b["track_id"] and a["track_id"] and b["track_id"]:
            print(f"    track {a['track_id']}->{b['track_id']} gap={b['ts_ms']-a['ts_ms']}ms "
                  f"cx {a['cx']:.3f}->{b['cx']:.3f}")
