#!/usr/bin/env python3
"""End-to-end assertion harness against the controlled ground-truth stream.

Flow
  1. log in to the hub, PUT rules for both streams, note the current max alert id
  2. tap the detections topic for both streams for OBSERVE_S seconds
  3. align the observed cx series against truth.json (proves the stream really is
     the designed trajectory), then derive the ground-truth crossing / entry
     instants in device-clock ms from the very detections the hub judged
  4. pull GET /api/alerts, match each NEW alert to its truth instant PER RULE
     (each rule has its own expected event sequence), and assert
  5. download snapshots, pull /api/devices, print everything raw
"""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import paho.mqtt.client as mqtt

# Paths and endpoints are overridable so this runs outside the machine it was
# written on. Defaults match the layout the acceptance run used.
FIX = Path(os.environ.get("ESK_FIXTURE_DIR", Path.home() / "edge-security-fixture"))
OUT = FIX / "e2e-out"
HUB = os.environ.get("ESK_HUB", "http://127.0.0.1:18080")
BASE = HUB + "/api"
MQTT_HOST = os.environ.get("ESK_MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.environ.get("ESK_MQTT_PORT", "1884"))
USER = os.environ.get("ESK_HUB_USER", "admin")
PW = os.environ.get("ESK_HUB_PASSWORD", "e2e-truth-run-2026")
OBSERVE_S = float(sys.argv[1]) if len(sys.argv) > 1 else 130.0

# Device/stream ids are overridable so the same assertions can be pointed at an
# accelerated platform without forking the harness. ESK_SKIP_ADL runs the truth
# stream alone, for a board hosting a single detector.
TRUTH_DEV = os.environ.get("ESK_TRUTH_DEVICE", "generic-truth")
TRUTH_STREAM = os.environ.get("ESK_TRUTH_STREAM", "cam-0")
ADL_DEV = os.environ.get("ESK_ADL_DEVICE", "generic-adl")
ADL_STREAM = os.environ.get("ESK_ADL_STREAM", "cam-1")
SKIP_ADL = os.environ.get("ESK_SKIP_ADL", "").lower() in ("1", "true", "yes")

truth = json.loads((FIX / "truth.json").read_text())
FPS, NF, CX = truth["fps"], truth["n_frames"], truth["cx_by_frame"]
LINE = truth["geometry"]["line"]
# The line x is overridable, and on a hardware-scaled platform it has to be.
# The detector's coordinates are quantized to the decoder's output grid: MPP/RGA
# emits 640 px wide for a 1280x720 source, so a published cx can only land on a
# multiple of 1/640 -- and 0.5 is exactly one of them (320/640). A track walking
# across a line at exactly x=0.5 therefore produces a sample sitting ON the line
# rather than straddling it, side() returns 0, and neither this harness nor the
# hub sees the sign flip that defines a crossing. Nudging the line off the grid
# makes the crossing unambiguous. See the RK platform README.
if os.environ.get("ESK_LINE_X"):
    _lx = float(os.environ["ESK_LINE_X"])
    LINE = {"start": [_lx, LINE["start"][1]], "end": [_lx, LINE["end"][1]]}
ZONE = truth["geometry"]["zone_points"]
DWELL = truth["geometry"]["dwell_seconds"]
ZX = (min(p[0] for p in ZONE), max(p[0] for p in ZONE))
ZY = (min(p[1] for p in ZONE), max(p[1] for p in ZONE))
COOLDOWN = 3.0
TOL_S = 0.5

TRUTH_RULES = {
    "zones": [{"id": "zone-dwell", "name": "zone-dwell", "points": ZONE,
               "dwell_seconds": DWELL}],
    "lines": [
        {"id": "line-any", "name": "line-any", "start": LINE["start"],
         "end": LINE["end"], "direction": "any"},
        {"id": "line-fwd", "name": "line-fwd", "start": LINE["start"],
         "end": LINE["end"], "direction": "forward"},
        {"id": "line-bwd", "name": "line-bwd", "start": LINE["start"],
         "end": LINE["end"], "direction": "backward"},
    ],
    "features": {"zone_detection": True, "loitering": True, "line_crossing": True},
    "cooldown": COOLDOWN,
}
# Second stream: the real ADL clip. The subject lies in the lower-left and
# barely moves, so only a zone rule can fire -- which is the point: it shows the
# two streams are judged independently off their own rule bodies.
ADL_RULES = {
    "zones": [{"id": "adl-zone", "name": "adl-zone",
               "points": [[0.02, 0.02], [0.98, 0.02], [0.98, 0.98], [0.02, 0.98]],
               "dwell_seconds": 10}],
    "lines": [{"id": "adl-line", "name": "adl-line", "start": [0.5, 0.05],
               "end": [0.5, 0.95], "direction": "any"}],
    "features": {"zone_detection": True, "loitering": True, "line_crossing": True},
    "cooldown": 30.0,
}
RULE_EXPECT = {"line-any": ("forward", "backward"), "line-fwd": ("forward",),
               "line-bwd": ("backward",)}

COOKIE = OUT / "cookies.txt"


def curl(method, path, body=None, raw_out=None):
    cmd = ["curl", "-sS", "-b", str(COOKIE), "-c", str(COOKIE), "-X", method,
           f"{BASE}{path}"]
    if body is not None:
        cmd += ["-H", "Content-Type: application/json", "-d", json.dumps(body)]
    if raw_out is not None:
        cmd += ["-o", str(raw_out), "-w", "%{http_code} %{content_type} %{size_download}"]
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def api(method, path, body=None):
    out = curl(method, path, body)
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return {"_raw": out}


class Tap:
    def __init__(self):
        self.rows: dict[tuple[str, str], list[dict]] = {}
        self.lock = threading.Lock()

    def on_msg(self, _c, _u, msg):
        try:
            p = json.loads(msg.payload)
        except Exception:
            return
        dets = [d for d in p.get("detections") or [] if int(d.get("track_id", 0)) >= 1]
        best = max(dets, key=lambda d: d.get("score", 0)) if dets else None
        with self.lock:
            self.rows.setdefault((p["device_id"], p["stream_id"]), []).append({
                "frame_id": p["frame_id"], "ts_ms": p["timestamp"],
                "tap_recv_ms": time.time() * 1000.0,
                "n": len(p.get("detections") or []), "n_tracked": len(dets),
                "cx": best["bbox"][0] if best else None,
                "cy": best["bbox"][1] if best else None,
                "track_id": int(best["track_id"]) if best else None,
                "score": best.get("score") if best else None,
                "pipeline_ms": p.get("pipeline_ms"),
                "inference_time_ms": p.get("inference_time_ms"),
            })


def align(rows):
    pts = [(r["ts_ms"], r["cx"]) for r in rows if r["cx"] is not None]
    if len(pts) < 50:
        return {"ok": False, "n": len(pts)}
    t_start = pts[0][0]
    period_ms = NF / FPS * 1000.0

    def err_for(t0):
        e = 0.0
        for ts, cx in pts:
            fr = ((ts - t0) / 1000.0 * FPS) % NF
            i = int(fr)
            frac = fr - i
            e += abs(cx - (CX[i] * (1 - frac) + CX[(i + 1) % NF] * frac))
        return e / len(pts)

    best = min(((t_start - off, err_for(t_start - off))
                for off in range(0, int(period_ms), 5)), key=lambda p: p[1])
    t0, mean_err = best
    res = []
    for ts, cx in pts:
        fr = ((ts - t0) / 1000.0 * FPS) % NF
        i = int(fr)
        frac = fr - i
        res.append(abs(cx - (CX[i] * (1 - frac) + CX[(i + 1) % NF] * frac)))
    res.sort()
    return {"ok": True, "n": len(pts), "t0_ms": t0, "period_ms": period_ms,
            "mean_abs_cx_err": round(mean_err, 5),
            "p95_abs_cx_err": round(res[int(0.95 * len(res))], 5),
            "max_abs_cx_err": round(res[-1], 5)}


LINE_X = LINE["start"][0]


def side(px):
    s, e = LINE["start"], LINE["end"]
    cross = (e[0] - s[0]) * (0.5 - s[1]) - (e[1] - s[1]) * (px - s[0])
    return 1 if cross > 0 else (-1 if cross < 0 else 0)


def in_zone(px, py):
    return ZX[0] <= px <= ZX[1] and ZY[0] <= py <= ZY[1]


def observed_events(rows):
    seq = [r for r in rows if r["cx"] is not None]
    out, inside_state = [], {}
    # A track already alive when the tap opened has a history the harness cannot
    # see: if its first observed detection is inside the zone we cannot tell an
    # entry from a continuation, and the hub -- which HAS the history -- will not
    # re-fire. That first track's pseudo-entry is an observation-window artifact,
    # so it is excluded. Tracks born later inside the zone are real entries for
    # both sides (a fresh track_id has no zone state on either).
    preexisting = seq[0]["track_id"] if seq else None
    for prev, cur in zip(seq, seq[1:]):
        if prev["track_id"] != cur["track_id"]:
            continue
        sp, sc = side(prev["cx"]), side(cur["cx"])
        direction = "forward" if (sp > 0 and sc < 0) else (
            "backward" if (sp < 0 and sc > 0) else None)
        if not direction:
            continue
        frac = (LINE_X - prev["cx"]) / (cur["cx"] - prev["cx"])
        out.append({"event_type": "line_cross", "direction": direction,
                    "truth_ts_ms": prev["ts_ms"] + frac * (cur["ts_ms"] - prev["ts_ms"]),
                    "frame_id": cur["frame_id"], "track_id": cur["track_id"],
                    "cx_prev": prev["cx"], "cx_curr": cur["cx"]})
    for r in seq:
        tid = r["track_id"]
        now, was = in_zone(r["cx"], r["cy"]), inside_state.get(tid, False)
        first_sight = tid not in inside_state
        if now and not was and not (first_sight and tid == preexisting):
            out.append({"event_type": "zone_enter", "truth_ts_ms": float(r["ts_ms"]),
                        "frame_id": r["frame_id"], "cx": r["cx"], "cy": r["cy"],
                        "track_id": tid})
            out.append({"event_type": "loitering",
                        "truth_ts_ms": float(r["ts_ms"]) + DWELL * 1000.0,
                        "frame_id": r["frame_id"], "entered_ts_ms": float(r["ts_ms"]),
                        "track_id": tid, "expect_dwell_s": DWELL})
        inside_state[tid] = now
    out.sort(key=lambda e: e["truth_ts_ms"])
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    COOKIE.unlink(missing_ok=True)

    print("== login")
    print(json.dumps(api("POST", "/auth/login", {"username": USER, "password": PW})))

    print("== PUT rules")
    targets = [(TRUTH_DEV, TRUTH_STREAM, TRUTH_RULES)]
    if not SKIP_ADL:
        targets.append((ADL_DEV, ADL_STREAM, ADL_RULES))
    for dev, st, body in targets:
        r = api("PUT", f"/rules/{dev}/{st}", body)
        print(f"  {dev}/{st} rev={r.get('rev')} persisted_ms={r.get('persisted_ms')}")
    (OUT / "rules-truth.json").write_text(json.dumps(TRUTH_RULES, indent=2))
    (OUT / "rules-adl.json").write_text(json.dumps(ADL_RULES, indent=2))

    pre = json.loads(curl("GET", "/alerts?limit=1")).get("alerts", [])
    baseline_id = pre[0]["id"] if pre else 0
    print(f"== baseline max alert id = {baseline_id} (only newer alerts are asserted)")

    print("== tapping detections for %.0f s" % OBSERVE_S)
    tap = Tap()
    cli = mqtt.Client(client_id="e2e-tap")
    cli.on_message = tap.on_msg
    cli.connect(MQTT_HOST, MQTT_PORT, 30)
    cli.subscribe("sensecraft/security/+/detections/+", 0)
    cli.loop_start()
    t_begin = time.time() * 1000.0
    time.sleep(OBSERVE_S)
    cli.loop_stop()
    cli.disconnect()

    with tap.lock:
        rows = {f"{k[0]}/{k[1]}": v for k, v in tap.rows.items()}
    (OUT / "detections-tap.json").write_text(json.dumps(rows))
    for k, v in rows.items():
        ts = [r["ts_ms"] for r in v]
        cxs = [r["cx"] for r in v if r["cx"] is not None]
        gaps = sorted(b - a for a, b in zip(ts, ts[1:]))
        tids = sorted({r["track_id"] for r in v if r["track_id"]})
        print(f"  {k}: {len(v)} msgs / {(ts[-1]-ts[0])/1000:.1f}s = "
              f"{len(v)/((ts[-1]-ts[0])/1000):.2f} msg/s, tracked={len(cxs)}, "
              f"gap p50={gaps[len(gaps)//2]}ms max={gaps[-1]}ms, track_ids={tids}"
              + (f", cx {min(cxs):.3f}..{max(cxs):.3f}" if cxs else ""))

    trows = rows.get(f"{TRUTH_DEV}/{TRUTH_STREAM}", [])
    print("\n== trajectory alignment against truth.json")
    al = align(trows)
    print(json.dumps(al, indent=2))
    (OUT / "alignment.json").write_text(json.dumps(al, indent=2))

    print("\n== observed ground-truth instants (from the detections the hub judged)")
    obs = observed_events(trows)
    for e in obs:
        print(f"  t+{(e['truth_ts_ms'] - t_begin) / 1000:7.3f}s {e['event_type']:11s} "
              f"{e.get('direction') or '':8s} track={e.get('track_id')} "
              f"frame_id={e.get('frame_id')}")
    (OUT / "observed-events.json").write_text(json.dumps(obs, indent=2))

    print("\n== GET /api/alerts (raw)")
    alerts_raw = curl("GET", "/alerts?limit=500")
    (OUT / "alerts.json").write_text(alerts_raw)
    alerts = json.loads(alerts_raw).get("alerts", [])
    new_alerts = [a for a in alerts if a["id"] > baseline_id]
    (OUT / "alerts-new.json").write_text(json.dumps({"alerts": new_alerts}, indent=2))
    print(json.dumps({"alerts": new_alerts}, indent=2))

    print("\n== GET /api/devices (raw)")
    dev_raw = curl("GET", "/devices")
    (OUT / "devices.json").write_text(dev_raw)
    print(dev_raw)
    print("\n== GET /api/health (raw)")
    print(curl("GET", "/health"))

    # ------------------------------------------------ matching / assertions
    tal = [a for a in new_alerts if a["device_id"] == TRUTH_DEV]
    fails: list[str] = []
    matched: list[dict] = []

    # line rules: one independent expected sequence per rule_name
    for rule, wanted_dirs in RULE_EXPECT.items():
        want = [e for e in obs if e["event_type"] == "line_cross"
                and e["direction"] in wanted_dirs]
        got = sorted([a for a in tal if a["rule_name"] == rule], key=lambda x: x["id"])
        if len(got) != len(want):
            fails.append(f"{rule}: {len(got)} alerts for {len(want)} matching truth "
                         f"crossings (truth dirs="
                         f"{[e['direction'] for e in want]}, alert dirs="
                         f"{[a['direction'] for a in got]})")
        for a, e in zip(got, want):
            err = (a["received_ms"] - e["truth_ts_ms"]) / 1000.0
            m = {"id": a["id"], "event_type": a["event_type"], "rule_name": rule,
                 "direction": a["direction"], "expect_direction": e["direction"],
                 "dwell_s": a["dwell_s"], "snapshot_state": a["snapshot_state"],
                 "track_id": a["track_id"], "truth_ts_ms": e["truth_ts_ms"],
                 "alert_received_ms": a["received_ms"], "err_s": round(err, 4),
                 "truth_frame_id": e["frame_id"], "role": "first"}
            matched.append(m)
            if abs(err) > TOL_S:
                fails.append(f"{rule} alert {a['id']}: err {err:.3f}s > {TOL_S}s")
            if a["direction"] != e["direction"]:
                fails.append(f"{rule} alert {a['id']}: direction {a['direction']} "
                             f"!= truth {e['direction']}")
        # the direction filter itself
        for a in got:
            if a["direction"] not in wanted_dirs:
                fails.append(f"{rule} configured for {wanted_dirs} fired alert "
                             f"{a['id']} with direction={a['direction']}")

    # zone_enter: matched per track_id, nearest unused truth entry. Order-based
    # zip cannot be used because the tap window truncates the first and last
    # track's event series at both ends.
    z_want = [e for e in obs if e["event_type"] == "zone_enter"]
    z_got = sorted([a for a in tal if a["event_type"] == "zone_enter"],
                   key=lambda x: x["id"])
    # Counts are only enforced for tracks whose whole life is inside the window.
    seen_tracks = [r["track_id"] for r in trows if r["track_id"]]
    edge_tracks = {seen_tracks[0], seen_tracks[-1]} if seen_tracks else set()
    for tid in sorted({e["track_id"] for e in z_want} | {a["track_id"] for a in z_got}):
        if tid in edge_tracks:
            continue
        nw = len([e for e in z_want if e["track_id"] == tid])
        ng = len([a for a in z_got if a["track_id"] == tid])
        if nw != ng:
            fails.append(f"zone_enter track {tid}: {ng} alerts for {nw} truth entries")
    used_z: set[int] = set()
    pairs = []
    for a in z_got:
        cands = [(i, e) for i, e in enumerate(z_want)
                 if i not in used_z and e["track_id"] == a["track_id"]]
        if not cands:
            fails.append(f"zone_enter alert {a['id']} (track {a['track_id']}): "
                         f"no unused truth entry for that track")
            continue
        i, e = min(cands, key=lambda p: abs(p[1]["truth_ts_ms"] - a["received_ms"]))
        used_z.add(i)
        pairs.append((a, e))
    for a, e in pairs:
        err = (a["received_ms"] - e["truth_ts_ms"]) / 1000.0
        matched.append({"id": a["id"], "event_type": "zone_enter",
                        "rule_name": a["rule_name"], "direction": None,
                        "dwell_s": None, "snapshot_state": a["snapshot_state"],
                        "track_id": a["track_id"], "truth_ts_ms": e["truth_ts_ms"],
                        "alert_received_ms": a["received_ms"], "err_s": round(err, 4),
                        "truth_frame_id": e["frame_id"], "role": "first"})
        if abs(err) > TOL_S:
            fails.append(f"zone_enter alert {a['id']}: err {err:.3f}s > {TOL_S}s")

    # loitering: one first-fire per ZONE ENTRY, not per track.
    #
    # The obvious version keys expected loitering by track_id and calls the
    # first alert for that track the one under test. That silently assumes the
    # tracker keeps re-issuing IDs -- true of the CPU detector, whose frame gaps
    # expire tracks every loop. A detector with stable tracking (the RK platform
    # holds one track_id across the whole run) has ONE track with MANY entries,
    # so keying by track kept only the last entry and matched the earliest alert
    # against it, producing a ~-124 s "error" on a correct alert.
    l_want_by_track: dict[int, list[dict]] = {}
    for e in obs:
        if e["event_type"] == "loitering":
            l_want_by_track.setdefault(e["track_id"], []).append(e)
    for entries in l_want_by_track.values():
        entries.sort(key=lambda e: e["truth_ts_ms"])

    l_all = sorted([a for a in tal if a["event_type"] == "loitering"],
                   key=lambda x: x["id"])
    l_by_track: dict[int, list[dict]] = {}
    for a in l_all:
        l_by_track.setdefault(a["track_id"], []).append(a)

    tol_ms = TOL_S * 1000.0
    for tid, alerts in sorted(l_by_track.items()):
        wants = l_want_by_track.get(tid, [])
        unused = sorted(alerts, key=lambda x: x["received_ms"])
        for e in wants:
            # The first fire for this entry is the earliest alert at or after
            # the instant dwell_seconds elapsed.
            cands = [a for a in unused if a["received_ms"] >= e["truth_ts_ms"] - tol_ms]
            if not cands:
                fails.append(f"loitering: no alert for track {tid} entry at "
                             f"{e['truth_ts_ms']}")
                continue
            a = cands[0]
            unused.remove(a)
            err = (a["received_ms"] - e["truth_ts_ms"]) / 1000.0
            matched.append({"id": a["id"], "event_type": "loitering",
                            "rule_name": a["rule_name"], "direction": None,
                            "dwell_s": a["dwell_s"],
                            "snapshot_state": a["snapshot_state"],
                            "track_id": tid, "truth_ts_ms": e["truth_ts_ms"],
                            "alert_received_ms": a["received_ms"],
                            "err_s": round(err, 4),
                            "truth_frame_id": e["frame_id"], "role": "first"})
            if abs(err) > TOL_S:
                fails.append(f"loitering alert {a['id']} (track {tid}): first fire "
                             f"err {err:.3f}s > {TOL_S}s")
            if a["dwell_s"] is None or abs(a["dwell_s"] - DWELL) > 1.0:
                fails.append(f"loitering alert {a['id']}: dwell_s={a['dwell_s']} "
                             f"not within 1s of {DWELL}")
            # nothing may fire before the dwell has elapsed
            early = a["received_ms"] - e["entered_ts_ms"]
            if early < DWELL * 1000.0 - tol_ms:
                fails.append(f"loitering alert {a['id']} fired {early/1000:.3f}s after "
                             f"entry, before dwell_seconds={DWELL}")

        earliest = wants[0]["truth_ts_ms"] if wants else None
        for a in unused:
            if earliest is not None and a["received_ms"] < earliest - tol_ms:
                # Fired for an entry that happened before the tap opened: the hub
                # knows that entry instant and the harness does not, so the dwell
                # cannot be checked. With a stable track_id this is normal at the
                # start of every window, not just for a track at the edge.
                role = "skipped-entry-predates-tap"
            elif not wants:
                if tid not in edge_tracks:
                    fails.append(f"loitering alert {a['id']} for track {tid} "
                                 f"with no truth entry")
                    continue
                role = "skipped-entry-predates-tap"
            else:
                # The rule re-fires every cooldown while the target stays inside,
                # by design; reported but not matched to a fresh truth instant.
                role = "repeat-by-design"
            matched.append({"id": a["id"], "event_type": "loitering",
                            "rule_name": a["rule_name"], "direction": None,
                            "dwell_s": a["dwell_s"],
                            "snapshot_state": a["snapshot_state"],
                            "track_id": a["track_id"], "truth_ts_ms": None,
                            "alert_received_ms": a["received_ms"], "err_s": None,
                            "truth_frame_id": None, "role": role})
    matched.sort(key=lambda m: m["id"])
    (OUT / "matched.json").write_text(json.dumps(matched, indent=2))

    print("\n== MATCH TABLE: truth instant vs alert received_ms")
    print(f"{'id':>4} {'event':11s} {'rule':10s} {'dir':9s} {'exp_dir':9s} "
          f"{'dwell_s':>8s} {'err_s':>8s} {'trk':>4s} {'snapshot':9s} role")
    for m in matched:
        print(f"{m['id']:>4} {m['event_type']:11s} {m['rule_name']:10s} "
              f"{str(m['direction']):9s} {str(m.get('expect_direction')):9s} "
              f"{str(m['dwell_s']):>8s} {str(m['err_s']):>8s} "
              f"{m['track_id']:>4} {m['snapshot_state']:9s} {m['role']}")

    print("\n== DIRECTION FILTER EVIDENCE")
    tf = [e for e in obs if e["event_type"] == "line_cross" and e["direction"] == "forward"]
    tb = [e for e in obs if e["event_type"] == "line_cross" and e["direction"] == "backward"]
    print(f"  truth crossings: forward={len(tf)} backward={len(tb)}")
    for rule in RULE_EXPECT:
        got = [a for a in tal if a["rule_name"] == rule]
        dirs = sorted({a["direction"] for a in got})
        print(f"  {rule:9s} direction={TRUTH_RULES['lines'][[l['id'] for l in TRUTH_RULES['lines']].index(rule)]['direction']:8s} "
              f"alerts={len(got)} ids={[a['id'] for a in got]} directions_seen={dirs}")
    wrong_fwd = [a["id"] for a in tal if a["rule_name"] == "line-fwd"
                 and a["direction"] != "forward"]
    wrong_bwd = [a["id"] for a in tal if a["rule_name"] == "line-bwd"
                 and a["direction"] != "backward"]
    print(f"  line-fwd alerts with a backward crossing: {wrong_fwd or 'NONE'}")
    print(f"  line-bwd alerts with a forward crossing:  {wrong_bwd or 'NONE'}")
    if wrong_fwd:
        fails.append(f"line-fwd fired on a backward crossing: {wrong_fwd}")
    if wrong_bwd:
        fails.append(f"line-bwd fired on a forward crossing: {wrong_bwd}")

    # latency: only first-fire matches, each tied to a distinct truth instant
    lat = [m["err_s"] * 1000.0 for m in matched
           if m["role"] == "first" and m["err_s"] is not None]
    stats = {}
    if lat:
        s = sorted(lat)
        stats = {"n": len(s), "p50_ms": round(statistics.median(s), 1),
                 "p95_ms": round(s[min(len(s) - 1, int(0.95 * len(s)))], 1),
                 "min_ms": round(s[0], 1), "max_ms": round(s[-1], 1),
                 "method": "alert.received_ms - the ground-truth instant, where the "
                           "instant is the device-clock timestamp of the crossing "
                           "(linear interpolation between the two straddling "
                           "detections) or of the entry frame; loitering uses "
                           "entry + dwell_seconds. Detector capture->publish is "
                           "NOT included, see detector_pipeline_ms below."}
    pipe = sorted(r["pipeline_ms"] for r in trows if r.get("pipeline_ms"))
    inf = sorted(r["inference_time_ms"] for r in trows if r.get("inference_time_ms"))
    if pipe:
        stats["detector_pipeline_ms_p50"] = round(statistics.median(pipe), 1)
        stats["detector_pipeline_ms_p95"] = round(pipe[int(0.95 * len(pipe))], 1)
    if inf:
        stats["detector_inference_ms_p50"] = round(statistics.median(inf), 1)
        stats["detector_inference_ms_p95"] = round(inf[int(0.95 * len(inf))], 1)
    if lat and pipe:
        s = sorted(x + statistics.median(pipe) for x in lat)
        stats["capture_to_alert_p50_ms"] = round(statistics.median(s), 1)
        stats["capture_to_alert_p95_ms"] = round(s[min(len(s) - 1, int(0.95 * len(s)))], 1)
    print("\n== LATENCY", json.dumps(stats, indent=2))
    (OUT / "latency.json").write_text(json.dumps(stats, indent=2))

    print("\n== snapshots")
    snaps = []
    for a in sorted(tal, key=lambda x: x["id"])[:8]:
        dst = OUT / f"snapshot-{a['id']}.jpg"
        code = curl("GET", f"/alerts/{a['id']}/snapshot.jpg", raw_out=dst)
        snaps.append({"id": a["id"], "event_type": a["event_type"],
                      "rule_name": a["rule_name"], "curl": code.strip(),
                      "bytes": dst.stat().st_size if dst.exists() else 0})
        print(f"  alert {a['id']} ({a['event_type']}/{a['rule_name']}): {code.strip()}")
        if a["snapshot_state"] != "received":
            fails.append(f"alert {a['id']} snapshot_state={a['snapshot_state']}")
    (OUT / "snapshots.json").write_text(json.dumps(snaps, indent=2))

    if SKIP_ADL:
        print("\n== ADL stream skipped (ESK_SKIP_ADL)")
        print("\n== FAILURES")
        for f in fails:
            print("  FAIL", f)
        if not fails:
            print("  none")
        (OUT / "failures.json").write_text(json.dumps(fails, indent=2))
        return 1 if fails else 0

    print("\n== ADL stream (second camera) alerts")
    adl = [a for a in new_alerts if a["device_id"] == ADL_DEV]
    for a in sorted(adl, key=lambda x: x["id"]):
        print(f"  {a['id']} {a['event_type']:11s} {a['rule_name']:9s} "
              f"track={a['track_id']} score={a['score']} snap={a['snapshot_state']}")
    print(f"  total on {ADL_DEV}/{ADL_STREAM}: {len(adl)}")
    cross_rule = [a for a in adl if a["rule_name"] in RULE_EXPECT] + \
                 [a for a in tal if a["rule_name"].startswith("adl-")]
    if cross_rule:
        fails.append(f"rule bodies leaked across streams: {[a['id'] for a in cross_rule]}")

    print("\n== FAILURES")
    for f in fails:
        print("  FAIL", f)
    if not fails:
        print("  none")
    (OUT / "failures.json").write_text(json.dumps(fails, indent=2))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
