# rknn_toolkit_lite2 per-inference memory growth — standalone reproduction

Two scripts. Each is self-contained: no imports from this repository, no build
step, no packaging. Copy either one onto an RKNPU board and run it.

| file | what it does | dependencies |
|---|---|---|
| `leak_repro.py` | the growth, through `rknnlite.api.RKNNLite`, with `--gc-every N` as the intervention | `numpy`, `rknn_toolkit_lite2` |
| `ctypes_control.py` | the same API sequence with `rknn_toolkit_lite2` removed from the call path, plus a positive control | `numpy`, `ctypes` (stdlib) |

What the scripts establish:

`leak_repro.py` shows RSS climbing tens of kB per `inference()` call and never
levelling off. `ctypes_control.py` runs `rknn_inputs_set` / `rknn_run` /
`rknn_outputs_get` / `rknn_outputs_release` straight against `librknnrt.so` —
the sequence the closed Cython extension performs — and stays flat, so
`librknnrt` itself is clean and the growth is introduced by the
`rknn_toolkit_lite2` wrapper.

`leak_repro.py --gc-every 5` then shows the growth collapsing by 269× (42.4321
→ 0.1578 kB/inference) with RSS oscillating instead of trending. Memory that
comes back when `gc.collect()` is called is not lost: **`RKNNLite.inference()`
leaves one set of reference cycles per call, holding that call's ctypes
buffers.** Reference cycles are invisible to refcounting; only the cyclic
collector breaks them. This is not a C-level `malloc` without a matching
`free`.

The cycles are built inside `rknnlite/api/rknn_runtime.cpython-311-aarch64-linux-gnu.so`, which is Cython-compiled (`__pyx_*` symbols) and reaches `librknnrt` through **ctypes** (`CDLL`, `POINTER`, `RKNNRtTensorAttr`). So the objects in the cycle are ctypes objects, but the code constructing them is the vendor's compiled module, not the caller's — which is why `del outputs` on your side does nothing.

Both scripts use the same warmup, the same sampling interval and the same
kB-per-inference arithmetic, so their numbers are directly comparable. That is
the only reason the comparison means anything.

## Dependencies

```
python3 >= 3.8
numpy
rknn_toolkit_lite2      # leak_repro.py only
librknnrt.so            # ctypes_control.py only, on the board already
```

`rknn_toolkit_lite2` ships as a wheel in
[airockchip/rknn-toolkit2](https://github.com/airockchip/rknn-toolkit2) under
`rknn-toolkit-lite2/packages/`. On vendor images it is usually preinstalled at
`/usr/lib/python3/dist-packages/rknnlite/`.

## Model

Any static-shape `.rknn` with a uint8 NHWC image input works. Neither script
hard-codes an input size or an output count — both are queried from the loaded
model (`get_in_out_num()` + `get_tensor_attr()` for rknnlite,
`rknn_query(IN_OUT_NUM / INPUT_ATTR / OUTPUT_ATTR)` for ctypes).

The runs below use a yolov8n exported with rknn-toolkit2 2.3.2 from
[rknn_model_zoo](https://github.com/airockchip/rknn_model_zoo) — a 1x640x640x3
uint8 input and a 9-tensor raw detection head. Copies for the boards this repo
targets:

```
platforms/rknn/models/yolov8n_zoo_int8.rk3588.rknn
platforms/rknn/models/yolov8n_zoo_int8.rk3576.rknn
platforms/recamera-pro/models/yolov8n_zoo_int8.rv1126b.rknn
```

The growth is not specific to this graph: the same rate was measured through the
same wrapper with a yolo11n-pose head, which returns a different number of
output tensors of different shapes.

## Running

```bash
# the growth
python3 leak_repro.py --model yolov8n.rknn --seconds 300 --sample-every 30

# the intervention — same run with periodic gc.collect()
python3 leak_repro.py --model yolov8n.rknn --seconds 300 --sample-every 30 --gc-every 5

# the control — same sequence, rknn_toolkit_lite2 out of the path
python3 ctypes_control.py --model yolov8n.rknn --seconds 300 --sample-every 30

# positive control — proves the measurement can see a leak of this size
python3 ctypes_control.py --model yolov8n.rknn --omit-release
```

Useful flags (both scripts): `--warmup N` (default 20, excluded from the
statistics), `--max-iterations N`, `--min-free-mb MB` (default 250),
`--json out.json`. `ctypes_control.py` also takes `--lib` if `librknnrt.so` is
not at `/usr/lib/librknnrt.so`.

### `--gc-every N` is the verdict, not a tuning knob

`leak_repro.py --gc-every N` calls `gc.collect()` after every N measured
inferences, in the same place in the loop as the sampler, so the two runs stay
comparable. It is what separates a real leak from cyclic garbage:

| with `--gc-every 5` | reading |
|---|---|
| rate drops to the same order as the ctypes control, RSS oscillates without trend | cyclic garbage — memory the collector reclaims, not memory that was lost |
| rate essentially unchanged | a real leak, outside the reach of the collector |

Run both, back to back, on the same board and model:

```bash
python3 leak_repro.py --model m.rknn --seconds 300 --sample-every 30
python3 leak_repro.py --model m.rknn --seconds 300 --sample-every 30 --gc-every 5
```

N between 1 and 10 is enough to make the difference obvious; smaller N costs
more latency (`--gc-every 5` moved p50 from 19.8 ms to 24.3 ms on RK3588) and
does not change the verdict. Watch `infer_ms_p50` in both runs — the gc cost
shows up there, and if you are considering periodic `gc.collect()` as the
production mitigation, that number is the price.

### Minimum credible sample

At least **3000 inferences or 180 s**, whichever comes later, with
`--sample-every` no larger than 30 s so the per-interval slope is visible.

The first interval of any run is dominated by start-up allocation: model load,
glibc arena growth, the first-touch of the output buffers. A 30-inference
reading is almost entirely that, and it will report a per-inference rate that
neither reproduces nor scales. Only a rate whose per-interval slope stays
roughly constant across at least four intervals means anything.

### `--omit-release` is capped on purpose

Each skipped `rknn_outputs_release` retains the full dequantized output set —
about 4.6 MB for this graph. Unbounded, that exhausts a 1.27 GB headroom in
roughly ten seconds and takes the board off the network. So the mode caps
itself: `--max-iterations` defaults to **40** (about 190 MB) and
`MemAvailable` is checked against `--min-free-mb` after **every** iteration.
At 4.6 MB a call, checking every 32nd iteration would leave a 157 MB blind
spot — most of the headroom the guard exists to defend.

Do not raise the cap to "see it better". Forty iterations already separates
4600 kB/inference from 42 kB/inference by two orders of magnitude.

## `/dev/rknpu` is root-only on reCamera Pro

On the reCamera Pro (RV1126B) stock image the NPU character device is mode
`0600 root:root`. The default account is uid 1000 with no sudo, so
`rknn_init` / `init_runtime()` fails with:

```
failed to open rknpu module, need to insmod rknpu dirver!
```

That message reads like a missing kernel module. It is not — the module is
loaded and the device node exists; the process simply cannot open it. `insmod`
will not help. Check before you debug the wrong thing:

```sh
ls -l /dev/rknpu    # crw------- 1 root root  → permission, not driver
```

Two ways to get a process that can open it:

1. **adb.** `adbd` on the stock image runs as root, so an adb shell already is
   root. Over USB the board is at `192.168.42.1`; over the network,
   `adb connect <ip>:5555`. Push the script and run it:

   ```sh
   adb connect <ip>:5555
   adb push leak_repro.py /userdata/
   adb shell python3 /userdata/leak_repro.py --model /userdata/model.rknn
   ```

2. **Run it as an appmgr app.** `appmgr` starts apps as root, so a script
   invoked from an app's entry point inherits the access. This is the heavier
   route — the package has to be signed with an ECDSA-P256 key the device
   trusts — and it is only worth it if you need the growth measured inside a
   real app process rather than standalone.

Other RKNPU boards differ. On RK3588 (Radxa, kernel 6.1 BSP) the NPU is reached
through DRM render nodes (`/dev/dri/renderD12*`, group `render`), so a normal
user in the `render` group runs both scripts with no root at all — that is how
the sample output below was produced.

## Reading the output

Each script prints one row per sample and then the summary:

- `kb_per_inference` — the number that matters. Computed from the first to the
  last post-warmup sample: `ΔVmRSS / Δiterations`.
- `heap_MB` — size of the anonymous `[heap]` VMA. It tracking `rss_kb` places
  the growth on the heap rather than in new mappings or file-backed pages. It
  does **not** distinguish unfreed `malloc` from cyclic garbage: the ctypes
  buffers held by the cycles are heap allocations too. Only `--gc-every`
  separates those two.
- `free_MB` — `MemAvailable`, what the abort guard watches. `MemFree` would be
  the wrong input: most of what these boards hold is reclaimable page cache.
- `stopped_by` — `max_iterations` or `min_free_mb` if a guard fired.

Interpretation:

| `kb_per_inference` | reading |
|---|---|
| tens of kB, not decaying across samples | accumulation rate of memory not yet reclaimed by the collector — re-run with `--gc-every 5` to find out whether it is cyclic garbage or a real leak |
| under ~0.1 | flat; the residual is glibc arena settling, and it stops |
| thousands | the positive control, or something much worse |

The number by itself is a growth rate, not a leak. `leak_repro.py` at
42 kB/inference and `leak_repro.py --gc-every 5` at 0.16 kB/inference are the
same code doing the same work; the difference is whether the cyclic collector
got to run. Read row 1 as "not reclaimed yet", and use `--gc-every` to decide
which kind it is.

A single sample interval is not enough. Growth over the first interval alone is
dominated by arena settling; run at least 300 s and check that the per-interval
delta stays roughly constant instead of decaying.

## What you should see

Real output, RK3588 (Radxa ROCK 5T), kernel 6.1.84, Python 3.11,
rknn_toolkit_lite2 2.3.2, `librknnrt` 2.3.2 (429f97ae6b@2025-04-09), NPU driver
0.9.8, `yolov8n_zoo_int8.rk3588.rknn`. Informational `I RKNN:` banner lines are
elided.

### 1. `leak_repro.py` — the growth

```
$ python3 leak_repro.py --model yolov8n_zoo_int8.rk3588.rknn --seconds 300 --sample-every 30
model yolov8n_zoo_int8.rk3588.rknn
  input (1, 640, 640, 3) uint8 NHWC, 1 input(s), 9 output(s)

=== rknnlite baseline ===
      t_s     iters     rss_kb   heap_MB   free_MB
     0.00         0      80648      32.8   10809.2
    30.02      1355     218232     170.3   10677.4
    60.03      2664     264100     217.3   10630.7
    90.04      4004     312160     266.3   10597.3
   120.05      5344     362848     318.2   10545.5
   150.05      6747     410500     366.7   10500.4
   180.06      8094     457980     415.5   10453.2
   210.07      9350     506284     464.8   10442.6
   240.07     10734     553468     513.2   10393.9
   270.11     12048     600784     561.5   10346.4
   300.01     13359     648240     610.0   10301.8

   iterations_measured: 13359
          rss_delta_kb: 567592
      kb_per_inference: 42.4876
            mb_per_min: 110.854
          infer_ms_p50: 23.367
```

RSS goes 80 MB → 648 MB in five minutes. `heap_MB` tracks it one-for-one, so
the growth is heap. The per-interval slope does not decay: 47.6 MB over the
second interval, 47.5 MB over the last one.

### 2. `ctypes_control.py` — the same sequence, flat

```
$ python3 ctypes_control.py --model yolov8n_zoo_int8.rk3588.rknn \
    --lib /usr/lib/aarch64-linux-gnu/librknnrt.so --seconds 300 --sample-every 30
model yolov8n_zoo_int8.rk3588.rknn
  input (1, 640, 640, 3) uint8 NHWC, 1 input(s), 9 output(s)
  librknnrt 2.3.2 (429f97ae6b@2025-04-09T09:09:27) driver 0.9.8

=== ctypes control ===
      t_s     iters     rss_kb   heap_MB   free_MB
     0.00         0      56916      16.7   10879.1
    30.02      1342      57176      16.7   10878.2
    60.03      2743      57176      16.7   10875.8
    90.04      4250      57440      16.7   10879.3
   120.05      5623      57440      16.7   10876.8
   150.07      7071      57440      16.7   10875.5
   180.08      8550      57440      16.7   10863.6
   210.08     10019      57440      16.7   10852.0
   240.09     11467      57704      16.7   10843.9
   270.10     12985      57704      16.7   10847.3
   300.00     14551      57704      16.8   10844.4

   iterations_measured: 14551
          rss_delta_kb: 788
      kb_per_inference: 0.0542
            mb_per_min: 0.154
```

14 551 inferences, 788 kB total — three 264 kB glibc arena steps, then nothing.
Same graph, same input, same sampler, 784× lower per-inference growth. Inference
latency is also slightly lower (21.2 ms p50 against 23.4 ms), so the flat curve
is not "it did less work".

### 3. `ctypes_control.py --omit-release` — positive control

```
$ python3 ctypes_control.py --model yolov8n_zoo_int8.rk3588.rknn \
    --lib /usr/lib/aarch64-linux-gnu/librknnrt.so --seconds 300 --sample-every 1 --omit-release
model yolov8n_zoo_int8.rk3588.rknn
  input (1, 640, 640, 3) uint8 NHWC, 1 input(s), 9 output(s)
  librknnrt 2.3.2 (429f97ae6b@2025-04-09T09:09:27) driver 0.9.8

=== ctypes + omit release (POSITIVE CONTROL) ===
      t_s     iters     rss_kb   heap_MB   free_MB
     0.00         0      56920      16.7   10845.7
     0.81        40     242508     206.7   10665.3

   iterations_measured: 40
          rss_delta_kb: 185588
      kb_per_inference: 4639.7
            mb_per_min: 13425.058
            stopped_by: max_iterations
          infer_ms_p50: 19.433
```

4639.7 kB per iteration, which is the nine dequantized float32 output tensors —
the amount `rknn_outputs_release` is supposed to reclaim. The run stops itself
after 40 iterations (`stopped_by: max_iterations`), 0.81 s in, having spent
186 MB.

This is the row that makes rows 1 and 2 readable. The harness detects a
4.6 MB/iteration leak in under a second, so the 0.054 kB/inference in run 2 is
a real flat line and not a blind sampler. It also separates run 1 from a missing
`release`: run 1 grows 42 kB/inference, 1/110th of what an actually-missing
`release` costs — so what accumulates through `rknnlite` is not the output
buffers.

### 4. `leak_repro.py --gc-every 5` — the intervention

Same board, same model, same sampler, one added `gc.collect()` every fifth
measured inference:

| run | iterations | kb_per_inference | RSS | infer_ms_p50 |
|---|---:|---:|---|---:|
| `leak_repro.py` | 13 690 | 42.4321 | 80.6 → 661.5 MB, monotonic | 19.8 |
| `leak_repro.py --gc-every 5` | 9 733 | 0.1578 | sawtooth between 65 and 87 MB, no trend | 24.3 |

269× lower, the same order as the ctypes control's 0.0542. The RSS curve stops
trending and starts oscillating — memory is being handed back, so it was never
lost. The gc costs 23% on inference latency.

## Three readings the scripts separate

**"You forgot to release the outputs."** `RKNNLite.inference()` owns the entire
output path — it calls `set_inputs`, `run`, `get_outputs` on the Cython runtime
and returns converted arrays. There is no output handle a caller can release.
The only release available is `RKNNLite.release()`, which destroys the context;
`leak_repro.py` calls it once, in a `finally`, after the loop. The magnitudes
also disagree: an unreleased output set costs 4.6 MB per call (run 3), not
42 kB.

**"You are re-initialising the model every frame."** `load_rknn()` and
`init_runtime()` are each called exactly once, before the loop. Grep the file —
each name appears once as a call — and the measured loop body contains one RKNN
call and nothing else. The input array is allocated once outside the loop and
never rewritten; the returned list is `del`'d immediately.

**Reference cycles — this is what it actually is.** `--gc-every 5` drops the
rate 269×, and a diagnostic that calls `gc.collect()` after every 100
inferences returns RSS to **exactly 64632 kB** on all five collections,
reclaiming a fixed ~198 objects each time. With `gc.DEBUG_SAVEALL` after 20
inferences the cycles hold one `c_char_Array_1228800` (the 640x640x3 input
copy) per inference, one ctypes array per output tensor per inference, and 198
`RKNNRtTensorAttr`. The cycles are created inside `RKNNLite.inference()`, one
set per call.

This also explains why `del outputs` in the loop changes nothing: the same
diagnostic reports **`numpy bytes held in cycles: 0`**. What the cycles hold
are the ctypes buffers behind the conversion, not the numpy arrays handed back
to the caller. Dropping the caller's reference cannot reach them — only the
cyclic collector can.

## Mitigations

Two, with different trade-offs:

- **Periodic `gc.collect()`** in the caller's loop. One line, no dependency on
  library internals, costs ~23% inference latency at `--gc-every 5` (larger N
  costs less; check with the flag on your own graph).
- **Bypass the wrapper**: drive `librknnrt.so` over ctypes, as
  `ctypes_control.py` does. No gc needed, and inference latency is slightly
  lower (21.2 ms p50 vs 23.4 ms), at the cost of maintaining the bindings and
  a numerical-equivalence test against the wrapper's output.

## Note on the RV1126B numbers

This repository's postmortem (`docs/rknn-leak-postmortem.md`) reports
43.78 kB/inference on reCamera Pro (RV1126B) with the same
`rknn_toolkit_lite2` 2.3.2, `librknnrt` 2.3.2 and NPU driver 0.9.8. The
RK3588 run above measures 42.49 kB/inference on the same graph. The sample
output in this file is the RK3588 run because that is what these two scripts,
in this form, were executed on.
