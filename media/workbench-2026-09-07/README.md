# Workbench video wall — capture set, 2026-09-07

What the four restored features look like. Captured from `web/mock-server.js`,
not from a site: the pictures are the repo's own colour-bar fixture, so read the
chrome, the overlays and the numbers rather than the scene. Everything else on
screen — the grid, the boxes, the rule shapes, the sliders, the summary strip,
the dialog and its failure text — is the shipped frontend against the shipped
hub contract.

| File | Size | Shows |
|---|---|---|
| `wall-4up-zh.png` / `wall-4up-en.png` | 3520×1980 (DPR 2) | Four-tile wall: overlay boxes with track id and score, the zone polygon and the directed line, per-stream confidence sliders reading four different values, one offline tile under its veil, and the summary strip |
| `wall-9up-zh.png` | 3520×1980 | Nine-tile wall after five cameras were added from the dialog |
| `wall-fullscreen-zh.png` | 3520×1980 | The HDMI mode: top bar gone, nine tiles filling the screen, `退出全屏 F` |
| `wall-confidence-zh.png` / `-en.png` | 3520×1980 | A confidence slider mid-change |
| `wall-add-camera-zh.png` / `-en.png` | 880×~1150 | The add-camera dialog, with the RTSP password masked in the echo line |
| `wall-add-camera-timeout-zh.png` | 880×1208 | The outcome that matters most: the detector did not answer, so the dialog stays open and says the result is unknown |
| `wall-add-camera.gif` | 1120 wide, 31 frames | Add a camera and watch the tile appear on the wall |

Reproduce:

```bash
node web/mock-server.js --port 8099
# then drive a browser at http://127.0.0.1:8099/wall
# /mock/control?mode=timeout makes the control endpoints answer 504
```
