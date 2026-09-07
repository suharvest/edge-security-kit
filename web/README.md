# hub frontend

Implements `docs/FRONTEND_SPEC.md` P1 (login, alert workbench, rule editor), the P2
devices page and the §10 video wall. The device-local debug page (`§6`) is a placeholder
route.

## Layout

```
web/
├── dist/                     <- the entire deployable artifact; hub serves this directory
│   ├── index.html
│   ├── css/app.css
│   ├── assets/alert.mp3      2.5 KB alert tone (§3.1)
│   ├── vendor/preact-htm.js  Preact 10 + hooks + htm, esbuild ESM bundle (15.5 KB)
│   └── js/                   native ES modules, no build step
├── mock-server.js            dependency-free mock hub for development (HUB_SPEC §4 subset + /ws)
├── mock/                     fixture JPEGs used by the mock hub only
├── tools/check-i18n.js       zh/en dictionary parity + missing-key check
├── tools/test-wall.mjs       §10 wall layout / overlay / slider / add-camera unit tests
├── tools/test-coords.mjs     FRONTEND_SPEC §4.1 / MQTT.md direction unit tests
├── tools/check-streams-shape.mjs  device.streams array shape: contract fixture == mock == frontend
└── screenshots/              verification screenshots
```

## No build step

`dist/js/*` are plain ES modules loaded by the browser; editing a file and reloading is the
whole dev loop. The only pre-built file is `dist/vendor/preact-htm.js`, produced once with:

```bash
npm install preact@10 htm@3 esbuild@0.24
# entry.js re-exports h/render/hooks and html = htm.bind(h)
esbuild entry.js --bundle --format=esm --minify --outfile=web/dist/vendor/preact-htm.js
```

Nothing in `dist/` references an external host — no CDN, font, icon set or analytics.
Verify with `grep -rE "https?://" web/dist/`: the only matches are XML namespace
identifiers (`http://www.w3.org/2000/svg`) inside the Preact bundle, which are never
fetched.

## Hosting requirements for the hub

1. Serve `web/dist/` at the origin root; `/api/*` and `/ws` on the **same origin and port**
   (FRONTEND_SPEC §7 — the client never constructs a port).
2. **SPA fallback**: `GET /`, `/login`, `/devices`, `/rules`, `/debug` must all return
   `index.html`. Assets are requested with absolute paths (`/css/app.css`, `/js/main.js`),
   so no rewriting is needed beyond that fallback. If a deployment cannot do the fallback,
   the router degrades to hash routes (`/#/devices`) automatically, but bookmarked clean
   paths would 404 first.
3. Any authenticated endpoint answering `401` sends the UI to `/login`; the WS handshake
   must be rejected for a dead session (the client probes REST and redirects).

## Endpoint expectations beyond HUB_SPEC §4

Tolerated shapes, so the hub has latitude:

- `GET /alerts` may return `[…]` or `{alerts: […]}`; same for `GET /devices`
  (`[…]` / `{devices: […]}`).
- `POST /alerts/ack|dismiss` (batch) may return `[…]` or `{results: […]}` where an item is
  `{id, status}` and `status` is the per-row HTTP code (`200` / `409` / `404`).
- `PUT /rules/{d}/{s}` should answer `{rev, persisted_ms}`; a `400` may carry
  `{error, issues: [{target_type, target_id, name, message}]}` — with `issues` the editor
  flags the offending zone/line in the sidebar, without it the single `error` string is
  shown verbatim.
- `GET /live` and `GET /live/{d}/{s}` both answer `{received_ms, payload}` where `payload`
  is the detector's `sensecraft.detection/1` message. The mock once answered a flat
  `{objects: [...]}`, the rule editor was written against that, and its reference boxes
  therefore never appeared against a real hub. `util.js` `liveDetections()` is now the one
  reader, and the mock builds the payload in one place.
- `rule_name` is sent as a `/alerts` and `/alerts/export.csv` filter parameter even though
  HUB_SPEC §4 does not list it; the list is additionally filtered client-side, so a hub
  that ignores the parameter still behaves correctly (CSV export cannot be filtered
  client-side — see the report note).
- Username display comes from the `POST /auth/login` response (`{username, must_change}`)
  and localStorage; there is no session-introspection endpoint in the spec.

## Mock hub (development / verification)

```bash
node web/mock-server.js --port 8099            # login: operator / operator
node web/mock-server.js --must-change          # forced initial password change
node web/mock-server.js --any-session          # keep cookies valid across restarts
node web/mock-server.js --interval 8000        # auto-generated alert cadence (ms, 0 = off)
```

Scenario controls (dev only, unauthenticated):

| Route | Effect |
|---|---|
| `/mock/burst?n=3` | fabricate N alerts, snapshots landing 2.5 s later (`alert.update`) |
| `/mock/offline?device=ID` / `/mock/online?device=ID` | flip LWT online state |
| `/mock/decode?device=ID&stream=ID&decode=sw` | flip decode health |
| `/mock/reject?on=1` | `PUT /api/rules` answers `400` with per-item issues |
| `/mock/fail?on=1` | `PUT /api/rules` answers `503` (network-class failure) |
| `/mock/control?mode=ok\|refuse\|timeout` | which of the three control outcomes `PUT .../conf`, `POST .../streams` and `DELETE .../streams/{id}` produce |
| `/mock/state` | counters, WS client count, knob state |

Fixtures include a 4:3 stream (`jetson-01/cam-02`, 1280×960) so the letterbox coordinate
math is exercised, and a stream with no `preview_url` (`rk3588-02/cam-01`) for the grey
canvas fallback.

## Checks

```bash
node web/tools/check-i18n.js     # dictionary parity, missing keys, dynamic prefixes
node web/tools/test-coords.mjs   # §4.1 fit/round-trip/clamp + forward-direction convention
node web/tools/check-streams-shape.mjs   # device.streams is an array everywhere
node web/tools/test-wall.mjs     # §10 grid, overlay staleness, conf clamping, source masking
```

`check-streams-shape.mjs` exists because the mock hub once modelled `streams` as an
object keyed by stream id while the contract and the real hub used an array. The
frontend was written against the mock, so every screen showed `0` as the stream id and
the rule editor saved rules to a stream that did not exist. The script asserts the
contract fixture, the mock's `GET /api/devices` and the frontend's access pattern all
agree, and fails if any source file goes back to `Object.keys(device.streams)`.
