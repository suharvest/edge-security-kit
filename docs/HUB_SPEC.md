# Hub 汇聚层规格（edge-security-kit）

状态：草案 v0.1 · 2026-08-17
契约依赖：[`contracts/mqtt-detection.schema.json`](../contracts/mqtt-detection.schema.json) · [`contracts/MQTT.md`](../contracts/MQTT.md)

Hub 是纯 CPU 的 arm64 单容器：订阅探测层 MQTT，判定规则，管理告警生命周期，
持久化事件与配置，托管前端。不解码视频、不做推理、不聚合视频流。

---

## 1. 模块划分

```
┌──────────────────────────── hub 容器 ────────────────────────────┐
│ mqtt_ingest ──> rule_engine ──> alert_manager ──> storage        │
│      │              │               │                │           │
│      └─ device_registry             └─ ws_push       ├─ SQLite   │
│                                                      └─ snapshots│
│ http_api（REST + WS + 静态前端托管，cookie 会话鉴权）              │
└──────────────────────────────────────────────────────────────────┘
   sidecar: mosquitto
```

| 模块 | 职责 | 输入 → 输出 |
|---|---|---|
| mqtt_ingest | 订阅 detections/status/events/snapshot 四类 topic，JSON 三类过 schema 校验（校验失败计数并丢弃，不 raise）；snapshot 为 JPEG 二进制，校验 magic bytes + 200 KB 上限 | MQTT → 内部队列 |
| device_registry | 由 status(retain+LWT) 维护设备/流在线状态、decode 健康、版本 | status → 内存态 + devices 表 |
| rule_engine | 对 hub 模式设备的 detections 逐条判定 zone/line/loiter；单机模式设备直接采信其 events | detections/events → 候选告警 |
| alert_manager | 冷却去重、生成 event_id、发 cmd/snapshot、回发 events topic（必须打 `origin: "hub"`）、三态状态机 | 候选告警 → alerts 表 + WS 推送 |
| storage | SQLite（WAL 模式）+ 快照文件目录 + 配置原子写回 | — |
| http_api | REST/WS/静态文件，cookie 会话鉴权全覆盖（除 /api/health 与 /api/auth/login） | — |

### 1.1 events 回声防护（实测故障模式）

hub 既向 `events/<stream_id>` republish 自己的裁决，又订阅 `events/+`——两者
是同一个 topic family。没有防护时 hub 会吃下自己的回声，把 hub 模式设备判成
`single_box`，从此**完全停止判定该设备的规则**。集成测试里的表现是：先出 2 条
zone 告警，然后再无任何 line_cross / loitering。

两道叠加防护，缺一不可：

1. **origin 标记**：hub republish 的 event 必须带 `origin: "hub"`（契约见
   MQTT.md）；入站 event 处理路径丢弃 `origin=hub` 的消息。
2. **event_id 去重**：`event_id` 已在 alerts 表里的 event 视为重复，直接丢弃。
   兜底覆盖旧版本设备或第三方 bridge 转发时丢掉 origin 的情况。

### 1.2 mqtt_ingest 逐消息容错（实测故障模式）

ingest 的消息循环和 broker 连接在同一个 `async with` 里：处理单条消息的异常若
逃到 transport 层，会被当成连接错误，客户端断开并进入指数退避重连。故障模式是
放大式的——一条报文触发的 handler 异常（下游 handler 的 bug、settings 缺字段、
DB 约束冲突等）会连带停掉**全部** topic 的摄入，而且只要触发条件还在（例如 A1
那种持续撞 UNIQUE 的插入），每次重连收到同一条 retained/重发报文就再断一次，
形成断-连循环。

条款：

1. **per-message 隔离**：每条消息的 dispatch 必须包在自己的 try 里，
   `asyncio.CancelledError` 透传（停机路径），其余异常记日志后继续循环，不得
   影响连接。schema 校验失败本就是计数并丢弃（§1 表），这条覆盖的是校验通过
   之后 handler 内部的异常。
2. **计数可见**：被隔离的异常累加到 `handler_errors`，由 `GET /api/health` 暴露
   ——静默吞异常与让连接崩掉一样不可接受，运维要能看到"在丢消息"。

实现：`mqtt_ingest.py:103-119`（隔离与计数）、`app.py:167`（health 暴露）。

模式判定条款：device_registry 只能由 `origin=device`（含字段缺省）的 event 将
设备标记为 `single_box`。detections 到达不改变模式；模式一旦由设备配置或
device 侧 event 确定，hub 不再自行翻转。

进程模型：单进程 asyncio（aiomqtt + aiohttp 或等价物）。事件速率是每秒个位数，
detections 峰值约 8 设备 × 4 流 × 15 Hz = 480 msg/s、每条 ≈1 KB——JSON 解析
与几何判定在单核内完成（需核实，验收见 §10）。detections 不落库。

## 2. 规则引擎迁移对照

来源：上游 Industrial-security-demo（scratchpad 副本 `isd/`）。三条规则与几何
函数是纯 Python/无依赖，直接迁移；坐标本就是归一化配置（`demo_config.json`
的 points/start/end 均为 0-1），与 `frame_norm` 契约天然一致，迁移后不再需要
`norm_px` 的像素换算——全程在归一化空间判定。

| 旧代码 | 内容 | 新模块 | 变更 |
|---|---|---|---|
| behavior_demo.py:50-60 | `point_in_polygon` 射线法 | rules/geometry.py | 原样，输入改归一化坐标 |
| behavior_demo.py:63-68 | `ccw` / `segments_intersect` | rules/geometry.py | 原样 + 新增 `side()` 供方向判定 |
| behavior_demo.py:71-72 | `norm_px` 像素换算 | — | 删除（全程归一化空间） |
| multi_camera_manager.py:521-540 | zone_enter + loitering（`zone_entered_at` 进出簿记 + dwell 超时） | rules/zone.py | 原样迁移；dwell 计时改用 **hub 接收时刻的本地单调时钟**（设备 `timestamp` 仅展示/取证，见 §2.1 时钟条款） |
| multi_camera_manager.py:542-554 | line_cross（前后两帧质心连线与线段相交） | rules/line.py | 补方向：符号翻转判 forward/backward（约定见 MQTT.md），规则配置新增 `direction: any\|forward\|backward`，默认 any。翻转的比较对象不是上一帧，而是该 track 该线的**最后一个非零 side**（`side == 0` 条款见 §2.1） |
| multi_camera_manager.py:464-482 | `_emit` 冷却（挂在 track 上） | alert_manager | 冷却键改为五元组 `(device_id, stream_id, rule_name, event_type, track_id)`，修复换 track ID 重复报警。`event_type` 必须在键里：zone_enter 与 loitering 共用同一个 rule_name（区域名），旧代码的冷却本就分事件类型（`track.fired_events[etype]`）；退成四元组会让同一 track 的入侵告警吞掉随后的滞留升级——丢的是两者中更严重的那条。另有可选流级限速 `stream_rate_limit_s`（同一 `(device_id, stream_id, rule_name, event_type)` 每 N 秒最多 1 条，即去掉 track_id 的同一把键），**默认 0 = 不限流**：非零值会把同时触发同一规则的两个人合并成一条告警，漏报比重复报警更严重，故为 opt-in |
| 每 track `zone_entered_at`/`fired_events` | 规则状态 | rules/state.py | 状态按 `(device_id, stream_id, track_id)` 三级索引；track 在 detections 中消失超过 `track_expiry_s`（默认 5s）清除状态 |

**配置索引变更**：旧 `camera_zones` 按 `cam-0` 单级索引 → 新规则按
`device_id / stream_id` 两级索引。这是与旧版唯一的配置结构不兼容处。

### 2.1 判定边界条款

- **未跟踪目标**：规则引擎只对 `track_id >= 1` 的目标维持 zone/loiter/line
  状态；`track_id = 0`（未跟踪）的检测不参与任何规则判定，只计入统计。
  接 hub 判规则的平台必须带跟踪；无跟踪能力的设备（reCamera 类）走单机模式
  在设备端判规则、只发 events（契约条款见 MQTT.md track_id 语义）。
- **时钟**：dwell、冷却、line 断链超时一律用 hub 接收时刻的本地单调时钟；
  设备 `timestamp` 仅用于展示与取证，跨设备排序按 hub 接收时刻。设备侧建议
  NTP，但不作为正确性前提。
- **代际重置**：三类消息必含 `session_id`（设备进程每次启动生成）。hub 收到
  某设备新 `session_id` 时，重置该设备全部流的 track 状态与 line 断链起点；
  `frame_id`/`track_id`/事件序号的单调性只在同一 session 内成立。
- **乱序与丢包**：hub 按 `(device_id, stream_id, session_id)` 维护
  last `frame_id`，`frame_id <= last` 的 detections 直接丢弃（乱序保护）。
  line_cross 的相邻点对若两条消息的 hub 接收间隔超过 `line_chain_gap_ms`
  （默认 1000，可配），视为断链：重置该 track 的线段起点、本对不判穿越——
  防止 QoS0 丢帧造成的大跨度连线误判穿越。
- **质心正好落在线上（`side == 0`）**：引擎按 `(track, line)` 维护
  `last_nonzero_side` 与产生它的那个质心。当帧 `side == 0` 时不更新
  `last_nonzero_side`、不判穿越（视为仍在原侧）；当 side 变为非零且与
  `last_nonzero_side` 反号时判定穿越，方向由 `last_nonzero_side → 当前 side`
  决定，有限线段相交测试用"产生 `last_nonzero_side` 的质心 → 当前质心"这一对。
  尚无 `last_nonzero_side` 的 track（首帧、或起始就落在线上）只做播种，
  从线上向一侧移动不算穿越；断链会连同 `last_nonzero_side` 一并丢弃。

  这不是浮点边角料：RK3588 预处理把 1280 宽缩到模型输入的 640 宽，`cx` 因此
  量化到 1/640 网格，而 `x = 0.5` 恰好是 320/640。用户在规则画布上把线画在
  画面正中间——最自然的操作——该线每一帧都读到 `side == 0`。若逐帧比较，
  这条线永远不会触发，现象是同一路流上别的规则都正常、唯独它悄无声息。
  沿用最后一个非零 side 既能判出穿越，又不会让贴着线抖动的目标反复误报
  （它的 side 从不取反号）。契约条款见 contracts/MQTT.md `direction`。

## 3. 告警状态机

```
(rule fires) ──> new ──ack──────> acked ──┐
                  └──dismiss──> dismissed ─┴─(改判: acked <──> dismissed)
```

- 合法迁移：`new→acked`、`new→dismissed`、`acked↔dismissed`（改判）；不允许
  回到 `new`。非法迁移返回 409。无重开、无工单流转。`dismissed` 计入误报统计
  （按 rule_name 聚合，前端用于阈值建议）。
- 状态迁移记录 `acted_by`/`acted_at`；操作者固定为唯一账户，不做多用户。
- 告警主键：`alerts.id` 是 hub 本地单调自增 INTEGER，与设备侧 `event_id`
  分离（后者仅作关联列）。WS 断线补齐依赖此主键：`GET /alerts?after_id=`
  按 id 升序返回，保证 `alert.new` 不丢；快照迟到类 `alert.update` 不经
  after_id 补偿（见 §5）。
- **event_id 序列必须可从存储恢复**（实测故障模式）：hub 生成的 `event_id` 是
  `<device>-<stream>-<session>-<seq>`，`seq` 的计数器在内存里，而 `session_id`
  由设备进程持有。hub 重启时设备并不重启，`session_id` 不变——计数器若从 1 重
  新开始，生成的就是 alerts 表里已存在的 event_id，撞 UNIQUE 约束插入失败。
  失败还不是单条丢失：冷却戳在 INSERT 之前就已落下（见 §2 冷却行），所以重试
  被冷却挡住，**该设备的告警从此永久停止**，直到设备自己重启换 session。
  条款：某 `(device_id, stream_id, session_id)` 在本进程内第一次取序号时，必须
  以存储中该前缀已有的最大 `seq` 作为起点播种，而不是 0。
  实现：`storage.py:max_event_seq` / `alert_manager.py:73-85`。
- 快照获取异步：告警先入库推送（`snapshot_state=pending`），快照到达后
  UPDATE 并再推一次 WS。快照关联状态机
  `snapshot_state ∈ pending / received / timeout / none`：
  - hub 模式：INSERT 时 `pending`，发 `cmd/snapshot` 后 60 s 未到置
    `timeout`；此后快照晚到仍关联，置回 `received` 并推 `alert.update`。
  - 单机模式：无 cmd 往返，事件入库为 `none`，快照主动到达后置 `received`。
  - hub 重启：从 DB 恢复 `pending` 行并重新武装 60 s 计时。

### 3.1 快照取证时序（hub 模式）

```
rule_engine        alert_manager        MQTT                 device
    │ 候选告警 ──────> │
    │                  │ 冷却/限速通过
    │                  │ INSERT alerts(snapshot_state=pending)
    │                  │ WS push alert.new
    │                  │ ──publish──> cmd/snapshot {stream_id,event_id}
    │                  │                                 │ 编码最近帧 JPEG≤200KB
    │                  │ <─────────── snapshot/<event_id> ┘
    │                  │ 落盘 + UPDATE snapshot_path, snapshot_state=received
    │                  │ WS push alert.update
```

超时 60 s 未收到快照：`snapshot_state` 置 `timeout`，不重试（设备可能已断流，
device_registry 会另行标记）；晚到快照仍按 §3 状态机关联。单机模式
（reCamera）无 cmd 环节，快照随事件主动到达，按 event_id 关联；快照先于事件
到达时暂存 60 s 等待关联。快照载荷校验：JPEG magic bytes + 200 KB 上限
（该 topic 不适用 JSON Schema，见 MQTT.md）。

## 4. REST API

前缀 `/api`，全部 cookie 会话鉴权（`/api/health` 与 `/api/auth/login` 除外，
见 §7）；未登录返回 401。错误统一 `{"error": "..."}`。

| Method | Path | 说明 |
|---|---|---|
| GET | /health | 存活 + mqtt 连接态 + schema 校验失败计数，无鉴权 |
| GET | /alerts | 过滤：`state,device_id,stream_id,event_type,rule_name,date_from,date_to,after_id,limit,offset`（默认 limit=50，按 `ts_ms` 倒序；带 `after_id` 时按 id 升序返回 id 更大的行，供 WS 重连补齐）。`rule_name` 精确匹配 |
| POST | /alerts/{id}/ack | 迁移到 acked（合法来源 new/dismissed，见 §3）；非法迁移 409 |
| POST | /alerts/{id}/dismiss | 迁移到 dismissed（合法来源 new/acked）；非法迁移 409 |
| POST | /alerts/ack | 批量：入参 `{ids:[]}`，上限 500，返回逐条结果（成功/409/404） |
| POST | /alerts/dismiss | 批量：同上 |
| GET | /alerts/{id} | 单条告警。前端 WS 重连后用它刷新"渲染中但缺快照"的行（FRONTEND_SPEC §7：`alert.update` 不经 after_id 补偿） |
| GET | /alerts/stats | 按 `rule_name` 聚合的处置统计（total/new/acked/dismissed + 误报率），支持 /alerts 的同一组过滤参数；§3 的误报率阈值提示与 ShiftStats 由它驱动 |
| GET | /alerts/export.csv | 同 /alerts 过滤参数（含 `rule_name`），UTF-8 BOM CSV。列名 `id, ts_ms, device_id, stream_id, event_type, rule_name, track_id, score, state, acted_by, acted_at`。两个端点共用同一份查询参数解析，导出与它所来自的列表覆盖同一批行——任一过滤维度只在前端实现，导出就会漏过该维度 |
| GET | /alerts/{id}/snapshot.jpg | 快照文件；无则 404 |
| GET | /devices | 设备列表：在线态、最后心跳、各流 state/fps/decode、`fallback_active` 醒目标记；透传流的 `preview_url`/`live_url`（来自 status 消息，供规则画布底图与"实时画面"跳转） |
| GET | /rules | 全量规则（两级索引树） |
| GET | /rules/{device_id}/{stream_id} | 单流规则 |
| PUT | /rules/{device_id}/{stream_id} | 整体替换该流 zones+lines+features+cooldown；校验通过即生效并持久化 |
| POST | /rules/{device_id}/{stream_id}/simulate | 入参 `{rule_id}`；构造 `meta.simulated=true` 的告警走完整链路（入库、WS 推送），供画完规则后自测。**绕过冷却**（自测不该被上一条真实告警挡住），且**不发 cmd/snapshot**（没有对应真实帧），落库即 `snapshot_state=none` |
| GET | /devices/{device_id}/config | 该设备全部流的规则+摄像头配置 JSON 导出（备份下载） |
| PUT | /devices/{device_id}/config | 恢复上传，同 /rules 校验与持久化语义，rev 照常自增。**仅恢复规则**：契约没有摄像头配置的下行通道，导出体里的 `camera` 块是信息性的，恢复时不会推回设备（列入未来项，需要新增 `cmd/config` 下行才能闭环） |
| GET | /devices/{device_id}/streams/{stream_id}/preview.jpg | **单帧预览代理**。设备的 `preview_url` 是设备本地地址（`http://127.0.0.1:8099/...`），远端浏览器取不到；hub 服务端去拉一帧 JPEG 回传，缓存 1.5 s（超时 2 s）。设备不可达 / 非 200 / 返回体不是 JPEG，一律 502 + `{"error": "...", "preview_url": "..."}`，不静默降级成灰底。流未上报 `preview_url` 时 404 |
| GET | /live/{device_id}/{stream_id} | 该流最近一条 detections（内存态），供前端画规则时叠加参考框；无视频代理 |
| GET | /config | hub 自身配置（broker 地址、留存天数等） |
| PUT | /config | 同上；重启生效项在响应中列出 |
| POST | /auth/login | 入参用户名+密码；成功签发 HttpOnly+SameSite=Strict 会话 cookie，无鉴权 |
| GET | /auth/session | 当前会话的用户名与 `must_change`；前端据此决定是否强制跳改密页 |
| POST | /auth/logout | 作废当前会话 |
| POST | /auth/password | 修改唯一账户密码，入参旧密+新密；成功后其余会话作废 |

**单帧预览代理 ≠ 视频流聚合**（§11 边界的澄清）：该端点每次请求最多向设备取
一帧 JPEG，不保持连接、不转码、不做多路复用，缓存窗口内 N 个操作员合并成对设备的
一次请求。它替代的是浏览器直连设备本地地址这件本来就做不到的事，不是 RTSP/HLS
中继。视频观看仍然跳设备自己的 `live_url`。

**配置持久化语义**（修复旧版内存态丢失问题的核心条款）：
- 写盘时机：每次 PUT 成功即写，无延迟批处理。
- 原子性：写 `<file>.tmp` + `fsync` + `rename`；SQLite 同事务写 config_versions。
- 版本：规则每次 PUT 使 `rev` 自增；config_versions 保留最近 50 版，支持人工回滚（直接 PUT 旧版本体，不提供自动回滚接口）。
- 响应必须携带持久化结果（`rev` + 落盘时间），前端据此显示"已保存·rev N"。
- hub 自身配置（`PUT /config`）不另建表：存 config_versions 的 `scope='hub'`，
  同 scope 内 `rev` 最大的行即当前值。§6 的 DDL 因此保持不变。

## 5. WS 推送

`GET /ws`（同 origin，会话 cookie 随握手自动发送；无有效会话拒绝升级）。
服务端推送三类：

```json
{"type": "alert.new",    "alert": { ...alerts 行... }}
{"type": "alert.update", "alert": { ... }}          // 快照到达、状态被他端修改
{"type": "device.status","device": { ... }}          // 上下线、decode 变化、fallback_active 翻转
```

`alert` 载荷就是 §6 的 alerts 行（加 `snapshot_url`/`simulated` 两个派生字段），
字段名与列名一致：设备时间戳只叫 **`ts_ms`**，没有 `ts` 之类的别名。REST 与 WS
共用同一个序列化函数，前端因此不需要"两个名字都接受"的兼容分支——那种分支只会
让"hub 一个都没发"看起来像正常情况。

不做客户端上行（处置走 REST，客户端仅心跳 ping）。断线重连后前端以
`GET /alerts?after_id=<last_seen>` 补齐 `alert.new` 缺口（按 hub 本地自增 id
升序，见 §3）；断线期间的快照迟到类 `alert.update` 不在补偿范围——前端对
当前渲染中 snapshot 缺失的告警逐条重新 GET。

## 6. SQLite 表结构

WAL 模式，单文件 `/data/hub.db`。

```sql
CREATE TABLE alerts (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,  -- hub 本地单调自增；after_id 补齐依赖此列
  event_id      TEXT NOT NULL UNIQUE,      -- 设备/hub 规则引擎的 <device>-<stream>-<session>-<seq>，仅作关联
  ts_ms         INTEGER NOT NULL,          -- 设备 timestamp（展示/取证）
  received_ms   INTEGER NOT NULL,          -- hub 接收时刻（排序/计时基准）
  device_id     TEXT NOT NULL,
  stream_id     TEXT NOT NULL,
  event_type    TEXT NOT NULL CHECK(event_type IN ('zone_enter','loitering','line_cross')),
  rule_name     TEXT NOT NULL,
  track_id      INTEGER NOT NULL,
  score         REAL,
  bbox          TEXT NOT NULL,             -- JSON [cx,cy,w,h]
  direction     TEXT,                      -- line_cross: forward/backward
  dwell_s       REAL,                      -- loitering
  state         TEXT NOT NULL DEFAULT 'new' CHECK(state IN ('new','acked','dismissed')),
  acted_by      TEXT,
  acted_at      INTEGER,
  snapshot_state TEXT NOT NULL DEFAULT 'none' CHECK(snapshot_state IN ('pending','received','timeout','none')),
  snapshot_path TEXT,
  meta          TEXT                       -- JSON, additionalProperties 透传
);
CREATE INDEX idx_alerts_ts ON alerts(ts_ms DESC);
CREATE INDEX idx_alerts_state ON alerts(state, ts_ms DESC);
CREATE INDEX idx_alerts_scope ON alerts(device_id, stream_id, ts_ms DESC);

CREATE TABLE devices (
  device_id    TEXT PRIMARY KEY,
  online       INTEGER NOT NULL DEFAULT 0,
  last_seen_ms INTEGER,
  mode         TEXT NOT NULL DEFAULT 'hub' CHECK(mode IN ('hub','single_box')),
  info         TEXT                        -- JSON: versions/streams 最近快照
);

CREATE TABLE rules (
  device_id  TEXT NOT NULL,
  stream_id  TEXT NOT NULL,
  rev        INTEGER NOT NULL,
  body       TEXT NOT NULL,                -- JSON: zones/lines/features/cooldown
  updated_ms INTEGER NOT NULL,
  PRIMARY KEY (device_id, stream_id)
);

CREATE TABLE config_versions (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  scope      TEXT NOT NULL,                -- 'rules:<device>/<stream>' | 'hub'
  rev        INTEGER NOT NULL,
  body       TEXT NOT NULL,
  created_ms INTEGER NOT NULL
);

CREATE TABLE auth (
  username      TEXT PRIMARY KEY,
  password_hash TEXT NOT NULL,             -- bcrypt
  must_change   INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE sessions (                    -- §7：会话跨 hub 重启保持有效
  token       TEXT PRIMARY KEY,            -- cookie 里的值即主键
  username    TEXT NOT NULL,
  created_ms  INTEGER NOT NULL,
  expires_ms  INTEGER NOT NULL             -- 滑动空闲截止，每次使用后推
);
CREATE INDEX idx_sessions_expiry ON sessions(expires_ms);
```

留存策略：alerts 与快照默认保留 30 天（可配），每日定时清理；快照目录
`/data/snapshots/<YYYYMMDD>/<event_id>.jpg`。

## 7. 鉴权

- 唯一账户，cookie 会话鉴权：`POST /api/auth/login` 校验用户名+密码（bcrypt
  存储）后签发 HttpOnly + SameSite=Strict 会话 cookie；REST 与 WS 同用此
  cookie（浏览器无法给 WS 握手自定义 Authorization 头，Basic Auth 凭据是否随
  WS 发送依浏览器而异，故不采用）。未登录 REST 返回 401，前端跳登录页。
  **没有固定默认密码**（不接受 admin/admin 之类出厂凭据——安防产品自己的
  控制台带出厂密码是要避免的失效模式）。首次启动（auth 表为空）时：
  - 若设置了 `HUB_ADMIN_PASSWORD` 环境变量，用它；
  - 否则随机生成 ≥16 字符的密码，同时写入容器日志（WARNING 级）与
    `<data_dir>/initial-password.txt`（权限 0600，供漏看启动输出的运维读取）。

  两种情况都置 `must_change=1`，前端强制改密后才放行其余页面。
- **会话持久化**：会话服务端存储，落在同一份 SQLite 的 `sessions` 表
  （`token` 主键 + `created_ms` + `expires_ms`），**hub 重启后未过期的会话继续
  有效**。升级、崩溃重启、断电恢复都不该把当班的人踢出登录页。`expires_ms` 是
  滑动空闲截止时间，每次使用向后推，默认空闲过期 7 天；过期行在命中时删除，
  进程启动时统一清扫一次，所以没被清扫到的过期行也不会放行。
  `POST /auth/logout` 删除该行；改密删除除当前会话外的全部行（§4）——这两条
  语义同样跨重启保持。
- 传输安全：LAN 场景 v1 不内置 TLS；对外暴露时前置反向代理终结 TLS（文档注明）。
- MQTT broker：单机模式 mosquitto 只监听 127.0.0.1 可匿名；hub 模式监听 LAN
  必须启用账号（v1 全体探测设备共享一组凭据，per-device 凭据列入 v2）。
- 容器不需要 `privileged`、不加 cap（对照旧版 SYS_ADMIN+NET_ADMIN 的修复项）。

## 8. 部署形态

```yaml
# docker-compose.yml（hub 主机，任一 arm64 盒子或 x86 服务器）
services:
  hub:
    image: sensecraft-missionpack.seeed.cn/solution/edge-security-hub:<tag>
    restart: unless-stopped
    ports: ["8090:8090"]
    volumes: ["./data:/data"]
    environment:
      MQTT_HOST: mosquitto
    depends_on: [mosquitto]
  mosquitto:
    image: eclipse-mosquitto:2
    restart: unless-stopped
    ports: ["1883:1883"]
    volumes: ["./mosquitto:/mosquitto/config:ro"]
```

**同一 broker 上的多个 hub，MQTT `client_id` 必须唯一。** MQTT 的 client_id 在
broker 上是排他的：后连上来的客户端顶掉同名的旧连接，旧连接被踢下线后按退避
重连，再把新连接顶掉，形成约 1 秒一轮的互踢。detections 走 QoS0，被踢的一侧
直接丢消息，broker 与两侧日志都不会报错。实测症状是偏态的：需要连续两帧的
`line_cross` 大约丢一半，只需任一帧的 `zone_enter` 看起来一切正常——很难往
连接层上想。因此 `mqtt_client_id` 默认不再是固定值，而是每个 hub 进程生成的
`edge-security-hub-<主机名>-<4 位十六进制>`；要固定值时用 `MQTT_CLIENT_ID`
或 config.json 显式覆盖，此时唯一性由部署者负责。当前生效值由
`GET /api/health` 的 `mqtt_client_id` 给出，连接成功与断开都会记进日志——
排查互踢先看这两处。

资源预算（目标值，需核实）：8 设备 × 4 流 × 15 Hz 注入下 hub 稳态 < 1 vCPU、
RSS < 300 MB、SQLite 写放大可忽略（仅事件落库）。镜像目标 < 150 MB（python-slim
+ aiomqtt/aiohttp/bcrypt，无 numpy/cv2——几何判定是纯标量运算）。

## 9. 单机 Jetson preset 的复用方式

单机 = 探测容器 + hub 容器 + mosquitto 同机 compose，MQTT 走 localhost：

- 探测容器按 hub 模式发 detections/status（设备自己不判规则），hub 容器判定——
  与多设备部署是同一份镜像同一份代码，仅 compose 拓扑不同。
- MQTT.md 中"单机模式设备本地规则引擎"这一形态保留给 reCamera（无力跑 hub 的
  端侧），Jetson/RK 单机一律走同机 hub，避免维护两套规则引擎。
- 旧版升级路径：旧 `demo_config.json` 的 `rules` 段结构（zones/lines 字段名、
  归一化坐标）被新 rules body 原样兼容，迁移脚本只做 `cam-0` → `<device>/<stream>`
  的索引升维。

## 10. 验收要点（并入性能验收清单）

- 契约：CI 跑 `contracts/check_fixtures.sh`——遍历 `contracts/fixtures/` 下全部
  fixture 过 `validate_payload.py`，任一失败即非零退出。fixture 是从真实运行
  抓下来的报文（来源见 `contracts/fixtures/README.md`），不是手写样例。
  坏消息（缺 coordinate_space、坐标越出 [0,1]）必须被拒收且计数可见。
  schema 只能校验 [0,1] 范围——范围内的 letterbox 坐标错误 hub 无法在线识别，
  由各平台 conformance fixture 用非正方形源视频对照真值验证。
- 注入压测：mediamtx + 探测容器 8×15 Hz 实流，hub CPU/RSS 达 §8 预算；
  告警端到端（探测帧 → WS 推送）P95 < 1 s（需核实）。
- 断电重启：规则、告警、处置状态全存活（对照旧版内存态丢失）。
- LWT：拔网线 ≤ 心跳周期 + broker keepalive 内设备页转 offline。

## 11. 明确不做（防 VMS 化）

不聚合视频流（点开告警跳设备本地流）、无录像时间轴/NVR 存储管理、无 PTZ、
无电子地图、无多租户/角色树、无工单流转、无人脸/车牌检索。detections 不落库、
不回放。以上任何一项的需求出现时，答案是对接第三方 VMS，不是在 hub 里长出来。

**"不聚合视频流"的边界**：`GET /devices/{id}/streams/{sid}/preview.jpg`（§4）
是单帧 JPEG 代理，不在此列。区分标准是连接数与时序：视频聚合意味着 hub 为每路流
维持长连接、按帧率持续拉取并向前端扇出；单帧代理是一次请求取一帧、带缓存、无状态。
前者让 hub 的资源占用随流数和观看人数增长，后者不会。禁止在此基础上加 MJPEG
multipart、HLS 切片、WebRTC 转发或任何"顺手做成连续的"变体。

**无批量删除端点**：没有 `DELETE /alerts`、没有"清空全部事件"。告警是取证记录，
清理由 §6 的留存策略按天自动做；一个能一次抹掉全部证据的端点，对合法用户省下
的是几次点击，对拿到会话的人省下的是全部工作量。单条删除同理不提供。
