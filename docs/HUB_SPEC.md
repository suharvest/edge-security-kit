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
| alert_manager | 冷却去重、生成 event_id、发 cmd/snapshot、回发 events topic、三态状态机 | 候选告警 → alerts 表 + WS 推送 |
| storage | SQLite（WAL 模式）+ 快照文件目录 + 配置原子写回 | — |
| http_api | REST/WS/静态文件，cookie 会话鉴权全覆盖（除 /api/health 与 /api/auth/login） | — |

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
| multi_camera_manager.py:542-554 | line_cross（前后两帧质心连线与线段相交） | rules/line.py | 补方向：`side(prev)` 与 `side(curr)` 符号翻转判 forward/backward（约定见 MQTT.md），规则配置新增 `direction: any\|forward\|backward`，默认 any |
| multi_camera_manager.py:464-482 | `_emit` 冷却（挂在 track 上） | alert_manager | 冷却键改为 `(device_id, stream_id, rule_name, track_id)`，并叠加流级限速（同一 rule 每 N 秒最多 1 条，默认 N=cooldown），修复换 track ID 重复报警 |
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
| GET | /alerts | 过滤：`state,device_id,stream_id,event_type,date_from,date_to,after_id,limit,offset`（默认 limit=50，按 ts 倒序；带 `after_id` 时按 id 升序返回 id 更大的行，供 WS 重连补齐） |
| POST | /alerts/{id}/ack | 迁移到 acked（合法来源 new/dismissed，见 §3）；非法迁移 409 |
| POST | /alerts/{id}/dismiss | 迁移到 dismissed（合法来源 new/acked）；非法迁移 409 |
| POST | /alerts/ack | 批量：入参 `{ids:[]}`，上限 500，返回逐条结果（成功/409/404） |
| POST | /alerts/dismiss | 批量：同上 |
| GET | /alerts/export.csv | 同 /alerts 过滤参数，UTF-8 BOM CSV |
| GET | /alerts/{id}/snapshot.jpg | 快照文件；无则 404 |
| GET | /devices | 设备列表：在线态、最后心跳、各流 state/fps/decode、`fallback_active` 醒目标记；透传流的 `preview_url`/`live_url`（来自 status 消息，供规则画布底图与"实时画面"跳转） |
| GET | /rules | 全量规则（两级索引树） |
| GET | /rules/{device_id}/{stream_id} | 单流规则 |
| PUT | /rules/{device_id}/{stream_id} | 整体替换该流 zones+lines+features+cooldown；校验通过即生效并持久化 |
| POST | /rules/{device_id}/{stream_id}/simulate | 入参 `{rule_id}`；构造 `meta.simulated=true` 的告警走完整链路（入库、WS 推送），供画完规则后自测 |
| GET | /devices/{device_id}/config | 该设备全部流的规则+摄像头配置 JSON 导出（备份下载） |
| PUT | /devices/{device_id}/config | 恢复上传，同 /rules 校验与持久化语义，rev 照常自增 |
| GET | /live/{device_id}/{stream_id} | 该流最近一条 detections（内存态），供前端画规则时叠加参考框；无视频代理 |
| GET | /config | hub 自身配置（broker 地址、留存天数等） |
| PUT | /config | 同上；重启生效项在响应中列出 |
| POST | /auth/login | 入参用户名+密码；成功签发 HttpOnly+SameSite=Strict 会话 cookie，无鉴权 |
| POST | /auth/logout | 作废当前会话 |
| POST | /auth/password | 修改唯一账户密码，入参旧密+新密；成功后其余会话作废 |

**配置持久化语义**（修复旧版内存态丢失问题的核心条款）：
- 写盘时机：每次 PUT 成功即写，无延迟批处理。
- 原子性：写 `<file>.tmp` + `fsync` + `rename`；SQLite 同事务写 config_versions。
- 版本：规则每次 PUT 使 `rev` 自增；config_versions 保留最近 50 版，支持人工回滚（直接 PUT 旧版本体，不提供自动回滚接口）。
- 响应必须携带持久化结果（`rev` + 落盘时间），前端据此显示"已保存·rev N"。

## 5. WS 推送

`GET /ws`（同 origin，会话 cookie 随握手自动发送；无有效会话拒绝升级）。
服务端推送三类：

```json
{"type": "alert.new",    "alert": { ...alerts 行... }}
{"type": "alert.update", "alert": { ... }}          // 快照到达、状态被他端修改
{"type": "device.status","device": { ... }}          // 上下线、decode 变化、fallback_active 翻转
```

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
```

留存策略：alerts 与快照默认保留 30 天（可配），每日定时清理；快照目录
`/data/snapshots/<YYYYMMDD>/<event_id>.jpg`。

## 7. 鉴权

- 唯一账户，cookie 会话鉴权：`POST /api/auth/login` 校验用户名+密码（bcrypt
  存储）后签发 HttpOnly + SameSite=Strict 会话 cookie；REST 与 WS 同用此
  cookie（浏览器无法给 WS 握手自定义 Authorization 头，Basic Auth 凭据是否随
  WS 发送依浏览器而异，故不采用）。未登录 REST 返回 401，前端跳登录页。
  初始密码随机生成打印到容器日志，`must_change=1` 时前端强制改密后才放行
  其余页面。会话服务端存储（内存 + 重启失效即可），空闲过期默认 7 天。
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

- 契约：hub 对三类消息的 fixture 校验进 CI；坏消息（缺 coordinate_space、
  坐标越出 [0,1]）必须被拒收且计数可见。schema 只能校验 [0,1] 范围——
  范围内的 letterbox 坐标错误 hub 无法在线识别，由各平台 conformance
  fixture 用非正方形源视频对照真值验证。
- 注入压测：mediamtx + 探测容器 8×15 Hz 实流，hub CPU/RSS 达 §8 预算；
  告警端到端（探测帧 → WS 推送）P95 < 1 s（需核实）。
- 断电重启：规则、告警、处置状态全存活（对照旧版内存态丢失）。
- LWT：拔网线 ≤ 心跳周期 + broker keepalive 内设备页转 offline。

## 11. 明确不做（防 VMS 化）

不聚合视频流（点开告警跳设备本地流）、无录像时间轴/NVR 存储管理、无 PTZ、
无电子地图、无多租户/角色树、无工单流转、无人脸/车牌检索。detections 不落库、
不回放。以上任何一项的需求出现时，答案是对接第三方 VMS，不是在 hub 里长出来。
