# edge-security-kit 前端 Spec

状态：草案 v0.1
范围：hub 汇聚层前端（告警工作台 / 设备页 / 规则配置）+ 设备本地调试页。
上游参照：`Zhang-zu-hao/Industrial-security-demo` `web/index.html`（1612 行单文件，下文引用记为 `index.html:<line>`）。

架构前提（已定，本 spec 不讨论）：

- 探测层设备发 MQTT → hub 容器（规则引擎 + SQLite + REST/WS API + 静态前端托管，cookie 会话鉴权）。
- 单机 Jetson 形态下 hub 与探测器同机，前端不感知差异。
- 告警状态机三态：`new → acked / dismissed`。
- hub 不聚合视频流；点开告警跳转设备本地流。
- 不做：录像时间轴 / NVR / PTZ / 电子地图 / 多租户权限树 / 工单流转 / 人脸车牌检索。

---

## 1. 技术选型

约束：

- 离线环境，无 CDN，无外网字体/图标/脚本。
- 构建产物由 hub 容器（arm64）静态托管，必须自包含。
- 双语 zh/en，运行时切换。

选型：**Preact + htm（免编译）或 Vue 3 runtime 单包，二选一；推荐 Preact + htm。**

| 方案 | 优点 | 缺点 |
|---|---|---|
| 继续无框架单文件 | 零构建、上游有现成基础 | 上游 1612 行已到可维护性上限；新增 4 个页面 + WS 状态同步 + 双语后预计 5000+ 行，手工 DOM 同步是告警状态（new/acked/dismissed + 5s undo + 批量操作）出错的主要来源 |
| Preact + htm | 3KB runtime + 免 JSX 编译（htm 用模板字符串），可以保持"改完刷新即生效"的开发体验；组件化解决状态同步 | 无类型检查（可选 JSDoc 缓解） |
| Vue 3 | 生态熟 | runtime 34KB，SFC 需构建链；若不用 SFC 则写法退化 |

决定性理由：告警工作台是一个持续接收 WS 推送、本地三态流转、支持批量+撤销的列表——这是典型的"状态驱动渲染"问题，手工 DOM 操作在这个复杂度上的缺陷已在上游体现（`clearEvents` 清空后各统计块不同步，index.html:1115 附近）。Preact+htm 无需 Node 构建链，产物就是几个 .js 文件 + 一个 vendor 文件，满足自包含约束。

其余约定：

- 图标：内联 SVG sprite，禁外链。
- 字体:系统字体栈，不打包字体文件。
- i18n：单 `i18n.js` 字典对象 + `t(key)` 函数，延续上游做法（上游已有 zh/en 字典）；语言偏好存 localStorage。
- 快照图片：`<img src="/api/alerts/<id>/snapshot.jpg">`，走同源 REST（会话 cookie 随请求自动发送），浏览器缓存兜底。
- 兼容目标：Chrome/Edge/Safari 近两年版本；不支持 IE。

## 2. 信息架构

```
hub (单 origin, cookie 会话)
├── /login       登录页       （用户名+密码 → POST /api/auth/login；任意 REST 返回 401 时跳转至此；must_change 时强制改密）
├── /            告警工作台   （值班员，默认首页）
├── /devices     设备页       （IT 管理员）
├── /rules       规则配置     （安装工/管理员，选择 device/stream 后编辑）
└── /export      （无独立页面，工作台内弹出）

设备本地 (探测器自带, 仅单机/调试)
└── :8080/debug  调试页       （安装工：探测摄像头/看画面/画规则/模拟触发）
```

导航：hub 顶栏三个 tab + 语言切换 + 当前用户名。设备本地调试页无导航，页顶注明"调试模式——生产使用请访问 hub"。

### 2.1 告警工作台线框

```
┌────────────────────────────────────────────────────────────┐
│ TopBar: [工作台] [设备] [规则]        zh/EN  user  🔔(声音开关) │
├──────────────┬─────────────────────────────────────────────┤
│ FilterPanel  │  ShiftStats: 今日 12 · 待处理 3 · 误报率 25%   │
│  状态 ▾      │ ┌─────────────────────────────────────────┐ │
│  设备 ▾      │ │ AlertCard (new, 置顶高亮)                 │ │
│  规则 ▾      │ │ [快照缩略图] zone_enter · restricted_area │ │
│  时间范围     │ │ jetson-01/cam-02 · 14:03:21              │ │
│  [导出 CSV]  │ │        [✓ 确认]  [✗ 误报]  [跳转实时画面]  │ │
│              │ ├─────────────────────────────────────────┤ │
│  批量模式 ☐  │ │ AlertCard (acked, 灰化) ...               │ │
│              │ └─────────────────────────────────────────┘ │
│              │  UndoBar: 已标记 3 条误报 [撤销] (5s)         │
└──────────────┴─────────────────────────────────────────────┘
```

组件清单：`TopBar` `FilterPanel` `ShiftStats` `AlertList` `AlertCard` `AlertDetailModal` `UndoBar` `ExportDialog` `SoundToggle` `ConnBadge`（WS 连接状态）。

### 2.2 设备页线框

```
┌────────────────────────────────────────────────────────────┐
│ DeviceTable                                                 │
│ ┌──────────┬──────┬────────┬─────┬─────────┬─────────────┐ │
│ │ device   │ 状态 │ decode │ FPS │ 版本     │ 操作         │ │
│ │ jetson-01│ ●在线│ hw 🟢  │14.8 │ 0.2.0   │[配置][本地页]│ │
│ │ rk3588-02│ ●离线│ sw 🟡  │ 6.2 │ 0.2.0   │[配置][本地页]│ │
│ └──────────┴──────┴────────┴─────┴─────────┴─────────────┘ │
│ ConfigPanel(选中行展开): 备份下载 / 恢复上传 / MQTT 凭据      │
└────────────────────────────────────────────────────────────┘
```

组件：`DeviceTable` `DeviceRow` `DecodeBadge` `ConfigPanel` `RestoreDialog`。

### 2.3 规则配置线框

```
┌────────────────────────────────────────────────────────────┐
│ StreamPicker: device ▾ / stream ▾      [模拟触发] [保存]     │
├───────────────────────────────┬────────────────────────────┤
│ RuleCanvas                    │ RuleSidebar                 │
│  (设备快照/实时帧为底图)         │  ▸ zones (list)             │
│  多边形/线段叠加层              │    restricted_area  dwell 10s│
│  工具条: [▱区域] [╱线] [删除]   │  ▸ lines (list)             │
│                               │    gate_line  方向: A→B ⇄   │
│                               │  SaveStatus: ✓已持久化 14:05 │
└───────────────────────────────┴────────────────────────────┘
```

组件：`StreamPicker` `RuleCanvas` `DrawToolbar` `ZoneList` `LineList` `DirectionToggle` `DwellInput` `SaveStatus` `SimulateButton`。

底图来源：`GET /api/devices` 透传的流 `preview_url`（设备本地单帧 JPEG 端点，status 消息可选字段），由浏览器直接向设备 origin 拉取——设备端需允许 CORS GET；hub 不代理图像。`preview_url` 缺失或不可达时降级：空画布（按 `frame.w/h` 比例的灰底）+ `GET /api/live/{device_id}/{stream_id}` 最近一条 detections 的叠加框，仍可完成绘制。

### 2.4 设备本地调试页

上游面板裁剪版：摄像头探测 + 单流预览 + 规则画布（复用 2.3 组件）+ 模拟触发。详见 §6。

## 3. 告警工作台

### 3.1 告警流

- 数据源：进入页面 REST 拉最近 200 条（`GET /api/alerts?state=new&...`），之后 WS 增量推送（§7）。
- 新告警：插入列表顶部，`new` 状态卡片带左侧色条 + 2s 高亮动画；正在浏览历史位置时不强制滚动，顶部出现"↑ 3 条新告警"悬浮条，点击回顶。
- 声音：单音效（打包本地 .mp3，<20KB）。默认开，`SoundToggle` 状态存 localStorage。浏览器自动播放策略：首次进入页面在用户任意一次点击后解锁 AudioContext，解锁前顶部展示一次性提示"点击任意处启用告警声音"。
- 浏览器通知：首次进入弹说明条（"允许通知后，页面在后台也能收到告警"）→ 用户点击"启用"才调 `Notification.requestPermission()`，不在加载时直接弹权限框。被拒绝后不再请求，设备页提供重新引导入口。

### 3.2 AlertCard 内容

| 区域 | 内容 |
|---|---|
| 左 | 快照缩略图（点击开 `AlertDetailModal` 看原图） |
| 中 | 规则事件类型（zone_enter/line_cross/loitering，图标+文案）、规则名、`device_id/stream_id`（设备显示别名）、时间（相对+绝对）、track_id、score |
| 右 | `[✓ 确认]` `[✗ 误报]` 两键 + `[实时画面]`（新窗口打开设备本地流 URL，hub 不代理视频） |

处置后卡片变灰、按钮换成状态徽标 + 处置人 + 处置时间。已处置卡片仍可反悔（徽标 hover 出"改判"）。

### 3.3 批量处置

`FilterPanel` 勾选"批量模式"→ 卡片出现 checkbox + 全选（当前筛选结果内）→ 底部操作条 `[批量确认] [批量误报]`（`POST /api/alerts/ack|dismiss {ids:[]}`，逐条结果中 409/404 项在 UI 上标注跳过原因）。执行后走 §8 的 UndoBar。批量上限单次 500 条，超出提示分批。

### 3.4 筛选与统计

- 筛选维度：状态（new/acked/dismissed/全部）、设备、stream、规则名、事件类型、时间范围（今天/24h/7 天/自定义）。筛选条件全部反映到 URL query，可收藏/分享。
- `ShiftStats`：当前筛选范围内的 总数 / 待处理 / 已确认 / 误报数与误报率。误报率超过阈值（默认 40%）时在对应规则名旁提示"建议调整阈值或区域"。

### 3.5 CSV 导出

`ExportDialog`：继承当前筛选条件，可改时间范围 → `GET /api/alerts/export.csv?...` 浏览器直接下载。列：id, ts, device_id, stream_id, event_type, rule_name, track_id, score, state, acted_by, acted_at。UTF-8 带 BOM（Excel 中文兼容）。

## 4. 规则编辑器

### 4.1 坐标系（规范性）

- 存储与传输一律 **frame_norm**：`[0,1]` 归一化，原点左上，x 右 y 下，相对**原始帧宽高**（探测器上报的 `frame.w/h`）。
- 画布换算：底图以 `object-fit: contain` 铺入画布，记 `scale = min(canvasW/frameW, canvasH/frameH)`，`offsetX = (canvasW - frameW*scale)/2`，同理 offsetY。
  - 归一化→画布：`px = nx * frameW * scale + offsetX`
  - 画布→归一化：`nx = (px - offsetX) / (frameW * scale)`，写回前 clamp 到 [0,1]。
- 上游用整帧拉伸绘制、无 letterbox 偏移，本节换算替换其实现；规则数据本身格式不变（多边形点数组 + 线段两端点），可直接迁移。

### 4.2 区域绘制（迁移）

交互沿用上游（逐点点击 → 双击/回车闭合，index.html:994-1011 `finalizeZone`），改动：

- 闭合后**不再立即生效**，进入"未保存"状态（§4.5），侧栏出现新 zone 条目，名称就地可编辑，`DwellInput` 默认 10s（仅 loitering 用，占位说明写明）。
- 顶点可拖拽微调；选中 zone 高亮 + Delete 键删除（带确认）。
- 至少 3 点；自交多边形保存时校验拒绝并提示。

### 4.3 画线工具（新增）

上游缺失（前端永远发 `lines: []`，index.html:1067），本节为全新实现：

- 工具条选 `[╱线]` → 第一次点击落 A 点，第二次点击落 B 点，成线。
- **方向语义**：与 hub 契约对齐，规则的 `direction` 取值 `forward | backward | any`（与 HUB_SPEC 规则配置、MQTT.md 事件字段同一套 token；事件上报时只会是 forward/backward——即实际发生的穿越方向）。
  - 画布渲染：`forward` 在线段中点画垂直于 start→end 的箭头（指向"从线左侧穿到右侧会告警"的判定侧，左右按 start→end 向量的叉积符号定义，见 MQTT.md）；`backward` 反向；`any` 画双向箭头。
  - `DirectionToggle` 点击循环 forward → backward → any，默认 any。
  - 前端 tooltip 用通俗文案（"仅此方向进入时告警"），叉积定义只出现在契约文档。
- 线可整体拖拽、端点拖拽；命名与删除同 zone。

### 4.4 模拟触发

`[模拟触发]` 按钮：`POST /api/rules/{device_id}/{stream_id}/simulate {rule_id}` → hub 构造一条 `event_type` 对应、`meta.simulated=true` 的告警走完整链路（入库、WS 推送、声音）。工作台卡片带"模拟"徽标。用途：画完规则 5 分钟内验证告警链路可达，不依赖真人走位。

### 4.5 保存与反馈

- 任何改动 → 顶部 `SaveStatus` 变"未保存更改"，`[保存]` 高亮；离开页面前 `beforeunload` 拦截。
- 保存：`PUT /api/rules/{device_id}/{stream_id}` 全量提交该流规则。成功 → "✓ 已持久化 HH:MM"（明确措辞含"持久化"，回应上游改内存不落盘的信任问题）。
- 失败区分：
  - 网络/5xx：保留本地编辑态，提示"保存失败：无法连接 hub，改动未丢失，可重试"。
  - 400 校验：逐条列出（"zone 'dock' 自相交" / "线 'gate' 两端点重合"），定位到侧栏对应条目标红。
- 不做自动保存：规则误触发影响生产告警，保存必须显式。

## 5. 设备页

- 数据源：`GET /api/devices`（hub 由 status topic retain + LWT 汇总）+ WS `device.status` 增量。
- 行内容：
  - 在线状态：●绿在线 / ●灰离线（LWT），离线行整体降饱和，显示最后在线时间。
  - `DecodeBadge`：`health.decode` — `hw` 绿色徽标；`sw` 黄色徽标 + tooltip"硬件解码回退到 CPU，性能受损，检查设备解码插件"。黄色状态同时计入设备页 tab 上的角标数。
  - 每流 FPS：`health.fps`，低于阈值（默认 8）标黄。
  - 版本：探测器镜像版本（status payload `version` 字段）。
  - 操作：`[配置]` 展开 ConfigPanel；`[本地页]` 新窗口打开设备本地调试页（地址取流的 `live_url`，缺失时该按钮禁用并 tooltip 提示设备固件不支持）。
- `ConfigPanel`：
  - 备份下载：`GET /api/devices/{id}/config` → JSON 文件下载（含该设备全部流的规则+摄像头配置）。
  - 恢复上传：`RestoreDialog` 选文件 → 预览 diff 摘要（zone/line/流数量变化）→ 确认后 `PUT`。
  - 通知渠道（P2）：webhook URL 配置 + 测试按钮。
- 密码修改入口（调用 hub `POST /api/auth/password`，成功后其余会话作废、当前会话保留）。

## 6. 设备本地调试页

探测器容器自带的独立页面（`:8080/debug`），供安装工无 hub 场景使用。功能为上游面板裁剪：

1. **RTSP 探测**：复用上游 probe 交互（index.html:1523-1553）：填 IP/用户名/密码 → `POST /api/cameras/probe` 逐路径试连 → 成功项列出可用 RTSP URL。
2. **首帧确认**：选中 URL → 拉取首帧缩略图展示，"确认画面正确"后写入配置。
3. **三分支失败诊断**（探测失败时按错误类型给文案）：
   - 连接超时/拒绝 → "网络不通：确认摄像头 IP 可达（同网段/网线/PoE 供电），尝试 ping {ip}"
   - 401/描述协商失败 → "用户名或密码错误：核对摄像头 Web 管理页中的 RTSP 账号"
   - 连通但无路径命中 → "URL 路径不对：常见路径已试完，查看摄像头手册中的 RTSP 地址格式"，附 Hikvision/Dahua 模板
4. **规则画布**：复用 §4 组件（同一份 js，探测器容器同样托管），保存目标为设备本地配置文件。
5. **模拟触发**：同 §4.4，本地验证 MQTT 出口（提示当前配置的 hub 地址与连通状态）。

页面顶部常驻提示："调试模式。日常值守请使用 hub 工作台：{hub_url}"（hub 地址来自设备配置，未配置则提示接入 hub 的文档链接）。

## 7. WS 契约

- 端点：`wss?://<hub>/ws`，**与 REST 同 origin 同端口**，hub 内部反代/复用 HTTP 服务器。解决上游 8081/8082 独立端口 + 冲突自动漂移、前端无从发现的问题——前端代码中不出现任何端口拼接逻辑。
- 认证：会话 cookie（HttpOnly，浏览器对同 origin WS 握手自动携带；无 Authorization 头方案——浏览器 WS API 不支持自定义头）。会话失效时握手被拒，前端跳登录页。
- 消息形状（server→client，`type` 区分）：

```json
{"type": "alert.new",    "alert": { "id": 1024, "ts": 1755400000123, "device_id": "jetson-01", "stream_id": "cam-02", "event_type": "zone_enter", "rule_name": "restricted_area", "track_id": 7, "score": 0.82, "state": "new", "snapshot_url": "/api/alerts/1024/snapshot.jpg", "simulated": false }}
{"type": "alert.update", "alert": { "id": 1024, "state": "acked", "acted_by": "operator", "acted_at": 1755400012000 }}
{"type": "device.status","device": { "device_id": "rk3588-02", "online": false, "streams": {"cam-01": {"decode": "sw", "fps": 6.2}} }}
```

- client→server 仅心跳 `{"type":"ping"}`（30s），处置操作走 REST（幂等、可重试、留审计字段）。
- 重连：指数退避 1s→2s→4s→…→30s 封顶；重连成功后以最后收到的 alert id 调 `GET /api/alerts?after_id=` 补齐 `alert.new` 缺口（id 为 hub 本地单调自增主键，升序返回，无丢单）。断线期间的 `alert.update`（快照迟到、他端改判）不在 after_id 补偿范围——重连后对当前渲染中 `snapshot_state=pending` 或缺图的告警逐条重新 `GET /api/alerts/{id}` 刷新。`ConnBadge` 三态：已连接/重连中(黄)/已断开(红+横幅"实时推送中断，告警可能延迟")。

## 8. 危险操作

| 操作 | 保护 |
|---|---|
| 清空事件（上游一键即清，index.html:1115） | 二次确认弹窗，要求输入设备名或"全部"字样；按钮文案写明不可恢复范围 |
| 批量误报/批量确认 | 执行后 `UndoBar` 底部悬浮 5s："已标记 N 条误报 [撤销]"，期间操作入暂存，5s 后提交 REST；撤销即放弃提交（单条处置同样走此机制，成本一致） |
| 规则删除 | 侧栏删除需确认；保存前可放弃全部更改（"还原为已保存版本"） |
| 配置恢复上传 | diff 摘要预览 + 确认，见 §5 |

## 9. 分期与验收

### P1（与 hub 首版同发）：告警工作台 + 规则编辑器

验收标准：

- 工作台：WS 推送的新告警 2s 内出现在列表顶部并响声；两键处置后状态即时更新且刷新页面不回退；筛选条件可从 URL 恢复；CSV 导出内容与筛选一致；WS 断开有可见提示，重连后 `after_id` 补齐无丢单。
- 规则编辑器：可完成"画一个区域 + 画一条带方向的线 + 保存 + 模拟触发收到告警"全流程 ≤5 分钟；保存成功显示"已持久化"，探测器重启后规则仍在（联动 hub 持久化验收）；校验失败能定位到具体条目；不同分辨率流上画的规则，换浏览器窗口大小后渲染位置不漂移（frame_norm 换算正确性）。
- 通用：zh/en 全量覆盖无缺翻 key；断网加载无外部资源报错（自包含验证：DevTools offline 模式全功能可用）。

### P2：设备页 + 设备本地调试页 + 统计强化

验收标准：

- 设备页：拔掉一台设备网线 ≤10s 内（LWT 超时）状态变离线；`sw` 解码徽标在人为删除设备端硬解插件后出现；配置备份→恢复到另一台同型号设备后规则一致。
- 调试页：三类故障（错 IP/错密码/错路径）分别注入，诊断文案命中对应分支；probe→首帧确认→写入配置全流程 ≤15 分钟内完成出画面旅程。
- 统计：误报率按规则聚合正确；阈值建议提示在误报率 >40% 时出现。

### 非目标（重申）

录像时间轴、NVR 存储管理、PTZ、电子地图、多租户权限、工单流转、人脸/车牌检索、hub 侧视频墙。
