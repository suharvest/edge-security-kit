# RKNN 推理内存泄漏：排查与修复复盘

2026-08-18 ~ 08-19。设备 reCamera Pro（RV1126B），`rknn_toolkit_lite2` 2.3.2。

## 症状

检测器 RSS 持续增长，不回落，也不收敛。2 GB 板子上数小时内被 OOM 杀死。

| 量 | 值 |
|---|---|
| 泄漏速率 | **43.78 kB / 次推理** |
| 与输出形状的关系 | 无关（yolov8n 检测头 9 输出与 yolo11 pose 头速率相同） |
| 18.8 fps 下 | 约 50 MB/min |

初期把它记成 "~13 MB/min"，后来发现相机路径实际跑在 18.8 fps 而不是以为的 6 fps，
所有按分钟计的数字都是速率相关的，改为按次推理表述。

## 定位

### 用对照组把范围收窄

三组实验，一次比一次窄：

| 实验 | 迭代 | kB/次 |
|---|---:|---:|
| 完整流水线 | — | 43.78 |
| 裸推理循环（固定输入，无解码/letterbox/发布） | 15 842 | 43.8 |
| **去掉推理调用的空循环** | **273 149** | **0** |

第三行是关键。二十七万次相同循环 RSS 纹丝不动，说明泄漏就在 `infer()` 调用内部，
不在帧缓冲、跟踪器、发布链路或 Python 侧的任何地方。

同时排除了两件事：泄漏不是输出缓冲（9 个反量化张量约 4.9 MB，与 43.8 kB 差三个
数量级），也不是 Python 对象未回收（`gc.tracked_objects` 平稳、`vma.mapping_count`
318→318 不动、增长全在匿名 `[heap]`、glibc free pool 反而缩小 13 MB）。是 C 级
malloc 没有配对的 free。

### 用 ctypes 把调用链切开

调用链有四层，其中两层不可读：

```
kit.runtime.engine.RknnModel.infer          Python，我们的代码
  → rknnlite.api.RKNNLite.inference          Python 壳层，只转发三个调用
    → rknn_runtime.cpython-311-*.so          Cython 二进制，无源码
      → librknnrt.so                          vendor C 库
```

用 ctypes 直接驱动 `librknnrt`，跑四个变体，每个都与 Cython 扩展执行**逐字节相同**
的 API 序列：

| 变体 | 迭代 | kB/次 | heap |
|---|---:|---:|---|
| `ctypes_leak`（阳性对照，故意省略 release） | 40 | **4801.8** | 20.9M → 217.3M |
| `ctypes_get`（`inputs_set`+`run`+`outputs_get`+`outputs_release`） | 7 185 | **0.041** | 6465 次迭代**字节级不变** |
| `ctypes_run`（只 run，不取输出） | 7 185 | 0 | 不变 |
| `ctypes_iomem`（`create_mem`+`set_io_mem` 预分配） | 2 879 | 0 | 不变 |
| rknnlite 基线 | 15 842 | **43.78** | — |

阳性对照使这张表可读：它保留 4801.8 kB/次（与九个反量化张量约 4.9 MB 吻合），证明
测量确实能检出这个量级的泄漏，所以另外三行的"平"不是假阴性。

**结论**：`ctypes_get` 与 Cython 扩展走同一条 librknnrt 路径而不泄漏，因此缺失的
`free` 在 `rknn_runtime.cpython-311-aarch64-linux-gnu.so` 内部。**改用 ctypes 是修复，
不是规避**——不需要定期重建 context，没有泄漏预算。

## 被推翻的假设

排查中有七个假设被证据推翻，记在这里比记结论有用。

| 假设 | 怎么被推翻 |
|---|---|
| 泄漏在 vendor 的 `rc_infer.cpp` probe 路径 | `librecamera_ext.so` 只导出 `rc_ext_*`，无任何 `rknn_*` 符号——它根本不在调用链上 |
| `RKNNLite.inference()` 拿了 outputs 却漏了 release | 可读的 `.py` 壳层只转发三个调用，无任何分配；且 ctypes 走同样序列不漏 |
| 泄漏与输出形状相关 | pose 模型在同一 wrapper 上以相同速率泄漏 |
| `fall-detection` 不泄漏，所以是形状相关 | 它当时**没有在推理**（无 WS 消费者），不是不泄漏 |
| `iterations: 0` 与后续运行矛盾 | 设备上跑的是守卫部署前的版本，进程被 OOM 硬杀在 30 s flush 之前，JSON 只留下 t=0 一个采样。两次行为一致，矛盾不存在 |
| 设备失联需要物理断电 | 只探了 NetBird 一条路。LAN 与 USB 一直通着 |
| `kit` 是第三方框架，只能绕过 | 它是我们自己的代码。修复落点从"我们的 app"改为 kit 层，受益方从 1 个 app 变成 7 个 |

最后一条的代价最大：在把 kit 当第三方看待的整段时间里，方案都偏向"绕过"而不是"修好"。

## 修复

`kit/runtime/ctypes_rknn.py`（新增）+ `kit/runtime/engine.py`：`RknnModel` 的内部实现
换成 ctypes 直调，对外签名与返回值语义不变，调用方一行未改。`ESK_RKNN_BACKEND=rknnlite`
保留退路，回退会打印而不是静默发生。

两个实现细节值得留意：

- 输出数据在 `rknn_outputs_release` **之前**拷贝。header 明确 release 会释放该缓冲，
  `np.frombuffer` 视图会指向已释放内存。
- `release` 放在 `finally` 块里。`_get` 与 `_release` 之间抛异常，会把这个类要修的泄漏
  换成另一条路径上的同一个泄漏。

选 `get` 而非更快的 `iomem`：后者返回原始 int8 未整形，而 `rknnlite.inference()` 返回
反量化 float32 NCHW。直接替换会**静默破坏**解码而不抛异常。`iomem` 需要数值等价测试
通过后才能作为可选项。

## 验证

**数值等价 — bit-identical**

| 计算图 | 输入 | 张量 | float32 元素 | 不同的元素 |
|---|---:|---:|---:|---:|
| yolov8n zoo | 64（60 真值帧 + 黑/白/灰/噪声） | 576 | 77 952 000 | **0** |
| yolo11n pose | 34 | 306 | 33 129 600 | **0** |

bbox delta 0.0，score delta 0.0。pose 图的输出形状不同，`decode_zoo_head` 正确拒绝它
（`no axis of (1,1,80,80) has 64 box channels`），证明新 runtime 没有把 zoo head 的假设
写死。

**长跑**：22.25 分钟真实流水线，warm-up 后 RSS **+32 kB / 25 200 次推理 = 0.0013 kB/次**，
19.1 fps（基线 18.8，无回退）。按旧速率这些推理本该增加约 1 116 MB——曲线本身排除了
"静默回退到 rknnlite"，这在 `/proc/PID/maps` 读不到时是唯一的判别手段。

**跨 app**：`fall-detection`（pose 模型，不同输出形状）在新后端下两次独立 12 分钟采样，
第二次**每个采样点都是 62 652 kB**。

改善约 33 000 倍。

## 修好之后：为什么用户还是拿不到

修复推了 GitHub、打进 v1.6.3 包，但线上配置仍指向 kit v1.6.1。发布链路有三步，
第三步历史上从未成功过：

| 步骤 | v1.6.2 | v1.6.3（修复前） |
|---|---|---|
| 打包 | ✓ | ✓ |
| 传包到 CDN | ✓ | ✗ |
| **更新 CDN 上的 `restore_config.yaml`** | **✗** | ✗ |

App 从 CDN 拉 `restore_config.yaml`（带 etag 缓存、`config_version` 单调递增校验、
fail-open 回退到内置副本）。仓库里那份只是 fallback——**改它不会让任何用户拿到新版本**。

根因很可能是：`ossutil cp` 覆盖已有对象时会**静默跳过**。返回 exit 0、打印耗时，实际是
`Upload done:(0 objects), 0.000%`——覆盖需要 `--force`，而无 stdin 时确认提示默认为否。
新对象（各版本的包）不受影响，唯独每次都要覆盖的那个 yaml 静默失败。这解释了为什么
v1.6.2 的包全在 CDN 上、配置却停在 v1.6.1。

顺带发现配置里 v1.6.2 的 frontend md5 `d0f00b05` 实际属于 **v1.6.1**——错了不止一个版本。

**送达的最终证据**不是包名或 HTTP 200，而是：从 CDN 下载线上 kit 包、解开、确认
`ctypes_rknn.py` 存在且 `engine.py` md5 为 `930aaebe`。

## 可复用的方法

1. **对照组决定结论有没有意义**。零增长对照（去掉被测调用的空循环）证明测量装置本身不漏；
   阳性对照（故意制造泄漏）证明测量能检出目标量级。缺任一个，"平"都不能说明问题。
2. **阳性对照必须限量**。这次的对照组每次保留 4.9 MB，配了 60 秒时长，十秒内吃光 1.27 GB
   把板子搞死了。对照组只需证明信号可检出，40 次迭代（约 196 MB）足够——跑满时长是纯粹的
   风险。事后补了 `max_iterations` 与每次迭代都检查的 `min_free_mb`。
3. **有源码也要验证它是否真的在调用链上**。`rc_infer.cpp` 读起来完全吻合，符号表一查
   根本不在链上。`nm -D` 比读代码快，也更可靠。
4. **设备失联时把可达路径都探一遍**。三个 watcher 只探 NetBird 一条路，据此判定"设备下线
   需要物理断电"，白折腾半小时还多上了一次电。
5. **换实现前做数值等价测试**。逐元素对比 + 解码后的 bbox 对比，两者都要。签名不变的替换
   出错时不会抛异常。
6. **修好不等于送达**。代码推了、包打了、CDN 传了，都不是用户拿到了。验证要一直做到
   "从用户会下载的那个 URL 取回字节，确认修复在里面"。

## 上游

`rknn_toolkit_lite2` 2.3.2 是最新版，也是 RV1126B 上唯一支持的版本，仓库 16 个月无新版，
changelog 无相关修复条目。公开渠道找不到等价报告；最接近的
[airockchip#42](https://github.com/airockchip/rknn-toolkit2/issues/42) 症状同类但无版本、
无平台、无量化、无分层定位，开了 22 个月零官方回复。

issue 草稿（含最小复现、ctypes 对照、阳性对照）已备好。提交前需人工确认：复现脚本在板上
实跑一遍、zbox 网盘是否有 >2.3.2 的构建、Redmine（需账号）是否已有报告。
