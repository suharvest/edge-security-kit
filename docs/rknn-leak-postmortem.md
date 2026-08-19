# RKNN 推理内存泄漏：排查与修复

设备 reCamera Pro（RV1126B），`rknn_toolkit_lite2` 2.3.2，`librknnrt` 2.3.2，NPU 驱动 0.9.8。

## 症状

检测器 RSS 持续增长，不回落也不收敛，2 GB 板子上数小时内被 OOM 杀死。

| 量 | 值 |
|---|---|
| 泄漏速率 | 43.78 kB / 次推理 |
| 与输出形状的关系 | 无关，yolov8n 检测头（9 输出）与 yolo11 pose 头速率相同 |
| 18.8 fps 下 | 约 50 MB/min |

按次推理而非按分钟计：相机路径的帧率随负载变化，速率相关的数字无法横向比较。

## 定位

### 一、用对照组把范围收窄

| 实验 | 迭代 | kB/次 |
|---|---:|---:|
| 完整流水线 | — | 43.78 |
| 裸推理循环（固定输入，无解码/letterbox/发布） | 15 842 | 43.8 |
| 去掉推理调用的空循环 | 273 149 | 0 |

第三行把泄漏限定在 `infer()` 调用内部：二十七万次相同循环 RSS 不变，排除了帧缓冲、
跟踪器、发布链路与 Python 侧。

同时排除两条：

- **不是输出缓冲**。九个反量化张量约 4.9 MB，与 43.8 kB 差三个数量级。
- **不是 Python 对象未回收**。`gc.tracked_objects` 平稳，`vma.mapping_count` 318→318
  不变，增长全部落在匿名 `[heap]`，glibc free pool 反而缩小 13 MB。是 C 级 malloc
  没有配对的 free。

### 二、用 ctypes 切开调用链

调用链四层，中间两层不可读：

```
kit.runtime.engine.RknnModel.infer          Python
  → rknnlite.api.RKNNLite.inference          Python 壳层，只转发三个调用
    → rknn_runtime.cpython-311-*.so          Cython 二进制，无源码
      → librknnrt.so                          vendor C 库
```

用 ctypes 直接驱动 `librknnrt`，四个变体各自执行与 Cython 扩展逐字节相同的 API 序列：

| 变体 | 迭代 | kB/次 | heap |
|---|---:|---:|---|
| `ctypes_leak`（阳性对照，故意省略 release） | 40 | 4801.8 | 20.9M → 217.3M |
| `ctypes_get`（`inputs_set`+`run`+`outputs_get`+`outputs_release`） | 7 185 | 0.041 | 6465 次迭代字节级不变 |
| `ctypes_run`（只 run，不取输出） | 7 185 | 0 | 不变 |
| `ctypes_iomem`（`create_mem`+`set_io_mem` 预分配） | 2 879 | 0 | 不变 |
| rknnlite 基线 | 15 842 | 43.78 | — |

阳性对照保留 4801.8 kB/次，与九个反量化张量约 4.9 MB 吻合，说明测量能检出这个量级，
另外三行的"平"不是假阴性。

`ctypes_get` 与 Cython 扩展走同一条 librknnrt 路径而不泄漏，因此缺失的 `free` 在
`rknn_runtime.cpython-311-aarch64-linux-gnu.so` 内部。**改用 ctypes 是修复而非规避**：
不需要定期重建 context，没有泄漏预算。

## 排查中被证据推翻的技术假设

| 假设 | 推翻依据 |
|---|---|
| 泄漏在 vendor `rc_infer.cpp` 的 probe 路径 | `librecamera_ext.so` 只导出 `rc_ext_*`，无任何 `rknn_*` 符号，不在调用链上 |
| `RKNNLite.inference()` 拿了 outputs 却漏了 release | 可读的 `.py` 壳层只转发三个调用、无分配；ctypes 走同样序列不泄漏 |
| 泄漏与输出形状相关 | pose 模型在同一 wrapper 上以相同速率泄漏 |
| `fall-detection` 长跑不涨，故与形状有关 | 该进程当时没有在推理（无 WS 消费者） |
| 首次实验 `iterations: 0` 与后续运行矛盾 | 进程被 OOM 硬杀在 30 s flush 之前，JSON 只留下 t=0 一个采样，两次行为一致 |

`nm -D` 查符号表比通读源码更快也更可靠：`rc_infer.cpp` 读起来与症状完全吻合，
但它编译出的库根本不在调用链上。

## 修复

新增 `kit/runtime/ctypes_rknn.py`，改 `kit/runtime/engine.py`：`RknnModel` 内部实现换成
ctypes 直调，对外签名与返回值语义不变，调用方未改。`ESK_RKNN_BACKEND=rknnlite` 保留退路，
回退时打印而非静默发生。

两处实现细节：

- 输出数据在 `rknn_outputs_release` **之前**拷贝。header 明确 release 会释放该缓冲，
  `np.frombuffer` 视图会指向已释放内存。
- `release` 放在 `finally` 块内。`_get` 与 `_release` 之间抛异常会把泄漏换到另一条路径上。

选 `get` 而非更快的 `iomem`：后者返回原始 int8 未整形，`rknnlite.inference()` 返回反量化
float32 NCHW，直接替换会静默破坏解码而不抛异常。`iomem` 需数值等价测试通过后才能启用。

## 验证

**数值等价，bit-identical**

| 计算图 | 输入 | 张量 | float32 元素 | 不同的元素 |
|---|---:|---:|---:|---:|
| yolov8n zoo | 64（60 真值帧 + 黑/白/灰/噪声） | 576 | 77 952 000 | 0 |
| yolo11n pose | 34 | 306 | 33 129 600 | 0 |

bbox delta 0.0，score delta 0.0。pose 图输出形状不同，`decode_zoo_head` 正确拒绝
（`no axis of (1,1,80,80) has 64 box channels`），说明新 runtime 未内嵌 zoo head 假设。

**长跑**：22.25 分钟真实流水线，warm-up 后 RSS +32 kB / 25 200 次推理 = 0.0013 kB/次，
19.1 fps（基线 18.8）。按旧速率这些推理本应增加约 1 116 MB，曲线本身即可排除静默回退到
rknnlite——`/proc/PID/maps` 对非特权账号不可读时，这是唯一的判别手段。

**跨 app**：`fall-detection`（pose 模型，不同输出形状）在新后端下两次独立 12 分钟采样，
第二次每个采样点均为 62 652 kB。

## 测量方法

1. **零增长对照**：去掉被测调用的空循环，证明测量装置本身不泄漏。缺它则"平"无意义。
2. **阳性对照**：故意制造已知量级的泄漏，证明测量能检出目标量级。缺它则"平"可能是假阴性。
3. **阳性对照必须限量**。故意不释放的变体每次保留约 4.9 MB，按时长运行会在十秒内耗尽
   1.27 GB 可用内存。对照只需证明信号可检出，40 次迭代（约 196 MB）足够；配 `max_iterations`
   与每次迭代都检查的 `min_free_mb`（4.9 MB/次时，32 次一检查就是 157 MB 盲区）。
4. **换实现前做数值等价测试**：逐元素对比 + 解码后 bbox 对比。签名不变的替换出错时不抛异常。

## 独立复现

`tools/rknn-leak-repro/` 是不依赖本仓库的两个脚本，拷到任意 RKNPU 板子上直接跑：

| 脚本 | 路径 | 依赖 |
|---|---|---|
| `leak_repro.py` | `rknnlite.api.RKNNLite`，`load_rknn` + `init_runtime` 各一次，稳定 `inference()` 循环 | `numpy` + `rknn_toolkit_lite2` |
| `ctypes_control.py` | ctypes 直调 `librknnrt.so`，与 Cython 扩展逐字节相同的 API 序列 | `numpy` + `ctypes` |

```bash
python3 leak_repro.py     --model yolov8n.rknn --seconds 300 --sample-every 30
python3 ctypes_control.py --model yolov8n.rknn --seconds 300 --sample-every 30
python3 ctypes_control.py --model yolov8n.rknn --omit-release   # 阳性对照，内建限量
```

两个脚本用同一套 warmup、采样间隔与 kB/次算法，数字可直接对比；输入尺寸与输出张量数
从模型查询得到，不硬编码。`--omit-release` 每次保留约 4.6 MB，默认 40 次迭代封顶，
并在每次迭代后检查 `MemAvailable`。

RK3588（librknnrt 2.3.2、驱动 0.9.8、rknn_toolkit_lite2 2.3.2，与 RV1126B 同版本）
实跑结果：

| 模式 | 迭代 | kB/次 |
|---|---:|---:|
| `leak_repro.py` | 13 359 | 42.4876 |
| `ctypes_control.py` | 14 551 | 0.0542 |
| `ctypes_control.py --omit-release` | 40（限量停住） | 4639.7 |

完整原始输出、`/dev/rknpu` 权限问题与取得 root 的两种方式见
`tools/rknn-leak-repro/README.md`。

## 上游状态

`rknn_toolkit_lite2` 2.3.2 是最新版，也是 RV1126B 上唯一支持的版本；仓库 16 个月无新版本，
changelog 无相关修复条目；板上 `librknnrt` 与驱动由固件提供，换 runtime 需动固件层。

公开渠道未见等价报告。最接近的
[airockchip#42](https://github.com/airockchip/rknn-toolkit2/issues/42) 症状同类，但无版本、
无平台、无量化数据、无分层定位，开启 22 个月无官方回复。Rockchip Redmine 需账号，未覆盖。

issue 草稿（含最小复现、ctypes 对照、阳性对照）引用 `tools/rknn-leak-repro/` 下的脚本，
三种模式已在 RK3588 上实跑，输出见该目录的 README。RV1126B 上的复现待设备可用时补测。
