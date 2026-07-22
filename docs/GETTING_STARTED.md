# RUBIMPACT 用户指南

> 安装、运行、输出解读与故障排除

---

## 目录

1. [安装](#1-安装)
2. [快速开始](#2-快速开始)
3. [仿真输出](#3-仿真输出)
4. [后处理](#4-后处理)
5. [故障排除](#5-故障排除)
6. [性能调优](#6-性能调优)

---

## 1. 安装

**环境要求**：Python 3.12+（由 pixi 管理），Intel MKL，16 GB+ RAM。

```bash
pixi install
pixi run python -c "import rubimpact; print(rubimpact.__version__)"
```

---

## 2. 快速开始

### 终端启动（推荐，高性能）

项目根目录提供了两个终端启动脚本，均以高进程优先级在前台运行：

**cmd（命令提示符）：**

```bash
rmbopcl submit input/demo_contact.inp
```

在项目根目录打开终端，直接运行：

```bash
# 默认单核（安全）
pixi run rubimpact submit input/demo_contact.inp

# 多核加速
pixi run rubimpact submit input/demo_contact.inp --cores 4
```

### 多核加速

`--cores` 控制 MKL PARDISO 稀疏求解器的线程数。默认 1 核。
**上限自动检测物理核心**（非逻辑处理器）——超线程对稠密线性代数通常无收益。

- 若 `--cores` 超过物理核心数，自动截断并打印提示
- 也可设置环境变量 `RMBOPCL_CORES=4` 作为持久默认值
- `--cores` 优先级 > `RMBOPCL_CORES` > 默认值 1

输出写入 `<ModelName>/output/`。CLI 仅支持 `rubimpact submit <inp_file>` 调用方式。

INP 关键字语法和每个关键字的数学模型详见 **[关键字手册](KEYWORD_MANUAL.md)**。

### 动态松弛软启动

若初始几何存在干涉（叶尖已嵌入涂层），瞬态分析在 t=0 会产生冲击载荷。在 INP 中声明 `*DYNAMIC_RELAXATION` 可先求解静力平衡，消除启动冲击：

```inp
*DYNAMIC_RELAXATION
    max_steps=10000, tolerance=1e-10, relaxation=0.5, force_tol=1e-6
```

| 参数 | 说明 |
|------|------|
| `relaxation` (β) | 步长比例。0.3=保守，0.5=常用，0.7=快速。接触刚度大时减小。**必需参数** |
| `force_tol` | 力平衡容差。DR 收敛时残差 R = −F_contact − K·u → 0。**必需参数** |
| `max_steps` | 最大迭代步数。未收敛时打印 `[WARN]` 并继续。**必需参数** |
| `tolerance` | 位移收敛容差 ‖Δu‖∞。**必需参数** |

注释掉 `*DYNAMIC_RELAXATION` 块则跳过软启动，从零位移直接开始瞬态分析。

---

## 3. 仿真输出

### 输出目录

```
<JobName>/
├── input_copy.inp
└── output/
    ├── history/          时间序列（流式 HDF5）
    │   ├── U_TIP.h5      [N × n_tip_dof]     叶尖位移
    │   ├── PEN.h5        [N × n_tip_nodes]   穿透深度
    │   ├── CF.h5         [N × n_r]            ROM 接触力
    │   └── ENERGY.h5     [N × 4]             [KE, SE, TE, 预留]
    └── field/            场快照
        ├── COATING_H_step00000000.h5   [n_θ × n_x]
        └── COATING_EP_step00000000.h5  [n_θ × n_x]
```

### HDF5 格式

- HISTORY：gzip level 4 压缩，分块存储 (chunks=1024)，每次追加向后 resize
- FIELD：gzip level 1 压缩，固定大小

### 读取示例

```python
import h5py
with h5py.File("output/history/U_TIP.h5", "r") as f:
    u = f["U_TIP"][:]  # [N × n_tip_dof]
# 节点 0 的 y 位移：u[:, 1]
# 节点 0 的 z 位移：u[:, 2]
# 径向位移需配合初始坐标 (y0, z0) 自行换算
```

---

## 4. 后处理

- **FFT 频谱**：对叶尖径向位移做 `np.fft.rfft`，频率分辨率 = 1/(N·h)
- **磨损轮廓**：`H_final` 为剩余厚度（派生自 `ep`），或用 `EP_final`（主状态，累积塑性应变）直接可视化，extent=[0°, 360°, 0, L]
- **力-位移曲线**：`CF.h5` + `U_TIP.h5` 做 XY 图（注意 CF 符号：负值 = 推向机匣内侧）

详细的 Python 后处理代码见 **[架构指南](ARCHITECTURE.md)** 的执行流程了解各变量的物理含义。

---

## 5. 故障排除

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| `UnicodeDecodeError` | 文件编码不是 UTF-8 | MTX/CSV 使用 UTF-8 或 UTF-8-BOM；INP 使用纯 UTF-8 |
| `FileNotFoundError` | FILE= 路径不正确 | 路径相对于项目根目录 |
| 内存不足 | 全阶矩阵过大 | 启用 ROM（`*ROM, TYPE=CRAIG_BAMPTON`） |
| 时间步不收敛 | h 太大 | h < 2/ω_max ≈ T_rev/40000 |
| 接触力为 0 | 叶尖未接触 | 检查初始间隙；减小 R₀；确认干涉量 ≥ 0 |
| ROM 初始化失败 | 所有叶尖 DOF 被约束 | 检查 FE 模型的位移边界条件 |
| 矩阵维度不匹配 | 刚度矩阵来自不同网格 | 确保所有 MTX 文件从同一 FE 模型导出 |
| DR 不收敛 (max_steps exhausted) | β 太大或接触刚度远大于结构刚度 | 减小 `relaxation`（如 0.1），或增大 `max_steps` |
| DR 发散 (non-finite force) | β 过大导致位移超调 | 减小 `relaxation`，从 0.1 开始逐步增大 |

### ROM 收敛检验

逐步增加 n_modal（6 → 10 → 20 → 30），比较各 n_modal 下的 U_TIP 和能量结果。当结果不再显著变化时即为收敛。典型收敛值：n_modal ≥ 15。

### 内存估算

ROM 模式极小：n_r=44 时矩阵共 ~46 KB，涂层场 ~480 KB，总计 < 10 MB。全阶模式取决于 MTX 文件大小（典型 ~260 MB 稀疏存储）。

---

## 6. 性能调优

### 基础参数

| 调优项 | 范围 | 影响 |
|--------|------|------|
| `--cores N` | 1–物理核心数 | MKL 稀疏求解器并行度；默认 1，超出物理核自动截断 |
| `n_modal` | 6–30 | ROM 精度与速度权衡 |
| `n_theta` | 200–1080 | 涂层周向分辨率 |
| `n_x` | 20–100 | 涂层轴向分辨率 |
| `h` | T_rev/40000 | 稳定性与精度 |
| `T_f` | 10–100 周期 | 仿真物理时长 |
| `FREQUENCY` | 100–1000 (HISTORY) | I/O 开销 |

### `--cores` 选择建议

`--cores` 应基于**物理核心数**设置（`pixi run rubimpact version` 不会显示，首次运行 `submit` 时会打印检测结果）。超线程逻辑处理器对稠密线性代数无益。

| 场景 | 推荐值 | 说明 |
|------|--------|------|
| 小模型 (n_r < 20) | 1–2 | 并行开销大于收益 |
| 中型 ROM (n_r ≈ 50) | 2–物理核数 | 物理核即上限 |
| 大型 ROM / 全阶 | 物理核数 − 1 | 留 1 核给系统 I/O |

Intel 12 代+ 混合架构（P-core / E-core）建议在 BIOS 中关闭 E-core 或使用 Windows 电源计划限制计算到 P-core。

---

> **相关文档**：[关键字手册](KEYWORD_MANUAL.md) · [架构指南](ARCHITECTURE.md) · [代码更改原则](CODE_CHANGE_PRINCIPLES.md) · [模块添加指南](MODULE_ADDITION_GUIDE.md) · [参考文献](REFERENCES.md)
