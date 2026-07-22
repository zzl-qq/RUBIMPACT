# HDF5 输出数据格式参考手册

> 版本: rubimpact 2.0 | 单位制: **SI(mm)** — 长度 mm、质量 tonne、力 N、时间 s、应力 MPa
>
> 本文档完全基于 `src/rubimpact/infra/output_catalog.py` + `src/rubimpact/model_assembler.py` 源码分析生成，
> 覆盖所有已声明、可输出、以及隐藏可提取的输出量。阅读本文档即可完整了解输出数据的结构，
> 无需查看代码。

---

## 目录

1. [输出目录结构](#1-输出目录结构)
2. [HISTORY 时间序列变量](#2-history-时间序列变量)
   - [U — 叶尖节点位移](#21-u--叶尖节点位移)
   - [PEN — 穿透深度](#22-pen--穿透深度)
   - [CF — ROM 总力向量 F_total](#23-cf--rom-总力向量-f_total)
   - [ENERGY — 系统能量](#24-energy--系统能量)
   - [CFN — 法向接触力](#25-cfn--法向接触力)
   - [CFT — 摩擦力](#26-cft--摩擦力)
3. [FIELD 场快照变量](#3-field-场快照变量)
   - [COATING_H — 涂层剩余厚度场](#31-coating_h--涂层剩余厚度场)
   - [COATING_EP — 等效塑性应变场](#32-coating_ep--等效塑性应变场)
   - [COATING_S — 接触应力场](#33-coating_s--接触应力场)
4. [力分量提取指南](#4-力分量提取指南)
5. [输出系统架构](#5-输出系统架构)
6. [INP 配置语法](#6-inp-配置语法)
7. [读取辅助](#7-读取辅助)
8. [数据量估算](#8-数据量估算)

---

## 1. 输出目录结构

```
<JobName>/output/
├── history/                    时间序列（流式 HDF5，gzip level 4 压缩，chunk=1024）
│   ├── U.h5                    叶尖节点位移（每行一个时间步）
│   ├── PEN.h5                  穿透深度
│   ├── CF.h5                   ROM 总力向量
│   ├── CFN.h5                  法向接触力（按需输出）
│   ├── CFT.h5                  摩擦力（按需输出）
│   └── ENERGY.h5               系统能量（动能 + 势能）
├── field/                      场快照（gzip level 1 压缩，固定大小，每帧独立文件）
│   ├── COATING_H_step00000000.h5   涂层剩余厚度场
│   ├── COATING_EP_step00000000.h5  涂层等效塑性应变场
│   └── COATING_S_step00000000.h5   涂层接触应力场
├── job.sta                     运行状态日志（可读文本，含进度信息）
├── job.msg                     运行消息日志（可读文本）
└── job.profile                 性能分析（仅当 --profile 标志启用时生成）
```

**HDF5 写入机制**：
- HISTORY：`HDF5Writer.open_history(name, row_size)` 创建 shape=`(0, cols)` 的可扩展数据集，每行写入时通过 `dset.resize()` 向后扩展一行。使用 gzip level 4 压缩，chunk 大小为 `(1024, cols)`。
- FIELD：`HDF5Writer.dump_field(name, step, data)` 每次写入一个独立 `.h5` 文件。使用 gzip level 1 压缩。

---

## 2. HISTORY 时间序列变量

HISTORY 输出频率由 INP `*OUTPUT, TYPE=HISTORY, FREQUENCY=N` 控制（默认 `FREQUENCY=1`，即每步输出）。总行数 `N_rows = floor(T_f / h) / FREQUENCY`。

所有 HISTORY 文件的行索引严格对应同一时间轴（同一 FREQUENCY 下输出步对齐）。

### 2.1 U — 叶尖节点位移

| 属性 | 值 |
|------|-----|
| **代码定义** | `output_catalog.define("U", category="HISTORY", source="time_integrator", ndim="n_tip_dof")` |
| **数据来源** | `StateManager.u_n` → 提取 tip DOF 对应分量 |
| **`_collect` 路径** | `source="time_integrator"` → `state.u_n[tip_idx[tip_valid]]` |
| **HDF5 文件名** | `output/history/U.h5` |
| **Dataset 名** | `"U"` |
| **Shape** | `(N_steps, n_tip_dof)` |
| **dtype** | `float64` |
| **单位** | **mm** |
| **压缩** | gzip level 4, chunk=`(1024, n_tip_dof)` |

**维度说明**：
```
n_tip_dof = n_tip_nodes × 3
```
- `n_tip_nodes`：叶尖节点数（由 `主节点.csv` 定义）
- 每节点 3 个 DOF（x 轴向, y, z）

**列排布规则**：所有 tip 节点按 `主节点.csv` 中节点 ID **升序**排列，每个节点占连续 3 列：

```
列 3i+0 : 节点 i 的 x 向位移  u_x  (轴向)
列 3i+1 : 节点 i 的 y 向位移  u_y
列 3i+2 : 节点 i 的 z 向位移  u_z
```

**物理含义**：
- 存储的是 ROM 广义位移中对应**边界自由度**的分量
- 对于 Craig-Bampton 降阶（`*ROM, TYPE=CRAIG_BAMPTON`），边界 DOF 对应**物理位移**（相对于无载初始构型的偏移量）
- **不是绝对坐标**，不能直接作为节点位置使用

**HDF5 内部结构**：
```
U.h5
└── U          Dataset {N_steps, n_tip_dof}
    ├── dtype: float64
    ├── chunks: (1024, n_tip_dof)
    ├── compression: gzip (level 4)
    └── maxshape: (None, n_tip_dof)
```

**读取示例**：
```python
import h5py, numpy as np

with h5py.File("output/history/U.h5", "r") as f:
    u = f["U"][:]          # shape: (N_steps, n_tip_dof)

# 节点 0 的三向位移 (mm)
ux = u[:, 0]               # x 位移
uy = u[:, 1]               # y 位移
uz = u[:, 2]               # z 位移

# 计算物理坐标（需配合 主节点.csv 初始坐标）
y_phys = y0 + uy           # 物理 y 坐标 (mm)
z_phys = z0 + uz           # 物理 z 坐标 (mm)
r_phys = np.sqrt(y_phys**2 + z_phys**2)  # 径向坐标 (mm)
```

---

### 2.2 PEN — 穿透深度

| 属性 | 值 |
|------|-----|
| **代码定义** | `output_catalog.define("PEN", category="HISTORY", source="contact_detector", ndim="n_tip_nodes")` |
| **数据来源** | `ContactDetector.detect()` → `_pen_buf.copy()` → `db.set("penetration", ...)` |
| **`_collect` 路径** | `source="contact_detector"` → `db.get("penetration")` |
| **HDF5 文件名** | `output/history/PEN.h5` |
| **Dataset 名** | `"PEN"` |
| **Shape** | `(N_steps, n_tip_nodes)` |
| **dtype** | `float64` |
| **单位** | **mm** |
| **压缩** | gzip level 4, chunk=`(1024, n_tip_nodes)` |

**列排布**：列 `i` = 第 `i` 个叶尖节点（按节点 ID 升序），与 U 的节点顺序一致。

```
列 0 : 节点 0 的穿透量 δ₀
列 1 : 节点 1 的穿透量 δ₁
...
列 n_tip_nodes-1 : 最后一个节点的穿透量
```

**穿透定义（gap function 计算）**：
```
δ = max(0, r_phys - (R_casing - h_loc))
```
- `r_phys`：叶尖节点当前径向坐标 `sqrt(y_phys² + z_phys²)`
- `R_casing`：机匣在当前 (x, θ) 处的内半径
- `h_loc`：涂层在该位置的当前剩余厚度（通过 bilinear 插值获得）
- **δ > 0 表示发生接触**，δ = 0 表示未接触

**HDF5 内部结构**：
```
PEN.h5
└── PEN        Dataset {N_steps, n_tip_nodes}
    ├── dtype: float64
    ├── chunks: (1024, n_tip_nodes)
    ├── compression: gzip (level 4)
    └── maxshape: (None, n_tip_nodes)
```

**读取示例**：
```python
with h5py.File("output/history/PEN.h5", "r") as f:
    pen = f["PEN"][:]        # shape: (N_steps, n_tip_nodes)

# 节点 0 的穿透 (mm)
pen_node0 = pen[:, 0]
# 统计接触占比
contact_ratio = (pen > 0).sum(axis=0) / pen.shape[0]  # 每个节点的接触时间比例
```

---

### 2.3 CF — ROM 总力向量 F_total

> ⚠️ **重要：CF 输出的是 `F_total`（总力），不是纯接触力。**

| 属性 | 值 |
|------|-----|
| **代码定义** | `output_catalog.define("CF", category="HISTORY", source="force_assembler", ndim="n_r")` |
| **数据来源** | `ForceAssembler.assemble()` → `F_total` → `db.set("F_total", F_total)` |
| **`_collect` 路径** | `source="force_assembler"` → `db.get("F_total")` |
| **HDF5 文件名** | `output/history/CF.h5` |
| **Dataset 名** | `"CF"` |
| **Shape** | `(N_steps, n_r)` |
| **dtype** | `float64` |
| **单位** | **N** |
| **压缩** | gzip level 4, chunk=`(1024, n_r)` |

**维度说明**：
```
n_r = n_b + n_k
```
- `n_b = n_tip_nodes × 3`：边界 DOF 数（= U 的列数）
- `n_k = n_modal`：保留的固定界面模态数（由 INP `*ROM, n_modal=N` 指定）

**列排布**：
```
列 0 .. n_b-1    : 边界 DOF 上的力分量（物理力，与 U 列一一对应）
列 n_b .. n_r-1  : 模态力分量（广义力，对应于保留的固定界面模态）
```

#### CF 的精确组成

CF 不是纯接触力，而是 **F_total**，由以下分量求和得到（来自 `ForceAssembler.assemble()`）：

```
F_total = Σ F_contact_normal_i  +  Σ F_contact_friction_i  +  Σ F_other_modules
           \________  ________/     \__________  __________/     \_______  _______/
                    ↓                           ↓                         ↓
              法向接触力                  摩擦力（切向）              其他力模块
           (PCL/Penalty → F_n)     (Coulomb/Stribeck → F_t)    (aero, inertial, …)
```

**详细计算链**（`src/rubimpact/runtime/force_assembler.py:140-195`）：

```
对每个接触节点 i:
  1. F_n = contact_force.compute({delta, h_loc, ep_loc, ...})
     ├─ PCL_CONTACT: σ = f_PCL(δ, h, ep, α; E, Y, K_plas),  F_n = A_cell × σ
     └─ PENALTY:     F_n = k_penalty × δ

  2. if F_n ≤ 0: continue (跳过该节点)

  3. F_t = friction_force.compute(F_n, v_rel)
     ├─ COULOMB:  F_t = μ × F_n
     └─ STRIBECK: F_t = μ(v_rel) × F_n

  4. 力分解为 ROM DOF 分量:
     Fy = (F_n × yc − F_t × zc) / r    ← 法向力 + 摩擦力混合
     Fz = (F_n × zc + F_t × yc) / r    ← 法向力 + 摩擦力混合

  5. 累加到 F_total:
     F_total[dof_idx[i, 0]] += 0.0      (轴向力 = 0)
     F_total[dof_idx[i, 1]] += Fy
     F_total[dof_idx[i, 2]] += Fz

对所有非接触力模块:
  F_total += other_module.assemble(context)
  ├─ AerodynamicForce (STEADY_AERO): 当前为 placeholder → zeros
  └─ InertialForce (RIGID_ROTATION): 当前为 placeholder → zeros
```

**关键结论**：
- **当前 `F_total` 中法向力和摩擦力已混合在 Fy/Fz 分量中，无法从 CF 单独分离**
- 当前 aero/inertial 模块为 placeholder（返回零向量），所以实际上 CF ≈ 接触力+摩擦力
- 未来如果启用 aero/inertial，CF 会包含更多物理成分

**力的符号约定**（来自 Central Difference 的 corrector）：
```
u_{n+1} = u_p − A⁻¹ · F_total

CF > 0: blade 受到使位移减小的力
CF < 0: blade 受到使位移增大的力
```

**提取单节点力分量**：
```python
i = 0                        # 节点索引
Fx = cf[:, i*3 + 0]          # x 向力 (N) — 通常为 0
Fy = cf[:, i*3 + 1]          # y 向力 (N) — 混合法向+摩擦
Fz = cf[:, i*3 + 2]          # z 向力 (N) — 混合法向+摩擦
```

**HDF5 内部结构**：
```
CF.h5
└── CF         Dataset {N_steps, n_r}
    ├── dtype: float64
    ├── chunks: (1024, n_r)
    ├── compression: gzip (level 4)
    └── maxshape: (None, n_r)
```

**读取示例**：
```python
with h5py.File("output/history/CF.h5", "r") as f:
    cf = f["CF"][:]          # shape: (N_steps, n_r)

# 节点 i 的力分量（每节点 3 DOF）
i = 0
Fx = cf[:, i*3 + 0]          # x 向力 (N) — 接触力轴向分量为 0
Fy = cf[:, i*3 + 1]          # y 向力 (N) — 法向+摩擦混合
Fz = cf[:, i*3 + 2]          # z 向力 (N) — 法向+摩擦混合

# 模态力分量
n_b = n_tip_nodes * 3
F_modal = cf[:, n_b:]        # shape: (N_steps, n_modal)
```

---

### 2.4 ENERGY — 系统能量

| 属性 | 值 |
|------|-----|
| **代码定义** | `output_catalog.define("ENERGY", category="HISTORY", source="step_kernel", ndim=4)` |
| **数据来源** | `OutputDispatcher.write_history()` 中**直接计算**（不通过 `_collect`） |
| **`write_history` 路径** | 特殊分支：读取 `rom.M_r`、`rom.K_r`、`current_h`、`state.u_n`、`state.u_nm1` 直接计算 |
| **HDF5 文件名** | `output/history/ENERGY.h5` |
| **Dataset 名** | `"ENERGY"` |
| **Shape** | `(N_steps, 4)` |
| **dtype** | `float64` |
| **单位** | **mJ** (N·mm = 10⁻³ J) |
| **压缩** | gzip level 4, chunk=`(1024, 4)` |

**列定义**：
```
列 0 : KE  — 动能 (Kinetic Energy)
列 1 : SE  — 势能 (Strain/Potential Energy)
列 2 : TE  — 总能量 (Total Energy) = KE + SE
列 3 : 预留 — 当前恒为 0.0（实现中未赋值）
```

**计算公式**（来自 `output_catalog.py:167-170`）：
```python
v = (state.u_n - state.u_nm1) / h      # 速度 (mm/s)
KE = 0.5 * v^T @ M_r @ v               # 动能 = ½ vᵀ M_r v
SE = 0.5 * state.u_n^T @ K_r @ state.u_n  # 势能 = ½ uᵀ K_r u
TE = KE + SE
```

**注意**：
- `ENERGY` 不通过 `_collect()` 获取数据，而是 `write_history()` 中的特殊路径
- 列 3（dissipation / 耗散能）当前始终为 `0.0`，未被赋值
- 速度由中心差分近似：`v ≈ (u_n - u_{n-1}) / h`

**HDF5 内部结构**：
```
ENERGY.h5
└── ENERGY     Dataset {N_steps, 4}
    ├── dtype: float64
    ├── chunks: (1024, 4)
    ├── compression: gzip (level 4)
    └── maxshape: (None, 4)
```

**读取示例**：
```python
with h5py.File("output/history/ENERGY.h5", "r") as f:
    energy = f["ENERGY"][:]  # shape: (N_steps, 4)

KE = energy[:, 0]            # 动能 (mJ)
SE = energy[:, 1]            # 势能 (mJ)
TE = energy[:, 2]            # 总能量 (mJ)
# energy[:, 3] 当前恒为 0
```

---

### 2.5 CFN — 法向接触力

| 属性 | 值 |
|------|-----|
| **代码定义** | `output_catalog.define("CFN", category="HISTORY", source="force_assembler", ndim="n_r")` |
| **数据来源** | `ForceAssembler.assemble()` → `F_normal` → `db.set("F_normal", F_normal)` |
| **`_collect` 路径** | `source="force_assembler"`, `CFN` → `db.get("F_normal")` |
| **HDF5 文件名** | `output/history/CFN.h5` |
| **Dataset 名** | `"CFN"` |
| **Shape** | `(N_steps, n_r)` |
| **dtype** | `float64` |
| **单位** | **N** |
| **压缩** | gzip level 4, chunk=`(1024, n_r)` |

**维度说明**：
```
n_r = n_b + n_k = n_tip_nodes × 3 + n_modal
```

**物理含义**：
- **纯法向接触力**在 ROM 空间的投影，不包含摩擦力、气动力或惯性力
- 仅包含 `F_n`（接触对法向力标量）的 Y/Z 分量分解
- X 向（轴向）力始终为 0

**计算链**：
```
对每个接触节点 i:
  F_n = contact_force.compute({delta, h_loc, ep_loc, ...})
  Fy_n = F_n × yc / r  →  法向力的 y 分量
  Fz_n = F_n × zc / r  →  法向力的 z 分量
  F_normal[节点 i 的 DOF] += (0, Fy_n, Fz_n)
```

**列排布**：与 CF 完全一致 — 前 `n_b` 列为边界 DOF 力分量，后 `n_k` 列为模态力分量。

**按需分配**：`CFN` 仅在 INP 中请求时才分配 buffer 并写入 DataBus。未请求时 overhead 为零。

**HDF5 内部结构**：
```
CFN.h5
└── CFN        Dataset {N_steps, n_r}
    ├── dtype: float64
    ├── chunks: (1024, n_r)
    ├── compression: gzip (level 4)
    └── maxshape: (None, n_r)
```

---

### 2.6 CFT — 摩擦力

| 属性 | 值 |
|------|-----|
| **代码定义** | `output_catalog.define("CFT", category="HISTORY", source="force_assembler", ndim="n_r")` |
| **数据来源** | `ForceAssembler.assemble()` → `F_friction` → `db.set("F_friction", F_friction)` |
| **`_collect` 路径** | `source="force_assembler"`, `CFT` → `db.get("F_friction")` |
| **HDF5 文件名** | `output/history/CFT.h5` |
| **Dataset 名** | `"CFT"` |
| **Shape** | `(N_steps, n_r)` |
| **dtype** | `float64` |
| **单位** | **N** |
| **压缩** | gzip level 4, chunk=`(1024, n_r)` |

**物理含义**：
- **纯摩擦力**在 ROM 空间的投影（切向摩擦力），不包含法向力、气动力或惯性力
- 仅包含 `F_t`（摩擦力标量）的 Y/Z 分量分解
- 摩擦力方向与法向力垂直（周向偏转 90°）

**计算链**：
```
对每个接触节点 i:
  F_t = friction_force.compute(F_n, v_rel)  ← Coulomb 或 Stribeck
  Fy_t = -F_t × zc / r  →  摩擦力的 y 分量（法向力旋转 90°）
  Fz_t =  F_t × yc / r  →  摩擦力的 z 分量（法向力旋转 90°）
  F_friction[节点 i 的 DOF] += (0, Fy_t, Fz_t)
```

**列排布**：与 CF/CFN 完全一致。

**按需分配**：`CFT` 仅在 INP 中请求时才分配 buffer 并写入 DataBus。未请求时 overhead 为零。

**HDF5 内部结构**：
```
CFT.h5
└── CFT        Dataset {N_steps, n_r}
    ├── dtype: float64
    ├── chunks: (1024, n_r)
    ├── compression: gzip (level 4)
    └── maxshape: (None, n_r)
```

**CF/CFN/CFT 验证关系**：
```python
# 任意步：CF = CFN + CFT（aero/inertial 当前为 placeholder 时）
np.allclose(cf[k], cfn[k] + cft[k])   # True
```

---

## 3. FIELD 场快照变量

FIELD 输出为独立 HDF5 文件，文件名包含步数序号。输出频率由 `*OUTPUT, TYPE=FIELD, FREQUENCY=N` 控制。每帧输出调用 `HDF5Writer.dump_field(name, step_idx, data)`。

**文件名规则**：`{NAME}_step{step_idx:08d}.h5`，例如 `COATING_H_step00000100.h5`。

### 3.1 COATING_H — 涂层剩余厚度场

| 属性 | 值 |
|------|-----|
| **代码定义** | `output_catalog.define("COATING_H", category="FIELD", source="coating", ndim="(n_theta,n_x)")` |
| **数据来源** | `db.get("coating.h")` → 涂层厚度二维网格 |
| **`_collect` 路径** | `source="coating"`, `COATING_H` → `db.get("coating.h")` |
| **HDF5 文件名** | `field/COATING_H_step{step_idx:08d}.h5` |
| **Dataset 名** | `"COATING_H"` |
| **Shape** | `(n_theta, n_x)` |
| **dtype** | `float64` |
| **单位** | **mm** |
| **压缩** | gzip level 1 |

**网格维度**：
- `n_theta`：周向网格点数（INP `*COATING` 中 `n_theta` 参数）
- `n_x`：轴向网格点数（INP `*COATING` 中 `n_x` 参数）

**物理含义**：
- 涂层网格单元中的**当前剩余厚度** `h_loc(θ, x)`
- 由磨损模块在每步力计算中**原地修改**（in-place mutation）
- `h_loc = h_coat · (1 − ε_p)`（当使用 PLASTIC_STRAIN 磨损律时）
- 初始值 = `h_coat`（标称涂层厚度）
- `h_loc = 0` 表示涂层已完全磨穿（`ε_p = 1.0`）

**网格坐标映射**：
```python
θ[i] = i * 2π / n_theta      # i ∈ [0, n_theta), 单位 rad
x[j] = j * L / n_x           # j ∈ [0, n_x), 单位 mm
```

**HDF5 内部结构**：
```
COATING_H_step00000100.h5
└── COATING_H  Dataset {n_theta, n_x}
    ├── dtype: float64
    └── compression: gzip (level 1)
```

**读取示例**：
```python
with h5py.File("output/field/COATING_H_step00000100.h5", "r") as f:
    h_step100 = f["COATING_H"][:]     # shape: (n_theta, n_x)
```

---

### 3.2 COATING_EP — 等效塑性应变场

| 属性 | 值 |
|------|-----|
| **代码定义** | `output_catalog.define("COATING_EP", category="FIELD", source="coating", ndim="(n_theta,n_x)")` |
| **数据来源** | `db.get("coating.ep")` → 塑性应变二维网格 |
| **`_collect` 路径** | `source="coating"`, `COATING_EP` → `db.get("coating.ep")` |
| **HDF5 文件名** | `field/COATING_EP_step{step_idx:08d}.h5` |
| **Dataset 名** | `"COATING_EP"` |
| **Shape** | `(n_theta, n_x)` |
| **dtype** | `float64` |
| **单位** | **无量纲** (mm/mm) |
| **压缩** | gzip level 1 |

**物理含义**：
- 涂层材料的**累积塑性应变** `ε_p(θ, x)`，**主状态变量**
- 由 PCL (Plastic Coating Law) 返回映射算法计算
- 磨损律 `PLASTIC_STRAIN` 在每个接触步原地修改此网格
- **单调递增**，初始值 = 0，上界 1.0
- 剩余厚度严格派生：`h_loc = h_coat · (1 − ε_p)`
- `α`（磨损比）当前实现中与 `ε_p` 同步更新（`coating_alpha[jt, jx] = ep_c`）

**HDF5 内部结构**：
```
COATING_EP_step00000100.h5
└── COATING_EP  Dataset {n_theta, n_x}
    ├── dtype: float64
    └── compression: gzip (level 1)
```

**读取示例**：
```python
with h5py.File("output/field/COATING_EP_step00000100.h5", "r") as f:
    ep_step100 = f["COATING_EP"][:]   # shape: (n_theta, n_x)

# 磨损深度 = ep * h_coat
wear_depth = ep_step100 * h_coat  # mm
```

---

### 3.3 COATING_S — 接触应力场

| 属性 | 值 |
|------|-----|
| **代码定义** | `output_catalog.define("COATING_S", category="FIELD", source="coating", ndim="(n_theta,n_x)")` |
| **数据来源** | `ForceAssembler.assemble()` → 按接触网格单元累积正应力 `sigma` |
| **`_collect` 路径** | `source="coating"`, `COATING_S` → `db.get("coating.s")` |
| **HDF5 文件名** | `field/COATING_S_step{step_idx:08d}.h5` |
| **Dataset 名** | `"COATING_S"` |
| **Shape** | `(n_theta, n_x)` |
| **dtype** | `float64` |
| **单位** | **MPa** |
| **压缩** | gzip level 1 |

**物理含义**：
- 涂层网格单元上的**累积接触正应力** `σ(θ, x)`
- 由 `ForceAssembler` 在每个接触步中将每个接触的 `sigma` 值通过 bilinear 权重分配到涂层网格，并在同一时间步的多个接触间**累加**
- 仅在有接触发生的网格单元中非零

**计算链**（`ForceAssembler.assemble()` + `_distribute_stress()`）：
```
对每个接触节点 i:
  F_n, sigma, dgamma = contact_force.compute(c)
  if F_n > 0:
      _distribute_stress(stress_acc, sigma, c)
          → 将 sigma 按 bilinear 权重 (jt, jx, w1..w4) 分配到
             stress_acc[θ网格索引, x网格索引] 的 4 个邻接单元

每步结束时: db.set("coating.s", stress_acc)
```

**注意**：
- 应力累积 buffer 在每步开始时清零，不会跨步累积
- 同一时间步内多个接触节点的应力贡献会叠加
- 该网格坐标与 `COATING_H`/`COATING_EP` 完全相同（同一 `n_theta × n_x` 网格）

**HDF5 内部结构**：
```
COATING_S_step00000100.h5
└── COATING_S  Dataset {n_theta, n_x}
    ├── dtype: float64
    └── compression: gzip (level 1)
```

**读取示例**：
```python
with h5py.File("output/field/COATING_S_step00000100.h5", "r") as f:
    s_step100 = f["COATING_S"][:]   # shape: (n_theta, n_x) 应力 (MPa)
```

---

## 4. 力分量提取指南

### 4.1 力分量关系

CF/CFN/CFT 三者在 ROM 空间维度一致（均为 `(N_steps, n_r)`），列排布完全对齐。在 aero/inertial 模块为 placeholder（返回零向量）的当前状态下：

```
CF = CFN + CFT    （逐分量精确可加，见 §2.6 验证关系）
```

### 4.2 提取单节点力

```python
import h5py, numpy as np

out_dir = "DemoContact/output/history"

with h5py.File(f"{out_dir}/CF.h5", "r") as f:
    cf = f["CF"][:]
with h5py.File(f"{out_dir}/CFN.h5", "r") as f:
    cfn = f["CFN"][:]
with h5py.File(f"{out_dir}/CFT.h5", "r") as f:
    cft = f["CFT"][:]

# 节点 i 的三分量（法向力 + 摩擦力分离）
i = 0
# 总力
Fx_total = cf[:, i*3 + 0]          # x 向（恒为 0）
Fy_total = cf[:, i*3 + 1]          # y 向 = Fy_n + Fy_t
Fz_total = cf[:, i*3 + 2]          # z 向 = Fz_n + Fz_t

# 法向力分量
Fy_n = cfn[:, i*3 + 1]             # y 向法向力
Fz_n = cfn[:, i*3 + 2]             # z 向法向力

# 摩擦力分量
Fy_t = cft[:, i*3 + 1]             # y 向摩擦力
Fz_t = cft[:, i*3 + 2]             # z 向摩擦力

# 验证
assert np.allclose(Fy_total, Fy_n + Fy_t)
assert np.allclose(Fz_total, Fz_n + Fz_t)
```

### 4.3 计算合力模量

```python
# 节点 i 的总合力模量
F_mag = np.sqrt(cf[:, i*3 + 1]**2 + cf[:, i*3 + 2]**2)   # 总力 (N)

# 法向力模量
Fn_mag = np.sqrt(cfn[:, i*3 + 1]**2 + cfn[:, i*3 + 2]**2)

# 摩擦力模量
Ft_mag = np.sqrt(cft[:, i*3 + 1]**2 + cft[:, i*3 + 2]**2)
```

### 4.4 模态力分量

所有力变量共享相同的模态自由度：

```python
n_b = n_tip_nodes * 3          # 边界 DOF 数
n_modal = n_r - n_b

# 模态力分量
F_modal_cf  = cf[:, n_b:]       # (N_steps, n_modal)  总力模态分量
F_modal_cfn = cfn[:, n_b:]      # (N_steps, n_modal)  法向力模态分量
F_modal_cft = cft[:, n_b:]      # (N_steps, n_modal)  摩擦力模态分量
```

### 4.5 按需激活

CFN 和 CFT 仅在 INP 中显式请求时才分配 buffer 并写入 DataBus。未请求时，`ForceAssembler.assemble()` 保持最小开销路径（仅维护 F_total 一个 buffer）。

```inp
*STEP, NAME=Demo
    Omega=1600.0, h=1.0e-8, T_f=0.03

    *OUTPUT, TYPE=HISTORY, FREQUENCY=100
        U,
        PEN,
        CF,           # 总力（向后兼容，始终可用）
        CFN,          # 法向接触力（按需激活）
        CFT,          # 摩擦力（按需激活）
```

---

## 5. 输出系统架构

### 5.1 核心类

| 类 | 文件 | 职责 |
|----|------|------|
| `OutputCatalog` | `src/rubimpact/infra/output_catalog.py:17` | 变量目录：`define(name, category, source, ndim, description)` 注册标准输出变量 |
| `OutputDispatcher` | `src/rubimpact/infra/output_catalog.py:80` | 输出调度：`initialize(writer, db)` → `write_history(var, db, state)` / `write_field(var, db, step)` |
| `HDF5Writer` | `src/rubimpact/infra/hdf5_writer.py:7` | HDF5 写入：`open_history()` → `append_history()` (流式) / `dump_field()` (快照) |
| `DataBus` | `src/rubimpact/infra/databus.py:5` | 全局键值存储：`set(key, value)` / `get(key, default)` |
| `StateManager` | `src/rubimpact/infra/state_manager.py:5` | 时间步状态：`u_n`, `u_nm1`, `t`, `step` |

### 5.2 数据流

```
每步主循环 (model_assembler._run_loop):
  ┌──────────────────────────────────────────────────────┐
  │ 1. ti.predict(u_n, u_nm1) → u_p                      │
  │ 2. cd.contacts_as_list(u_p, t, Omega)                │
  │    ├─ cd.detect(u_p, t, Omega, copy_to_bus=False)    │
  │    │   ├─ kinematics → coords_buf (n_nodes, 5)       │
  │    │   ├─ interpolator → interp_buf (n_nodes, 9)     │
  │    │   └─ gap_function → pen_buf (n_nodes,)          │
  │    └─ 构建 contact dicts (仅接触节点)                 │
  │ 3. fa.assemble(context, requested) → F_total (n_r,)   │
  │    db.set("F_total", F_total)                        │
  │    db.set("F_normal", F_normal)   ← 仅当 CFN 请求时    │
  │    db.set("F_friction", F_friction) ← 仅当 CFT 请求时  │
  │    db.set("coating.s", stress_acc) ← 仅当 COATING_S 请求时 │
  │    db.set("current_h", h)                             │
  │ 4. ti.correct(u_p, F_total) → u_new                  │
  │ 5. state.advance(u_new, h)                           │
  │ 6. 输出（按 FREQUENCY）:                              │
  │    od.write_history(var, db, state)                  │
  │    od.write_field(var, db, step_idx)                 │
  └──────────────────────────────────────────────────────┘
```

**注意**：在热循环中 `detect()` 的 `copy_to_bus=False`（避免分配），仅在输出步骤时 contact detector 的 buffer 数据通过 `contacts_as_list` 内部的 `detect` 调用更新且 force_assembler 写入 `F_total`。`PEN` 变量依赖的 `db.get("penetration")` 是通过 `contacts_as_list` → `detect(copy_to_bus=True)` 写入的。

### 5.3 `_collect` 分发逻辑

```python
def _collect(self, source: str, db, state, var_name: str):
    if source == "time_integrator":
        # 从 state.u_n 提取 tip DOF 分量 → (n_tip_dof,)
        row = np.zeros(self._n_tip_dof, dtype=np.float64)
        row[self._tip_valid] = state.u_n[self._tip_idx[self._tip_valid]]
        return row

    elif source == "force_assembler":
        # 按变量名分发到不同的 DataBus key
        mapping = {"CF": "F_total", "CFN": "F_normal", "CFT": "F_friction"}
        return db.get(mapping.get(var_name.upper(), "F_total"))

    elif source == "contact_detector":
        # 穿透深度 → (n_tip_nodes,)
        return db.get("penetration")

    elif source == "coating":
        # 涂层场数据 → (n_theta, n_x)
        field_map = {
            "COATING_H":  "coating.h",
            "COATING_EP": "coating.ep",
            "COATING_S":  "coating.s",
        }
        return db.get(field_map.get(var_name.upper(),
                                   f"coating.{var_name.lower()}"))

    elif source == "step_kernel":
        # 能量数据 → (4,)
        # 注：ENERGY 实际不经过此路径，在 write_history 中特殊处理
        return db.get(f"energy.{var_name.lower()}")
```

### 5.4 添加新输出变量的步骤

若需要在代码中注册新的输出变量：

```python
# 在 output_catalog.py 中添加 define 调用
output_catalog.define("COORDS", category="HISTORY", source="contact_detector",
                      ndim=(5,), description="Tip node coordinates [theta, x, yc, zc, r]")
```

并在 `OutputDispatcher._collect()` 中添加对应的 `source` 分支。

---

## 6. INP 配置语法

### 6.1 基础语法

```inp
*STEP, NAME=StepName
    Omega=1600.0, h=1.0e-8, T_f=0.03

    *OUTPUT, TYPE=HISTORY, FREQUENCY=100
        U,
        PEN,
        CF,
        ENERGY,

    *OUTPUT, TYPE=FIELD, FREQUENCY=1000
        COATING_H,
        COATING_EP,
        COATING_S,
```

### 6.2 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `TYPE` | — (必填) | `HISTORY` 或 `FIELD` |
| `FREQUENCY` | `1` | 输出间隔（每 N 步输出一次）。`FREQUENCY=100` 表示每 100 步输出一个数据行/帧 |
| `variables` | — | 变量名列表（逗号分隔，大小写不敏感） |

### 6.3 变量名速查

| INP 变量名 | 大小写不敏感 | 类别 | 说明 |
|-----------|-------------|------|------|
| `U` | `u`, `U` | HISTORY | 叶尖节点位移 |
| `PEN` | `pen`, `Pen` | HISTORY | 穿透深度 |
| `CF` | `cf`, `Cf` | HISTORY | ROM 总力向量 F_total |
| `CFN` | `cfn`, `Cfn` | HISTORY | 法向接触力 F_normal |
| `CFT` | `cft`, `Cft` | HISTORY | 摩擦力 F_friction |
| `ENERGY` | `energy`, `Energy` | HISTORY | 系统能量 (KE, SE, TE) |
| `COATING_H` | `coating_h`, ... | FIELD | 涂层剩余厚度场 |
| `COATING_EP` | `coating_ep`, ... | FIELD | 等效塑性应变场 |
| `COATING_S` | `coating_s`, ... | FIELD | 涂层接触应力场 |

### 6.4 解析逻辑（来自 `KeywordParser.parse()`）

```python
# *OUTPUT, TYPE=HISTORY, FREQUENCY=100
#   → current_output = {"TYPE": "HISTORY", "FREQUENCY": "100", "variables": []}
# 后续行（不以 * 或 = 开头）→ 添加变量名到 variables 列表

# 运行时解析 (model_assembler._run_loop):
#   otype = out.get("TYPE", "").upper()    → "HISTORY" 或 "FIELD"
#   freq  = int(out.get("FREQUENCY", "1"))
#   vars  = out.get("variables", [])       → ["U", "PEN", "CF"]
```

---

## 7. 读取辅助

### 7.1 时间轴重建

所有 HISTORY 文件的行索引对应同一时间轴：

```python
import numpy as np

h = 1e-8           # 时间步长 (s)，来自 INP *STEP, h=...
freq = 100          # 输出频率，来自 INP *OUTPUT, FREQUENCY=...
N_steps = u.shape[0]  # HISTORY 行数

t = np.arange(N_steps) * h * freq   # 时间轴 (s)
```

### 7.2 节点顺序

所有输出中节点顺序一致，按 `主节点.csv` 中的**节点 ID 升序**排列。示例（8 节点工况）：

```
列索引 0-2   → FE node 3921  (节点 0: ux, uy, uz)
列索引 3-5   → FE node 15757 (节点 1: ux, uy, uz)
列索引 6-8   → FE node 18311 (节点 2: ux, uy, uz)
列索引 9-11  → FE node 21416 (节点 3: ux, uy, uz)
列索引 12-14 → FE node 24028 (节点 4: ux, uy, uz)
列索引 15-17 → FE node 25727 (节点 5: ux, uy, uz)
列索引 18-20 → FE node 28820 (节点 6: ux, uy, uz)
列索引 21-23 → FE node 29503 (节点 7: ux, uy, uz)
```

### 7.3 单位制速查

| 物理量 | 单位 | 备注 |
|--------|------|------|
| 长度/位移 | mm | U, PEN, COATING_H |
| 力 | N | CF, CFN, CFT |
| 质量 | tonne (10³ kg) | 隐含于 M 矩阵 |
| 时间 | s | h, T_f, t |
| 频率 | Hz | FFT 结果 |
| 应力/模量 | MPa (N/mm²) | E, Y, K_plas |
| 角度 | rad | Omega, theta |
| 能量 | mJ (N·mm) | ENERGY |
| 密度 | tonne/mm³ | 隐含于质量矩阵 |
| 应变 | 无量纲 | COATING_EP |

### 7.4 完整读取示例

```python
import h5py
import numpy as np
from pathlib import Path

out_dir = Path("DemoContact/output")

# === 加载所有 HISTORY 数据 ===
with h5py.File(out_dir / "history/U.h5", "r") as f:
    u = f["U"][:]           # (N, n_tip_dof)
with h5py.File(out_dir / "history/PEN.h5", "r") as f:
    pen = f["PEN"][:]        # (N, n_tip_nodes)
with h5py.File(out_dir / "history/CF.h5", "r") as f:
    cf = f["CF"][:]          # (N, n_r)
with h5py.File(out_dir / "history/CFN.h5", "r") as f:
    cfn = f["CFN"][:]        # (N, n_r)  法向接触力
with h5py.File(out_dir / "history/CFT.h5", "r") as f:
    cft = f["CFT"][:]        # (N, n_r)  摩擦力
with h5py.File(out_dir / "history/ENERGY.h5", "r") as f:
    energy = f["ENERGY"][:]  # (N, 4)

# === 时间轴 ===
h = 1e-8; freq = 100
N = u.shape[0]
t = np.arange(N) * h * freq  # (s)

# === 节点 0 分析 ===
ux, uy, uz = u[:, 0], u[:, 1], u[:, 2]        # 位移 (mm)
pen_node0 = pen[:, 0]                           # 穿透 (mm)
cf_x = cf[:, 0]                                  # 总力 x 向 (N)
cfn_x = cfn[:, 0]                                # 法向力 x 向 (N)
cft_x = cft[:, 0]                                # 摩擦力 x 向 (N)

# === 物理坐标（需初始坐标） ===
y0, z0 = 0.648, 270.0  # mm, 来自 主节点.csv
y_phys = y0 + uy
z_phys = z0 + uz
r_phys = np.sqrt(y_phys**2 + z_phys**2)

# === 能量 ===
KE = energy[:, 0]  # 动能 (mJ)
SE = energy[:, 1]  # 势能 (mJ)
TE = energy[:, 2]  # 总能量 (mJ)

# === FFT 频谱 ===
from numpy.fft import rfft, rfftfreq
dr = r_phys - r_phys[0]                      # 径向位移变化 (mm)
fs = 1.0 / (h * freq)                        # 采样频率 (Hz)
freqs = rfftfreq(N, d=1.0/fs)
amp = np.abs(rfft(dr)) / N * 2               # 幅值 (mm)
```

### 7.5 读取 FIELD 快照

```python
# 读取特定步的场快照
step = 100
with h5py.File(out_dir / f"field/COATING_H_step{step:08d}.h5", "r") as f:
    coating_h = f["COATING_H"][:]      # (n_theta, n_x) 涂层厚度 (mm)

with h5py.File(out_dir / f"field/COATING_EP_step{step:08d}.h5", "r") as f:
    coating_ep = f["COATING_EP"][:]    # (n_theta, n_x) 塑性应变 (无量纲)

with h5py.File(out_dir / f"field/COATING_S_step{step:08d}.h5", "r") as f:
    coating_s = f["COATING_S"][:]      # (n_theta, n_x) 接触应力 (MPa)

# 遍历所有场快照
field_dir = out_dir / "field"
h_files = sorted(field_dir.glob("COATING_H_step*.h5"))
for hf in h_files:
    with h5py.File(hf, "r") as f:
        # dataset name matches the filename stem pattern
        dset_name = list(f.keys())[0]
        data = f[dset_name][:]
```

---

## 8. 数据量估算

以 Demo 工况为例（`h=1e-8`, `T_f=0.03`, `FREQUENCY=100`, `n_modal=20`, `n_tip_nodes=8`）：

| 文件 | N_steps | 列数 | 单行字节 | 未压缩大小 | 压缩后估计 |
|------|---------|------|---------|-----------|-----------|
| U.h5 | 30,000 | 24 | 192 B | ~5.5 MB | ~3 MB |
| CF.h5 | 30,000 | 44 (24+20) | 352 B | ~10.1 MB | ~6 MB |
| CFN.h5 | 30,000 | 44 (24+20) | 352 B | ~10.1 MB | ~6 MB |
| CFT.h5 | 30,000 | 44 (24+20) | 352 B | ~10.1 MB | ~6 MB |
| PEN.h5 | 30,000 | 8 | 64 B | ~1.8 MB | ~1 MB |
| ENERGY.h5 | 30,000 | 4 | 32 B | ~0.9 MB | ~0.5 MB |

**FIELD 快照**（每帧，以 `n_theta=1080, n_x=60` 为例）：

| 文件 | Shape | 单帧大小 |
|------|-------|---------|
| COATING_H_step*.h5 | (1080, 60) | ~0.5 MB/帧 |
| COATING_EP_step*.h5 | (1080, 60) | ~0.5 MB/帧 |
| COATING_S_step*.h5 | (1080, 60) | ~0.5 MB/帧 |

> 注：HDF5 使用 gzip 压缩，实际文件大小取决于数据的可压缩性。稀疏接触工况（如仅少数节点接触）的 PEN 和 CF 压缩率极高。
