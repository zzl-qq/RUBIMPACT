# RUBIMPACT 关键字手册

> 每个仿真关键字的 INP 语法、参数说明及底层数学/物理模型。**人读文档**——一眼看懂每个关键字在算什么。

---

## 目录

1. [INP 语法规则](#1-inp-语法规则)
2. [`*MODEL`](#2-model)
3. [`*EXTERNAL_DATA`](#3-external_data)
4. [`*CASING`](#4-casing)
5. [`*COATING`](#5-coating)
6. [`*MATRIX_ASSEMBLY`](#6-matrix_assembly)
7. [`*ROM`](#7-rom)
8. [`*CONTACT_DETECTOR`](#8-contact_detector)
9. [`*CONSTITUTIVE`](#9-constitutive)
10. [`*TIME_INTEGRATOR`](#10-time_integrator)
11. [`*FORCE_ASSEMBLER`](#11-force_assembler)
12. [`*DYNAMIC_RELAXATION`](#12-dynamic_relaxation)
13. [`*STEP` / `*OUTPUT`](#13-step--output)
14. [仿真流程](#14-仿真流程)
15. [参数速查表](#15-参数速查表)
16. [文件格式规范](#16-文件格式规范)

---

## 1. INP 语法规则

- 关键字以 `*` 开头，大小写不敏感（内部转为大写）
- 参数使用 `key=value` 格式，逗号分隔
- 缩进表示嵌套层级：0 空格 = 顶层关键字，4 空格 = 一级子模块 (`*SUBKEYWORD`)，8 空格 = 二级子模块 (`*NESTED_SUB`)
- 子模块关键字也以 `*` 开头，通过缩进层级区分层级
- `#` 开头的行视为注释
- `*END STEP` 终止当前分析步；`*END MODEL` 终止整个模型定义

### 子模块语法

子模块通过 `*SUBKEYWORD` + 缩进声明，替代旧的逗号前缀写法：

```
*TOP_KEYWORD, TYPE=<TYPE>
    *SUBKEYWORD1, TYPE=<TYPE>, param=value
    *SUBKEYWORD2, TYPE=<TYPE>
        *NESTED_SUB, TYPE=<TYPE>, param=value
    param=value
```

- 顶层关键字的直接参数（如 `E=3500.0, Y=10.0`）写在 4 空格缩进，不用 `*` 前缀
- 子模块关键字（如 `*HARDENING`, `*CONTACT_FORCE`）必须使用 `*` 前缀 + 4 空格缩进
- 更深层嵌套（如有）使用 `*` 前缀 + 8 空格缩进

---

## 2. `*MODEL`

定义模型名称，即作业输出文件夹名。

```
*MODEL, NAME=<作业名>
```

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `NAME` | str | 是 | 作业名，输出写入 `<NAME>/output/` |

---

## 3. `*EXTERNAL_DATA`

导入外部 FE 矩阵文件（MTX COO 格式）和叶尖节点坐标（CSV）。

```
*EXTERNAL_DATA
    *MATRIX, TYPE=MTX_COO, ROLE=MASS,      FILE=<path>
    *MATRIX, TYPE=MTX_COO, ROLE=STIFFNESS, FILE=<path>, Omega=<转速>
    *NODES,  TYPE=COORD,    ROLE=TIP,      FILE=<path>
```

> `*MATRIX` 和 `*NODES` 使用 `*` 前缀 + 4 空格缩进，与其他子模块关键字语法一致。

### *MATRIX 子关键字

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `TYPE` | enum | 是 | `MTX_COO` |
| `ROLE` | enum | 是 | `MASS` 或 `STIFFNESS` |
| `FILE` | path | 是 | 相对项目根目录的 MTX 文件路径 |
| `Omega` | float | STIFFNESS 时必需 | 该刚度矩阵对应的转速 (rad/s)，供离心刚化插值 |

- `ROLE=MASS`：只应出现一次。若出现多次，最后声明的生效。
- `ROLE=STIFFNESS`：可出现 1–3 次（不同 Ω），供 `*MATRIX_ASSEMBLY` 中 `TYPE=CENTRIFUGAL_POLY` 插值使用。

### *NODES 子关键字

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `TYPE` | enum | 是 | `COORD` |
| `ROLE` | enum | 是 | `TIP` |
| `FILE` | path | 是 | CSV 文件路径（`node_id,x,y,z`，无表头，# 注释行忽略） |

### DataBus 写入

| 键 | 内容 |
|----|------|
| `matrices.mass` | 全阶质量矩阵 M |
| `matrices.stiffness` | `{Ω: SparseMatrix}` 映射 |
| `nodes.tip` | `{node_id: (x, y, z)}` |

---

## 4. `*CASING`

定义机匣内表面几何形状。由轴向形状和周向形状正交组合：
**R(x, θ) = R₀ + f_axial(x) + f_circ(θ)**

```
*CASING
    R0=<float>
    *AXIAL_SHAPE, TYPE=<CYLINDRICAL | AXIAL_TAPER>, [slope=<float>]
    *CIRCUMFERENTIAL_SHAPE, TYPE=<UNIFORM | LOBE>, [N_lobe=<int>, d0=<float>]
```

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `R0` | float | 是 | 基准半径，mm |

### *AXIAL_SHAPE — 子模块（必需）

| TYPE | f(x) | 额外参数 |
|------|------|----------|
| `CYLINDRICAL` | f(x) = 0 | 无 |
| `AXIAL_TAPER` | f(x) = slope · x | `slope` (float) — 轴向斜率。slope < 0 = 收缩锥 |

### *CIRCUMFERENTIAL_SHAPE — 子模块（可选，默认 UNIFORM）

| TYPE | f(θ) | 额外参数 |
|------|------|----------|
| `UNIFORM` | f(θ) = 0（轴对称，默认） | 无 |
| `LOBE` | f(θ) = d₀ · sin(N_lobe · θ) | `N_lobe` (int) — 瓣数；`d0` (float) — 幅值 |

### 示例

**收缩圆锥 + 三瓣变形**：
```
*CASING
    R0=270.5
    *AXIAL_SHAPE, TYPE=AXIAL_TAPER, slope=-0.056
    *CIRCUMFERENTIAL_SHAPE, TYPE=LOBE, N_lobe=3, d0=0.5
```
→ R(x,θ) = 270.5 − 0.056·x + 0.5·sin(3θ)

**纯圆柱**（周向默认 UNIFORM）：
```
*CASING
    R0=272.0
    *AXIAL_SHAPE, TYPE=CYLINDRICAL
```
→ R(x,θ) = 272.0

### DataBus 写入

| 键 | 内容 |
|----|------|
| `casing.geometry` | 含 `get_radius(x, theta) → float` 和 `R_grid(theta, x) → ndarray` 的字典 |
| `casing.R_grid` | (n_θ, n_x) 预计算半径网格（涂层存在时填充） |

---

## 5. `*COATING`

定义可磨耗涂层的计算网格及初始状态。

```
*COATING, TYPE=UNIFORM_GRID
    h_coat=<厚度>, L=<长度>
    n_theta=<周向格数>, n_x=<轴向格数>
```

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `TYPE` | enum | 是 | `UNIFORM_GRID` — 当前唯一实现 |
| `h_coat` | float | 是 | 涂层名义厚度，mm |
| `L` | float | 是 | 涂层轴向长度，mm |
| `n_theta` | int | 是 | 周向网格数 |
| `n_x` | int | 是 | 轴向网格数 |

> **无默认值**：所有参数必须在 INP 中显式声明。

### 派生网格参数

```
Δθ = 2π / n_θ         周向格距 (rad)
Δx = L / n_x           轴向格距 (mm)
A_cell = R₀ · Δθ · Δx  格点面积 (mm²)
```

### 涂层状态（随磨损演化）

**ep**（累积塑性应变）是主状态变量，单调递增。`h` 和 `alpha` 均为派生量：

| 数组 | Shape | 初始值 | 更新 |
|------|-------|--------|------|
| `coating.ep` | (n_θ, n_x) | 0 | 每步 `ep += dgamma`（主状态，单调不减） |
| `coating.h` | (n_θ, n_x) | h_coat | `h = h_coat · (1 − ep)`（派生） |
| `coating.alpha` | (n_θ, n_x) | 0 | `alpha = ep`（派生，线性各向同性硬化） |

### DataBus 写入

`coating.grid`, `coating.h`, `coating.ep`, `coating.alpha`

### 无涂层运行模式

不声明 `*COATING` 和 `*CONSTITUTIVE` 时，框架自动进入无涂层模式：

- 间隙公式简化为 `gap = R_casing − r`
- 无磨损计算
- 法向力必须使用 `TYPE=PENALTY`（PCL_CONTACT 需要涂层）
- 输出中无 EP/H 场变量

---

## 6. `*MATRIX_ASSEMBLY`

组装全阶质量、刚度、阻尼矩阵并写入 DataBus。

```
*MATRIX_ASSEMBLY
    *MASS,      TYPE=<类型>
    *STIFFNESS, TYPE=<类型>
    *DAMPING,   TYPE=<类型>, <参数>=<值>
```

> 子关键字使用 `*` 前缀 + 4 空格缩进。

### *MASS — TYPE 选项（必需声明）

| TYPE | 说明 |
|------|------|
| `ORIGINAL` | 直接使用 `*EXTERNAL_DATA` 中 `ROLE=MASS` 的矩阵 |

### *STIFFNESS — TYPE 选项（必需声明）

| TYPE | 说明 |
|------|------|
| `DIRECT` | 直接使用声明的刚度矩阵（单转速） |
| `CENTRIFUGAL_POLY` | 用离心刚化多项式插值 K(Ω)（多转速） |

**离心刚化多项式插值**：

**3 个转速**（Ω=[0, Ω_half, Ω_max]）：
```
K(Ω) = K₀ + Ω²·K₁ + Ω⁴·K₂

K₁ = (16K_half − K_max − 15K₀) / (3Ω²_max)
K₂ = 4(K_max − 4K_half + 3K₀) / (3Ω⁴_max)
```

**2 个转速**（Ω=[0, Ω_max]）：
```
K(Ω) = K₀ + Ω²·K₁
K₁ = (K_max − K₀) / Ω²_max
```

**1 个转速**：直接使用。

### *DAMPING — TYPE 选项（必需声明）

| TYPE | 数学公式 | 必需参数 |
|------|----------|------|
| `RAYLEIGH` | D = α · M + β · K(Ω) | `alpha` (float), `beta` (float) |
| `NONE` | D = 0 | 无 |

---

## 7. `*ROM`

Craig-Bampton 固定界面子结构模态综合。将全阶系统（~10⁵ DOF）投影到缩减子空间（~10² DOF）。

```
*ROM, TYPE=CRAIG_BAMPTON, n_modal=<保留模态数>
```

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `TYPE` | enum | 是 | `CRAIG_BAMPTON` |
| `n_modal` | int | 是 | 保留的固定界面正则模态数 |

**ROM 维度**：n_r = n_b + n_modal，其中 n_b = 3 × （自由叶尖节点数）为界面 DOF。示例：8 节点，20 模态 → n_r = 44。

### Craig-Bampton 步骤

**Step 1 — DOF 划分**：边界 b = 叶尖 DOF（k_diag < 1e30），内部 i = 其余自由 DOF。

**Step 2 — 约束模态**：Ψ_c = −Kᵢᵢ⁻¹ Kᵢ_b（PARDISO 稀疏求解，多 RHS）

**Step 3 — 固定界面模态**：Kᵢᵢ φ = ω² Mᵢᵢ φ，取最小 n_modal 个（shift-invert ARPACK, σ=0）

**Step 4 — CB 变换矩阵**：
```
        ┌         ┐
Φ_CB =  │ I    0  │  (n_free × n_r)
        │ Ψ_c  Φ_n│
        └         ┘
```

**Step 5 — ROM 投影**：对 X ∈ {M, K, D}，X_r = Φ_CBᵀ · X · Φ_CB

### ROM vs 全阶

| 指标 | 全阶 | ROM (n_modal=20) |
|------|------|------------------|
| DOF | ~103,902 | 44 |
| 矩阵存储 | ~86 GB (稠密假设) | ~15 KB |

**无 `*ROM` 关键字时的处理**：自动回退到全阶模式（仅在小型测试系统中实用）。

---

## 8. `*CONTACT_DETECTOR`

在每时间步检测叶尖节点是否与机匣涂层表面发生穿透。子模块通过 JIT 注册表动态分派。

```
*CONTACT_DETECTOR, TYPE=PENETRATION_BASED
    *KINEMATICS, TYPE=RIGID_ROTATION_PLUS_VIBRATION
    *INTERPOLATOR, TYPE=BILINEAR
    *GAP_FUNCTION, TYPE=DEFAULT
```

### 运动学

```
θ = Ω·t + θ₀      周向位置（旋转 + 初始偏置）
x = x₀ + u_x       轴向位置（初始 + 振动位移）
y_c = y₀ + u_y     y 坐标
z_c = z₀ + u_z     z 坐标
r = √(y_c² + z_c²)  径向坐标
```

### 间隙函数

```
g(θ, x, t) = R(x, θ) − (h_coat − Δw) − r(t)
           = R(x, θ) − h_loc − r(t)

g < 0  →  穿透，δ = −g
g ≥ 0  →  无接触
```

其中 `R(x,θ)` 为机匣内半径（任意注册的 CASING 类型），`h_loc = h_coat − Δw` 为涂层剩余厚度（双线性插值）。

### interpolator, TYPE=BILINEAR

双线性插值（默认）。使用 2×2=4 个角点模板，C⁰ 连续。

叶尖处的 `(θ, x)` 映射到网格索引 `(i_θ, i_x)`，四个角点权重 `w00, w10, w01, w11`。h_loc、ep_loc、alpha_loc 均由角点值加权得到。

### interpolator, TYPE=BICUBIC_BSPLINE

双三次均匀 B 样条插值。使用 4×4=16 个控制点模板，C² 连续。

- 控制点为涂层网格 `(n_θ × n_x)` 的全部节点
- θ 方向周期性 wrap，x 方向边界截断 (`i_x ≤ n_x - 4`)
- 要求 `n_θ ≥ 4`, `n_x ≥ 4`
- 声明此类型后，磨损写回和应力分布自动切换为 16 格点模式
- 基函数：N₀(u)=(1-u)³/6, N₁(u)=(3u³-6u²+4)/6, N₂(u)=(-3u³+3u²+3u+1)/6, N₃(u)=u³/6

示例：
    *CONTACT_DETECTOR, TYPE=PENETRATION_BASED
       interpolator, TYPE=BICUBIC_BSPLINE
       kinematics, TYPE=RIGID_ROTATION_PLUS_VIBRATION
       gap_function, TYPE=DEFAULT

### 机匣半径计算

在 Python 层（非 JIT）通过 `get_radius(x, θ)` 逐节点计算后传入 Numba 热循环，任何注册的 CASING 类型均无需修改 JIT 代码。

### DataBus 读取/写入

**读取**：`casing.geometry`, `coating.h/ep/alpha`, `rom.tip_dof_map`

**写入**：`penetration`（OutputCatalog 变量 PEN 的数据源）

---

## 9. `*CONSTITUTIVE`

塑性涂层本构（Plastic Coating Law, PCL）——一维标量返回映射，含线性各向同性硬化。子模块通过 JIT 注册表分派。

```
*CONSTITUTIVE, TYPE=PLASTIC_COATING_LAW
    E=<弹性模量>, Y=<屈服应力>
    *HARDENING, TYPE=LINEAR_ISOTROPIC, K_plas=<塑性模量>
    *WEAR_LAW, TYPE=PLASTIC_STRAIN
    *WEAR_DISTRIBUTOR, TYPE=BILINEAR_WEIGHT
    *STATE_UPDATER, TYPE=PLASTIC_STRAIN_RATIO
```

### 顶层参数（全部必需）

| 参数 | 类型 | 说明 |
|------|------|------|
| `E` | float | 涂层弹性模量，MPa |
| `Y` | float | 初始屈服应力，MPa |

### *HARDENING 子模块（必需）

| TYPE | 屈服函数 | 必需参数 |
|------|---------|------|
| `LINEAR_ISOTROPIC` | σ_y(α) = Y + K_plas · α | `K_plas` (float) — 塑性模量，MPa |

### PCL 计算流程

**Step 0 — 应变加固**：
```
δ_eff = min(δ, h_loc)         防止 δε > 1.0 发散
Δε = δ_eff / h_loc            (h_loc > 0；否则 σ = 0)
```

**Step 1 — 弹性预测**：
```
σ_trial = E · Δε
```

**Step 2 — 屈服判断**：
```
f_trial = σ_trial − (Y + K_plas · α)
f_trial ≤ 0  →  弹性：σ = σ_trial, dw = 0, 完成
f_trial > 0   →  塑性：进入 Step 3
```

**Step 3 — 塑性修正（返回映射）**：
```
Δγ = f_trial / (E + K_plas)       塑性乘子增量
σ   = σ_trial − E · Δγ            修正后应力
ε_p = ε_p_loc + Δγ                更新塑性应变
α   = α_loc + Δγ                  更新硬化变量
```

**Step 4 — 状态更新（ep-master）**：
```
Δγ = ε_p − ε_p_loc                 塑性应变增量（主状态）
ep_corner += Δγ · w                按双线性权重累加到网格角点（上界 1.0）
h_loc = h_coat · (1 − ep_corner)   派生：剩余厚度
```
磨损深度 `dw = Δγ · h_coat` 由名义厚度换算，确保物理体积去除量独立于已磨损程度。**ep 作为主状态保证了单调性和 h/ep 的严格同步**——不再出现两变量独立更新导致的失配。

### 子模块类型速查

| 槽位 | TYPE 选项（全部必需声明） | 说明 |
|------|-----------|------|
| `*HARDENING` | `LINEAR_ISOTROPIC` | 线性各向同性硬化 |
| `*WEAR_LAW` | `PLASTIC_STRAIN` / `NONE` | PLASTIC_STRAIN=塑性应变驱动磨损；NONE=不考虑磨损，仅计算接触力。**省略时默认为 NONE**（不启用磨损） |
| `*WEAR_DISTRIBUTOR` | `BILINEAR_WEIGHT` | 双线性权重分配到涂层四角点 |
| `*STATE_UPDATER` | `PLASTIC_STRAIN_RATIO` | h = h_coat · (1 − ep)；ep 为主状态 |

---

## 10. `*TIME_INTEGRATOR`

显式中心差分时间推进。**predictor 和 corrector 子模块 TYPE 必须全部显式声明**。

```
*TIME_INTEGRATOR, TYPE=CENTRAL_DIFFERENCE
    *PREDICTOR, TYPE=LINEAR
    *CORRECTOR, TYPE=CONTACT_CONSTRAINED
```

### 中心差分公式

**预测步**：
```
A = M_r/h² + D_r/(2h)                          系统矩阵
b_n = (2M_r/h² − K_r)·u_n + (D_r/(2h) − M_r/h²)·u_{n−1}
u_p = A⁻¹ · b_n                                 预测位移
```

**修正步**：
```
u_{n+1} = u_p − A⁻¹ · F_total
```

其中 F_total 为 `*FORCE_ASSEMBLER` 组装的所有力贡献之和（接触力 + 气动力 + 惯性力等）。

**换步**：
```
u_{n−1} ← u_n,   u_n ← u_{n+1}
```

**稳定性准则**：h < 2 / ω_max，其中 ω_max 为 ROM 系统的最高固有频率。

### DataBus 读取

`rom.M_r`, `rom.K_r`, `rom.D_r`

---

## 11. `*FORCE_ASSEMBLER`

每时间步组装所有力贡献的总和 F_total。ForceAssembler 是编排器——其子模块 `*CONTACT_FORCE`（法向接触力 + 磨损）和 `*FRICTION_FORCE`（摩擦力）是**平级子模块**，不再有嵌套的 `*CONTACT_FORCE` 中间层。法向力与切向力的 Type 分派通过 JITRegistry，其他力模块（气动力、惯性力）通过 ModuleRegistry 的 `force_module` 类别分派。

```
*FORCE_ASSEMBLER
    *CONTACT_FORCE, TYPE=PCL_CONTACT
    *FRICTION_FORCE, TYPE=COULOMB, mu=<摩擦系数>
```

也可添加可选的力模块（气动力、惯性力）：
```
*FORCE_ASSEMBLER
    *CONTACT_FORCE, TYPE=PCL_CONTACT
    *FRICTION_FORCE, TYPE=STRIBECK, mu_s=0.3, mu_d=0.25, v_s=1.0
    *AERODYNAMIC_FORCE, TYPE=STEADY_AERO
    *INERTIAL_FORCE, TYPE=RIGID_ROTATION
```

### *CONTACT_FORCE — PCL_CONTACT

每个接触节点计算法向接触力。`*CONTACT_FORCE` 直接读取 DataBus 中的涂层数据（`coating.grid`, `coating.h`, `coating.ep`, `coating.alpha`），执行 PCL 返回映射计算接触应力，并将磨损增量写回涂层网格。

**法向力**（CELL_AREA_WEIGHTED）：
```
F_n = A_cell · σ
```
σ 来自 JIT 本构内核（`jit_registry.get("constitutive", "PLASTIC_COATING_LAW")`）。

**磨损写回**：塑性应变增量 `dgamma` 按双线性权重分配到涂层网格四角点，累加到 `coating.ep`（上界 1.0）。`coating.h` 和 `coating.alpha` 从 `ep` 派生。此职责属于 `*CONTACT_FORCE`——不是 ForceAssembler 编排器。

### *CONTACT_FORCE — PENALTY

罚函数法向接触力。无需涂层和材料本构。

```
*FORCE_ASSEMBLER
    *CONTACT_FORCE, TYPE=PENALTY, k_penalty=<float>
    *FRICTION_FORCE, TYPE=COULOMB, mu=<float>
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `k_penalty` | float | 是 | 罚刚度 (N/mm)，法向力 F_n = k_penalty × δ |

**使用条件**：
- 不需要 `*COATING` 块
- 不需要 `*CONSTITUTIVE` 块
- 摩擦模型正常使用

**间隙公式**（无涂层）：`gap = R_casing − r`（无 h_coat 扣除项）

### *FRICTION_FORCE — COULOMB / STRIBECK（必需）

切向摩擦力，`F_n` 和 `v_rel` 的纯函数，无 DataBus 依赖。

| TYPE | 数学公式 | 必需参数 |
|------|----------|------|
| `COULOMB` | F_t = μ · F_n | `mu` (float) — 库仑摩擦系数 |
| `STRIBECK` | μ(v) = μ_d + (μ_s − μ_d) · exp(−\|v\|/v_s)；F_t = μ(v)·F_n | `mu_s` (float) — 静摩擦系数；`mu_d` (float) — 动摩擦系数；`v_s` (float) — Stribeck 特征速度 mm/s |

Stribeck 行为：\|v\| → 0 时 μ → μ_s（静摩擦），\|v\| ≫ v_s 时 μ → μ_d（动摩擦）。

### 力分解与 ROM 映射（ForceAssembler 编排器）

ForceAssembler 负责力的几何分解（F_n, F_t → Fy, Fz）和 ROM DOF 映射——这些是几何簿记，不是物理模型。

机匣内法向 `n̂ = (0, −y_c/r, −z_c/r)`，反旋转切向 `t̂ = (0, +z_c/r, −y_c/r)`。物理接触力 `F_phys = F_n·n̂ + F_t·t̂`：
```
F_ROM_y = ( F_n·y_c − F_t·z_c) / r
F_ROM_z = ( F_n·z_c + F_t·y_c) / r
```
轴向力为 0（叶片叶尖不传轴向力）。

传给修正步的 ROM 力向量符号（使 `u_{n+1} = u_p − A⁻¹·F_total` 将叶片推向机匣内侧并减速）。

### 批量 JIT 快速路径

当 `contact_batch` 内核可用时（`jit_registry.get("force_batch", "PCL_FRICTION_WEAR")`），ForceAssembler 使用**批量快速路径**：单次 Numba 调用处理所有叶尖节点的 PCL + 摩擦 + 磨损 + 力分解 + ROM 映射，消除逐节点 Python↔JIT 调度开销。降级路径仍可用（per-contact Python 循环 + 独立 JIT 内核调用）。

### 其他力模块（可选）

| 槽位 | TYPE | 说明 |
|------|------|------|
| `*AERODYNAMIC_FORCE` | `STEADY_AERO` | 定常气动力（占位） |
| `*INERTIAL_FORCE` | `RIGID_ROTATION` | 离心/科氏惯性力（占位） |

可选力模块通过 `registry.get("force_module", TYPE)` 分派，接收 DataBus + cfg + const_params，返回 ForceModule 实例。

### DataBus 写入

`F_total`（OutputCatalog 变量 CF 的数据源）

---

## 12. `*DYNAMIC_RELAXATION`

在瞬态分析前对 t=0 时刻进行静力平衡预求解，消除初始干涉或突加外力引起的零时刻冲击载荷。

```
*DYNAMIC_RELAXATION
    max_steps=<步数>, tolerance=<容差>, relaxation=<松弛因子>, force_tol=<力容差>
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `max_steps` | int | 最大 DR 迭代步数 |
| `tolerance` | float | 位移变化收敛容差 ‖Δu‖∞ |
| `relaxation` | float | Jacobi 步长因子 β ∈ (0, 1]，越大越快但接触强时需减小 |
| `force_tol` | float | 残差力收敛容差 ‖R‖∞（平衡判据：R = −F_contact − K·u） |

> **全部必需**：四个参数均必须在 INP 中显式声明。

### 算法

Jacobi 预条件 Modified Newton 迭代，求解静力平衡 K·u = F_contact：

```
u = 0
while ‖Δu‖∞ > tolerance 或 ‖R‖∞ > force_tol:
    F_c  = fa.assemble(u)          ← 接触力（CD 符号约定：−F_contact_物理）
    R    = −F_c − K·u              ← 残差（平衡时 R=0）
    Δu   = β · diag(K)⁻¹ · R       ← Jacobi 预条件步
    u    = u + Δu
return u
```

收敛后 `u = u_eq` 作为瞬态分析的初始状态（速度从零开始）。DR 收敛后，从收敛位移施加**一次性**涂层磨损以初始化瞬态分析的涂层状态。

**涂层保护**：磨损是时变过程——不应在 DR 的静态迭代求解中累积。每次力评估前，涂层主状态 `ep`（及派生量 `h`、`alpha`）被恢复到初始快照；收敛后仅一次性写入磨损。若不保护，初始干涉达 ∼0.3 mm 时，∼7 次迭代即可磨穿 1.5 mm 涂层，导致瞬态分析中出现 ∼4 ms 周期位移跳跃。

**预条件器**：`diag(K)` 为 ROM 刚度矩阵对角元，每个自由度独立缩放——软模态大步快走，刚模态小步稳走。接触刚度增加时，有效 `K_ii` 变大，步长自动收缩，无需额外 safety 参数。

**收敛判据**：双重判据——位移停滞（‖Δu‖∞ < tolerance）**且**力平衡（‖R‖∞ < force_tol）。20 步 warm-up 后开始检查。

**调优**：β 控制步长比例，不是阻尼系数。β=0.5（默认）适合大多数情况；β=0.1–0.3 适合接触刚度远大于结构刚度时；β=0.7–0.9 适合接近线性的问题。若 DR 不收敛，减小 β 或增大 max_steps。收敛失败打印 `[WARN]` 并继续。

### 可选性

未声明 `*DYNAMIC_RELAXATION` 则从零位移直接启动瞬态分析。

---

## 13. `*STEP` / `*OUTPUT`

### *STEP

定义分析步的仿真控制参数和输出请求。

```
*STEP, NAME=<步名>
    Omega=<转速>, h=<时间步长>, T_f=<终止时间>

    *OUTPUT, TYPE=HISTORY, FREQUENCY=<频率>
        <变量1>,
        <变量2>,

*END STEP
```

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `NAME` | str | 否 | 步名称 |
| `Omega` | float | 是 | 转速，rad/s |
| `h` | float | 是 | 时间步长，s。要求 h < 2/ω_max |
| `T_f` | float | 是 | 终止时间，s。总步数 N = round(T_f/h) |

### *OUTPUT（全部参数必需）

| 参数 | 类型 | 说明 |
|------|------|------|
| `TYPE` | enum | `HISTORY`（时间序列）或 `FIELD`（场快照） |
| `FREQUENCY` | int | 每多少步输出一次 |
| `variables` | list | 逗号分隔的变量名列表（HISTORY: U/CF/PEN/ENERGY, FIELD: EP/H/S） |

> **至少一个 OUTPUT**：每个 STEP 必须声明至少一个 `*OUTPUT`，且 TYPE, FREQUENCY, variables 全部显式指定。

#### HISTORY 变量

| 变量 | 维度 | 说明 |
|------|------|------|
| `U` | (N_steps, n_tip_dof) | 叶尖节点 DOF 位移 |
| `PEN` | (N_steps, n_tip_nodes) | 每节点穿透深度，mm。无接触 = 0 |
| `CF` | (N_steps, n_r) | ROM 空间中的总接触力向量。符号：负值 = 推向机匣内侧 |
| `ENERGY` | (N_steps, 4) | [KE, SE, TE, 0] — 动能/势能/总能量/（预留） |

#### FIELD 变量

| 变量 | Shape | 说明 |
|------|-------|------|
| `H` | (n_θ, n_x) | 涂层剩余厚度快照 |
| `EP` | (n_θ, n_x) | 等效塑性应变快照 |
| `S` | (n_θ, n_x) | 接触应力场快照 |

> **OUTPUT 是必需的**：不再有默认行为。每个 STEP 必须显式声明至少一个 `*OUTPUT`。
>
> **变量名**：输出变量名由 OutputCatalog 预定义。HISTORY 变量使用简写（U/PEN/CF/ENERGY），FIELD 变量使用简写（EP/H/S），与 OutputCatalog 名称一致。

---

## 14. 仿真流程

按执行顺序的模块流水线（由 PipelineRegistry 拓扑排序生成，非硬编码）：

### INIT 层（离线，一次执行）

```
ExternalData → Casing → Coating → MatrixAssembly → ROM
```

### RUNTIME 层（每时间步循环）

```
① TimeIntegrator.predict()    u_p = A⁻¹ · b_n              (JIT: predictor)
② ContactDetector.detect()    pen, coords, interp            (JIT: kinematics → interpolator → casing_radius → gap_function)
③ ForceAssembler.assemble_batch()  F_total                   (JIT: force_batch 融合 PCL+friction+wear+ROM map)
   或 ForceAssembler.assemble()     F_total = Σ F_i            (降级：per-contact Python + JIT)
④ TimeIntegrator.correct()    u_{n+1} = u_p − A⁻¹ · F_total  (JIT: corrector)
⑤ StateManager.advance()      u_{n−1} ← u_n, u_n ← u_{n+1}
⑥ OutputDispatcher 输出（按频率）                            (OutputCatalog)
```

### 可选前置步

```
[DYNAMIC_RELAXATION]  →  t=0 静平衡预求解 → u_init → 瞬态启动
```

### JIT 内核分派概览

```
ContactDetector:   jit_registry("kinematics") → jit_registry("interpolator", TYPE) → jit_registry("casing_radius") → jit_registry("gap_function")
                    interpolator TYPE ∈ {BILINEAR, BICUBIC_BSPLINE}
ForceAssembler:    jit_registry("force_batch", "PCL_FRICTION_WEAR")  — 批量快速路径
                     融合 jit_registry("constitutive") + jit_registry("friction") + jit_registry("wear")
                   （降级：逐节点 jit_registry("constitutive") → jit_registry("friction") → jit_registry("wear")）
TimeIntegrator:    jit_registry("predictor") → ... → jit_registry("corrector")
```

---

## 15. 参数速查表

### 结构参数

| 符号 | INP 参数 | 典型值 | 来源 |
|------|----------|--------|------|
| n_free | — | ~103,902 | MTX 文件 |
| n_tip_nodes | — | 8 | 节点 CSV |
| n_b | — | 24 (= 8×3) | 自动 |
| n_modal | `n_modal` | 20 | `*ROM` |
| n_r | — | 44 | 自动 |

### 机匣与涂层（mm·MPa 制）

| 符号 | INP 参数 | 来源 | 典型值 |
|------|----------|------|--------|
| R₀ | `R0` | `*CASING` 顶层 | 270–272 mm |
| slope | `slope` | `*AXIAL_SHAPE` 子模块 | −0.056 |
| N_lobe | `N_lobe` | `*CIRCUMFERENTIAL_SHAPE` 子模块 | 3 |
| d₀ | `d0` | `*CIRCUMFERENTIAL_SHAPE` 子模块 | 0.5 |
| h_coat | `h_coat` | `*COATING` 顶层 | 1.0–1.5 mm |
| L | `L` | `*COATING` 顶层 | 30 mm |
| n_θ | `n_theta` | `*COATING` 顶层 | 400–1080 |
| n_x | `n_x` | `*COATING` 顶层 | 50–60 |

### 材料参数（全部必需声明）

| 符号 | INP 参数 | 典型值 | 单位 |
|------|----------|--------|------|
| E | `E` | 3500 | MPa |
| Y | `Y` | 10 | MPa |
| K_plas | `K_plas` | 1000 | MPa |
| μ | `mu` | 0.25 | — |
| μ_s | `mu_s` | 0.3 | — |
| μ_d | `mu_d` | 0.25 | — |
| v_s | `v_s` | 1.0 | mm/s |
| α | `alpha` | 2.5 | — |
| β | `beta` | 1e-6 | — |

### 仿真控制

| 符号 | INP 参数 | 典型值 |
|------|----------|--------|
| Ω | `Omega` | 1600 rad/s (≈ 15,279 rpm) |
| h | `h` | 1×10⁻⁷–1×10⁻⁸ s |
| T_f | `T_f` | 0.005–0.3 s |
| T_rev | — | 2π/Ω ≈ 3.93 ms |

### 单位制（mm · t · s · MPa · N）

| 物理量 | 单位 | 等效 SI |
|--------|------|---------|
| 长度 | mm | 0.001 m |
| 质量 | t (tonne) | 1000 kg |
| 时间 | s | 1 s |
| 应力/模量 | MPa | 10⁶ Pa (N/mm²) |
| 力 | N | 1 kg·m/s² |
| 能量 | mJ | 0.001 J |

一致性验证：F = m·a = t × mm/s² = N ✓；σ = E·ε = MPa × (mm/mm) = MPa ✓；F_n = A_cell · σ = mm² × MPa = N ✓

---

## 16. 文件格式规范

### MTX COO 格式

```
<row> <col> <value>
```

- 每行一个三元组：行索引 列索引 数值（空格分隔）
- **1-based 索引**（Matrix Market 标准）
- `%` 或 `#` 开头视为注释行
- 无头部行
- 编码：UTF-8（支持 BOM = UTF-8-SIG）
- 矩阵大小由最大行列索引推断，缺失零元素自动处理（稀疏存储）

### 节点坐标 CSV

```
<node_id>,<x>,<y>,<z>
```

- 逗号分隔，无头部行
- `#` 开头视为注释行
- 编码：UTF-8（支持 BOM）
- node_id 必须为 1-based，与 MTX 矩阵的节点编号一致

---

> **相关文档**：[用户指南](GETTING_STARTED.md) · [架构指南](ARCHITECTURE.md) · [代码更改原则](CODE_CHANGE_PRINCIPLES.md) · [模块添加指南](MODULE_ADDITION_GUIDE.md) · [参考文献](REFERENCES.md)
