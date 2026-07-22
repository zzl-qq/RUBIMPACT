# RUBIMPACT 架构与扩展指南

> 框架设计、模块系统、执行流水线与验证机制

---

## 目录

0. [设计契约（十条规则）](#0-设计契约十条规则)
1. [架构总览](#1-架构总览)
2. [执行流程](#2-执行流程)
3. [统一注册表 + 管道工厂](#3-统一注册表--管道工厂)
4. [JIT 内核与协议化模块](#4-jit-内核与协议化模块)
5. [DataBus 接口](#5-databus-接口)
6. [约束验证](#6-约束验证)
7. [重构验证状态](#7-重构验证状态)

---

## 0. 设计契约（十条规则）

以下规则定义了框架的**硬边界**。违反任一条即为 **bug**。

| # | 规则 |
|---|------|
| **R1** | 一切可替换组件通过 `ComponentRegistry` 注册。`ModelAssembler` 和 `ForceAssembler` 中禁止硬编码分派（`isinstance` 或 `if TYPE ==`）。 |
| **R2** | 模块遵循统一接口 `configure(cfg)` + `get_pipeline()`。禁止自定义 `execute(cfg, ...)` 签名。 |
| **R3** | 管道由 `PipelineFactory` 根据协议声明自动生成。禁止手写管道函数。 |
| **R4** | 所有参数在构建期（`configure()`）闭包捕获。运行期（`assemble()`/`detect()`）零分支、零 TYPE 判断。 |
| **R5** | JIT 内核为纯函数。禁止内部分配堆内存（`np.zeros`/`np.empty` 在 @njit 循环内）。 |
| **R6** | 共享内核（几何分解、ROM DOF 映射）归 `kernels/shared.py`。管道自动引用，禁止重复实现。 |
| **R7** | DataBus 仅用于跨模块通信和输出阶段。热路径禁止 DataBus 写入。 |
| **R8** | **物理参数**（E, Y, K_plas, k_penalty, mu, R0, h_coat, Omega 等）无默认值，缺失即 `ValueError` 报错。使用 `required_scalar()` / `required_array()` 提取。**模块存在性**（`*COATING` 不声明、`wear_law` 不指定）是结构性选择，允许合理的零行为回退（无涂层 = 无磨损 = 间隙公式不含 h_coat）。 |
| **R9** | 新增模块 = `kernels/` + `modules/` + `descriptors/` 各 1 个文件 + 2 个 import。禁止修改 `registry.py`、`pipeline_factory.py`、`force_assembler.py`、`model_assembler.py`。 |
| **R10** | 新增 TYPE 自动进入 L2 对比测试矩阵（`ComponentRegistry.list_combinations()`）。 |
| **R11** | 参数提取统一使用 `param_utils` 工具函数。物理参数用 `required_scalar()`（两级查找：DataBus → cfg → ValueError）；可选的结构性数据（涂层数组、模块存在性标志）用 `DataBus.get()` + 合理零行为回退。禁止裸 `.get(key, magic_number)`。 |

---

## 1. 架构总览

### 设计原则

| 原则 | 说明 |
|------|------|
| **模块化** | 每个物理/数值功能独立成模块，通过 DataBus 通信 |
| **可替换** | 同类模块共享接口，换模型只需替换对应模块 |
| **声明式** | 用户通过 Abaqus 风格关键字组合模型，无需编写胶水代码 |
| **可校验** | 框架在组装时验证模块间兼容性，提前发现配置冲突 |

### 四层架构（协议驱动）

```
 ┌──────────────────────────────────────────────────────┐
 │ L0: 基础设施层 (core/)                                │
 │   ComponentRegistry · PipelineFactory · Module(ABC)   │
 │   Scheduler · DataBus                                │
 ├──────────────────────────────────────────────────────┤
 │ L1: 编排器层 (orchestrators/)                         │
 │   ModelAssembler — 统一 configure() 循环，零特殊分支   │
 │   ForceAssembler — 协议驱动管道，零 isinstance         │
 ├──────────────────────────────────────────────────────┤
 │ L2: 模块层 (modules/)                                 │
 │   ContactForce · FrictionForce · ContactDetector      │
 │   TimeIntegrator · DynamicRelaxation                  │
 │   每个模块声明 PipelineProtocol → 管道工厂自动组合     │
 ├──────────────────────────────────────────────────────┤
 │ L3: 内核层 (kernels/)                                 │
 │   kinematics · interpolator · gap · casing            │
 │   constitutive · friction · wear · shared             │
 │   predictor · corrector                               │
 │   纯 @njit 函数，通过 KernelSpec 注册                   │
 └──────────────────────────────────────────────────────┘
```

**目录结构**：
```
src/rubimpact/
├── core/           # 统一基础设施 (registry, pipeline_factory, module_base, scheduler, databus, param_utils)
├── kernels/        # L3 标量 JIT 内核 (10 个模块)
├── modules/        # L2 协议化模块 (5 个模块)
├── orchestrators/  # L1 编排器 (force_assembler, model_assembler)
├── init/           # INIT 层模块 (casing, coating, external_data, matrix_assembly, rom)
├── descriptors/    # 精简 YAML 描述符 (~15 文件)
└── infra/          # 基础服务 (databus, sparse_matrix, job_manager, state_manager, hdf5_writer)
```

### 框架边界

**负责**：全阶矩阵导入、ROM 降阶、离心刚化插值、涂层网格初始化、接触检测、本构响应、力组装与磨损更新、时间推进、流式 HDF5 输出

**不负责**：FE 模型生成、后处理与可视化、参数扫描/批处理（通过脚本循环调用 CLI 实现）

---

## 2. 执行流程

### INIT 层数据流（流水线调度）

ModelAssembler 不硬编码 INIT 模块执行顺序。`PipelineRegistry.schedule()` 根据 YAML 描述符的 `lifecycle.order_after` 声明执行拓扑排序（Kahn 算法），自动生成有序执行列表。

```
ExternalData          Casing                Coating
     │                   │                     │
     ├─ matrices.mass ───┤                     │
     ├─ matrices.stiff ──┤                     │
     └─ nodes.tip ───────┤                     │
                         ├─ casing.geometry ───┤
                         │                     ├─ coating.grid
                         │                     ├─ coating.h
                         │                     ├─ coating.ep
                         │                     └─ coating.alpha

MatrixAssembly ◄────────┘
     │
     ├─ matrices.K_omega ──┐
     └─ matrices.D_full ───┤
                           │
ROM ◄──────────────────────┘
     │
     ├─ rom.M_r, rom.K_r, rom.D_r
     ├─ rom.n_r, rom.tip_dof_map
     └─ rom.Phi_CB, rom.free_dofs
```

### RUNTIME 层单步数据流

> **可选前置步：动态松弛 (DR)** — 若 INP 声明了 `*DYNAMIC_RELAXATION`，则在瞬态循环前执行 Jacobi 预条件 Modified Newton 迭代。DR 接收**管道函数**（非模块实例），涂层保护通过构建期模式标志实现。

```
StateManager (u_n, u_nm1)
     │
     ▼
① TimeIntegrator.predict()    ← M_r, K_r, D_r (ROM)
     │ u_p                        JIT: predictor kernel
     ▼
② ContactDetector.get_pipeline()(u_p, t, Omega)   ← coating.h/ep/alpha
     │ pen[], coords[], interp[]                     Protocol: 4-stage pipeline
     ▼
③ ForceAssembler.get_pipeline()(pen, coords, interp, Omega)
     │ F_total                      Protocol: PipelineFactory 自动组合 contact+friction+geometry+ROM
     ▼                               零 isinstance，零 TYPE 分派
④ TimeIntegrator.correct()
     │ u_{n+1}                    JIT: corrector kernel
     ▼
⑤ StateManager.advance()
     │
⑥ OutputDispatcher (按输出频率)
```

### 接触检测内部流程

每个叶尖节点的检测链通过四个 JIT 子函数完成，各子函数类型由 `jit_registry` 动态分派：

1. **旋转运动学**（JIT）：θ = Ω·t + θ₀, x = x₀ + u_x, r = √(y_c² + z_c²)  
   `jit_registry.get("kinematics", TYPE)`
2. **涂层双线性插值**（JIT，如有涂层）：从 `coating.h/ep/alpha` 网格插值得到 h_loc, ep_loc, alpha_loc  
   `jit_registry.get("interpolator", TYPE)`
3. **机匣半径**（JIT 快速路径）：对预计算的 `casing.R_grid` 做双线性插值，替代 Python 层逐节点 `get_radius()` 调用；回退路径使用 `casing.geometry.get_radius(x, θ)`  
   `jit_registry.get("casing_radius", TYPE)`
4. **间隙计算**（JIT）：g = R − h_loc − r；g < 0 → 穿透 δ = −g  
   `jit_registry.get("gap_function", TYPE)`

### JIT 内核调用链

```
ContactDetector.detect()
  │
  ├─ jit_registry.get("kinematics", TYPE)        → 运动学 (RIGID_ROTATION_PLUS_VIBRATION)
  ├─ jit_registry.get("interpolator", TYPE)      → 插值 (BILINEAR / BICUBIC_BSPLINE)
  ├─ jit_registry.get("casing_radius", TYPE)     → 机匣半径 (RGRID_BILINEAR)
  ├─ jit_registry.get("gap_function", TYPE)      → 间隙 (DEFAULT)
  │
  └─ ForceAssembler.assemble_batch(): 单个 Numba kernel 融合所有接触力计算
       └─ jit_registry.get("force_batch", TYPE)  → 批量接触力 (PCL_FRICTION_WEAR)
            融合：constitutive + friction + wear + force decomposition + ROM mapping
       └─ （降级路径：per-contact Python 循环）
            ├─ jit_registry.get("constitutive", TYPE) → 本构 (PLASTIC_COATING_LAW)
            ├─ jit_registry.get("friction", TYPE)     → 摩擦 (COULOMB / STRIBECK)
            └─ jit_registry.get("wear", TYPE)         → 磨损 (PLASTIC_STRAIN)

TimeIntegrator.predict()  → jit_registry.get("predictor", TYPE)   → 预测 (LINEAR)
TimeIntegrator.correct()  → jit_registry.get("corrector", TYPE)   → 修正 (CONTACT_CONSTRAINED)
```

### 双三次 B 样条插值器 (BICUBIC_BSPLINE)

`interpolator, TYPE=BICUBIC_BSPLINE` 使用双三次均匀 B 样条曲面进行涂层状态插值。

**数据流**:
1. `bicubic_bspline_interp_kernel` → `interp_buf (n_nodes, 21)`: [h, ep, α, i_θ, i_x, w00..w33]
2. `contacts_as_list()` → contact dict 含全部 16 个权重字段
3. `ContactForce.compute()` → 磨损核 `PLASTIC_STRAIN_BICUBIC` (16 格点分布)
4. `ForceAssembler._distribute_stress_bicubic()` → 16 格点应力累加

**配置唯一来源**: `interpolator.TYPE` 决定所有下游操作的模板大小。
`wear_distributor` 参数无需单独设置。

---

## 3. 统一注册表 + 管道工厂

### 3.1 ComponentRegistry — 统一注册表

`ComponentRegistry` (`src/rubimpact/core/registry.py`) 替代了之前的 `ModuleRegistry` 和 `JITRegistry`，所有组件共用单一 `(category, type_name)` 命名空间。

#### 核心 API

```python
from rubimpact.core.registry import components, KernelSpec, ModuleSpec

# 注册 JIT 内核
components.register("constitutive", "PLASTIC_COATING_LAW",
    KernelSpec(fn=pcl_kernel, signature="pcl_return_mapping", stage="compute_normal_force"))

# 注册可实例化模块
components.register("contact_force", "PCL_CONTACT",
    ModuleSpec(builder=build_pcl, protocol="NormalForceProtocol"))

# 查找
components.resolve("constitutive", "PLASTIC_COATING_LAW")      → KernelSpec | ModuleSpec
components.resolve_kernel("constitutive", "PLASTIC_COATING_LAW") → KernelSpec | None
components.resolve_module("contact_force", "PCL_CONTACT")        → ModuleSpec | None
components.list_types("contact_force")                            → list[str]

# 模块类注册（向后兼容旧 init 模块）
components.register_class("CASING", Casing)
components.register_typed_class("TIME_INTEGRATOR", "CD", TimeIntegrator)
```

### 3.2 PipelineFactory — 协议驱动管道编译器

`PipelineFactory` (`src/rubimpact/core/pipeline_factory.py`) 读取模块的 `PipelineProtocol` 声明，**自动生成**组合后的 @njit 管道函数。

```python
PipelineFactory.build(
    modules={"contact": contact_module, "friction": friction_module},
    protocol="ForceAssembler",
    mode=None,  # "readonly_coating" 禁用 wear 阶段
)
# → @njit pipeline(pen, coords, interp, Omega, dof_idx, F_total, F_normal, F_friction)
```

**工作流**：
1. 收集所有模块的 `PipelineStage` 声明
2. 按 `depends_on` 拓扑排序
3. 从 ComponentRegistry 解析内核函数
4. 冻结模块参数为闭包变量
5. 生成单次 `for i in range(n_nodes)` 循环 → @njit 编译
6. 可选阶段（`optional=True`）按构建期标志做死分支消除

### 3.3 PipelineProtocol — 模块协议声明

每个 L2 模块通过 `get_pipeline_protocol()` 声明它贡献的管道阶段：

```python
class ContactForceModule(Module):
    def get_pipeline_protocol(self):
        return PipelineProtocol(
            stages=[
                PipelineStage(name="compute_normal_force",
                    kernel_ref="constitutive/PLASTIC_COATING_LAW", depends_on=[]),
                PipelineStage(name="apply_wear",
                    kernel_ref="wear/PLASTIC_STRAIN", depends_on=["compute_normal_force"],
                    optional=True),
            ],
            params={"pcl_params": self._pcl_params, ...},
        )
```

### 3.4 Scheduler — 拓扑调度

`PipelineRegistry`（`src/rubimpact/core/scheduler.py`，别名 `scheduler`）管理 INIT/RUNTIME 模块执行顺序，使用 Kahn 算法拓扑排序。

### 3.5 param_utils — 参数提取工具

`src/rubimpact/core/param_utils.py` 提供两个函数，统一项目中所有参数提取模式：

```python
from rubimpact.core.param_utils import required_scalar, required_array

# 物理参数：两级查找 → DataBus（const_params）→ cfg → ValueError
E = required_scalar(self.db, cfg, "E", db_key="const_params",
                    source="*CONSTITUTIVE keyword in INP file")

# 数组数据：直接从 DataBus 读取
ch = required_array(self.db, "coating.h", source="*COATING module")
```

**`required_scalar(db, cfg, key, *, db_key=None, source="") -> float`**

两级查找：1) `db.get(db_key, {}).get(key)` → 2) `cfg.get(key)` → 3) `ValueError`。

**`required_array(db, key, *, source="") -> Any`**

直接从 DataBus 读取。无 cfg 回退——数组（涂层网格等）只通过 DataBus 传递。

**何时用 `required_*` vs 普通 `get()`**：

| 场景 | 工具 | 原因 |
|------|------|------|
| E, Y, K_plas, k_penalty, mu | `required_scalar()` | 物理参数，必须来自 INP |
| R0, h_coat, L, Omega | `required_scalar()` | 几何/工况参数，必须来自 INP |
| coating.h, coating.ep | `db.get()` + zeros 回退 | 涂层是可选模块，不存在时合法 |
| wear_law 配置 | `cfg.get()` + `{"TYPE": "NONE"}` 回退 | 不指定 = 不启用磨损 |
| max_steps, tolerance 等 | `cfg[key]` 直接访问 | INP 中必须声明 |

---

## 4. JIT 内核与协议化模块

### 文件结构

```
src/rubimpact/kernels/            ← L3 标量 JIT 内核
├── __init__.py                   # 按依赖顺序导入，触发注册
├── kinematics.py                 # 旋转运动学
├── interpolator.py               # 涂层插值 (BILINEAR/BICUBIC_BSPLINE)
├── gap.py                        # 间隙计算
├── casing.py                     # 机匣半径插值
├── constitutive.py               # 一维返回映射
├── friction.py                   # 摩擦力 (COULOMB/STRIBECK)
├── wear.py                       # 磨损分配
├── predictor.py                  # 中心差分预测
├── corrector.py                  # 中心差分修正
└── shared.py                     # 共享内核：几何分解 + ROM DOF 映射

src/rubimpact/modules/            ← L2 协议化模块
├── contact_force.py              # PCL_CONTACT + PENALTY
├── friction_force.py             # COULOMB + STRIBECK
├── contact_detector.py           # 4 阶段检测管道
├── time_integrator.py            # predict + correct 管道
└── dynamic_relaxation.py         # 管道函数注入
```

### 内核注册

每个 @njit 内核通过 `KernelSpec` 注册：

```python
from rubimpact.core.registry import components, KernelSpec

components.register("constitutive", "PLASTIC_COATING_LAW",
    KernelSpec(fn=pcl_kernel, signature="pcl_return_mapping",
               stage="compute_normal_force"))
```

### 模块协议声明

每个模块通过 `get_pipeline_protocol()` 声明它在管道中贡献的阶段：

```python
class ContactForceModule(Module):
    def get_pipeline_protocol(self):
        return PipelineProtocol(
            stages=[
                PipelineStage(name="compute_normal_force",
                    kernel_ref="constitutive/PLASTIC_COATING_LAW", depends_on=[]),
                PipelineStage(name="apply_wear",
                    kernel_ref="wear/PLASTIC_STRAIN",
                    depends_on=["compute_normal_force"], optional=True),
            ],
            params={"pcl_params": self.pcl_params, ...},
        )
```

### 管道自动组合

`PipelineFactory.build()` 读取所有模块协议 → 拓扑排序 → 解析内核 → 闭包捕获参数 → @njit 编译。几何分解和 ROM DOF 映射作为共享阶段自动插入，不归任何模块所有。

**新增 TYPE 无需修改任何管道工厂代码** — 只需注册新的 KernelSpec 和 ModuleSpec。

---

## 5. DataBus 接口

`DataBus` 是模块间通信的唯一通道——全局键值存储，解耦模块间直接依赖。

### 操作

```python
db.set(key: str, value: Any)           # 写入
db.get(key: str, default=None) -> Any  # 读取
db.has(key: str) -> bool               # 存在性检查
```

### INIT 层键

| 键 | 写入者 | 读取者 | 类型 |
|----|--------|--------|------|
| `matrices.mass` | ExternalData | MatrixAssembly, ROM | SparseMatrix |
| `matrices.stiffness` | ExternalData | MatrixAssembly | dict[float, SparseMatrix] |
| `nodes.tip` | ExternalData | ROM, ContactDetector | dict[int, tuple] |
| `casing.geometry` | Casing | Coating, ContactDetector | dict (含 get_radius) |
| `casing.R_grid` | Casing.build_R_grid | ContactDetector (JIT casing_radius) | ndarray (n_θ, n_x) |
| `coating.grid` | Coating | ForceAssembler, ContactDetector | dict |
| `coating.h` | Coating, ForceAssembler | ContactDetector | ndarray (n_θ, n_x) — 派生量，`h_coat·(1−ep)` |
| `coating.ep` | Coating, ForceAssembler | ContactDetector | ndarray (n_θ, n_x) — **主状态**，累积塑性应变 |
| `coating.alpha` | Coating, ForceAssembler | ContactDetector | ndarray (n_θ, n_x) — 派生量，`= ep` |
| `matrices.K_omega` | MatrixAssembly | ROM | SparseMatrix |
| `matrices.D_full` | MatrixAssembly | ROM | SparseMatrix |

### ROM 层键

| 键 | 写入者 | 读取者 | 类型 |
|----|--------|--------|------|
| `rom.M_r` | ROM | TimeIntegrator, dynamic_relaxation | ndarray (n_r, n_r) |
| `rom.K_r` | ROM | TimeIntegrator, dynamic_relaxation | ndarray (n_r, n_r) |
| `rom.D_r` | ROM | TimeIntegrator | ndarray (n_r, n_r) |
| `rom.Phi_CB` | ROM | (后处理) | ndarray |
| `rom.n_r` | ROM | ModelAssembler | int |
| `rom.enabled` | ROM | ContactDetector | bool |
| `rom.tip_dof_map` | ROM | ContactDetector | list[int] |

### RUNTIME 层键

| 键 | 写入者 | 读取者 | 类型 |
|----|--------|--------|------|
| `F_total` | ForceAssembler | TimeIntegrator.correct | ndarray (n_r,) |
| `penetration` | ContactDetector | OutputDispatcher | ndarray (n_nodes,) |
| `contacts_coords` | ContactDetector | (内部) | ndarray (n_nodes, 5) |
| `contacts_interp` | ContactDetector | (内部) | ndarray (n_nodes, 9) |
| `current_h` | ModelAssembler | OutputDispatcher | float |

---

## 6. 约束验证

### 四层验证（全部阻断）

所有验证错误都会**阻断**执行（不再有 advisory-only 的检查）。

1. **YAML 描述符验证** — 关键字存在性 + TYPE 校验 + 子模块完整性 + 跨模块约束（`validate_config()`）
2. **Port 类型校验** — 端口提供者-消费者类型兼容性检查（`validate_ports()`）
3. **Pipeline 调度校验** — 必须模块缺失检测 + 循环依赖检测
4. **运行时隐式检查** — ROM 矩阵正定性、DR 收敛性等

### validate_config() 规则

**R1** — 必需关键字（`lifecycle.required=true` 的所有模块必须在 INP 中声明）

**R2** — `requires_keywords`：声明的依赖关键词必须存在（如 CONSTITUTIVE 要求 COATING）

**R3** — `conflicts_with`：互斥模块 ID 不得同时出现在 INP 中

**R4** — `accepts_children`：子模块 TYPE 必须在 `candidates` 列表中；`cardinality > 0` 的子模块必须显式声明

**零默认值**：所有物理参数、子模块 TYPE 均无默认值。INP 中未声明的参数 = 不存在 → 报错。仅 JIT 内核内联的纯数学常量保留。

### 新 INP 语法规则

子模块使用 `*SUBKEYWORD` + 缩进声明。顶层关键字 0 空格缩进，一级子模块 4 空格，二级子模块 8 空格：

```
*TOP_KEYWORD, TYPE=<TYPE>
    *SUBKEYWORD1, TYPE=<TYPE>, param=value
    *SUBKEYWORD2, TYPE=<TYPE>
        *NESTED_SUB, TYPE=<TYPE>, param=value
    param=value
```

`KeywordParser` 根据 `*` 前缀和缩进层级自动识别嵌套结构。

### YAML 描述符结构

每个注册的 TYPE 在 `modules/<category>/<path>/<TYPE>.yaml` 中有一个描述符。路径中 `/` 表示子模块层级，由 `_infer_category_from_path()` 自动推断 category。

当前总计 **34 个描述符** 覆盖所有注册 TYPE。

---

## 7. 重构验证状态

> **日期**: 2026-07-22 · **分支**: `main` · **状态**: 协议驱动重构完成

### 重构内容

| 变更 | 状态 | 说明 |
|------|------|------|
| 统一注册表 (ComponentRegistry) | ✅ | 替代 ModuleRegistry + JITRegistry，单一命名空间 |
| 协议驱动管道 (PipelineFactory) | ✅ | 自动生成 @njit 组合函数，零手写管道 |
| ForceAssembler 零 isinstance | ✅ | configure() 构建管道，assemble() 单次调用 |
| ModelAssembler 零特殊分支 | ✅ | 统一 configure() 循环，所有模块共用 context dict |
| 共享内核 (geometry + ROM mapping) | ✅ | 消除 ~60 行重复代码 |
| 管道模式推广 | ✅ | ContactDetector, TimeIntegrator, DR 均协议化 |
| 目录重组 | ✅ | core/ kernels/ modules/ orchestrators/ 四层结构 |
| 旧代码清理 | ✅ | 删除 jit_registry, module_registry, force_modules, contact_batch |
| 后向兼容 | ✅ | ComponentRegistry 支持 .get(), register_typed_class() |

### 测试状态

| 组件 | 通过 | 说明 |
|------|------|------|
| 核心基础设施 | 143/143 | registry, pipeline_factory, module_base, scheduler, 所有内核 |
| 旧系统测试 | 待更新 | test_casing, test_bicubic_bspline 需 YAML 精简后更新 import | |

---

> **相关文档**：[关键字手册](KEYWORD_MANUAL.md) · [用户指南](GETTING_STARTED.md) · [代码更改原则](CODE_CHANGE_PRINCIPLES.md) · [模块添加指南](MODULE_ADDITION_GUIDE.md) · [参考文献](REFERENCES.md)
