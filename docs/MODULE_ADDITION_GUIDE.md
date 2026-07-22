# 模块添加指南

> 如何在本项目中添加各级别的新模块——从 JIT 内核到编排器

---

## 目录

1. [层级总览](#1-层级总览)
2. [L3：添加 JIT 内核](#2-l3添加-jit-内核)
3. [L2：添加协议化模块](#3-l2添加-协议化模块)
4. [L1：添加编排器](#4-l1添加-编排器)
5. [INIT 层：添加初始化模块](#5-init-层添加初始化模块)
6. [添加 TYPE 变体](#6-添加-type-变体)
7. [添加新 INP 关键字](#7-添加新-inp-关键字)

---

## 1. 层级总览

```
L0: core/         基础设施 — registry, pipeline_factory, module_base, param_utils
L1: orchestrators/ 编排器 — ModelAssembler, ForceAssembler
L2: modules/       协议化模块 — ContactForce, FrictionForce, ContactDetector, ...
L3: kernels/       JIT 内核 — constitutive, friction, wear, kinematics, ...
INIT: init/        初始化 — Casing, Coating, ExternalData, MatrixAssembly, ROM
```

**原则**：新增模块 = `kernels/` + `modules/` + `descriptors/` 各 1 个文件 + 注册。禁止修改 `registry.py`、`pipeline_factory.py`、`force_assembler.py`、`model_assembler.py`。

---

## 2. L3：添加 JIT 内核

JIT 内核是纯 `@njit` 函数——无 DataBus 依赖，无类实例，纯标量计算。

### 步骤

**Step 1：创建内核文件** `src/rubimpact/kernels/<name>.py`

```python
"""<Category> JIT kernels — <description>.

Registered types:
    <TYPE_NAME> — <one-line summary>

Signature: (arg1, arg2, ..., out)
"""
import numpy as np
from numba import njit
from rubimpact.core.registry import components, KernelSpec


@njit(cache=True, fastmath=True)
def my_kernel(arg1, arg2, out):
    """Per-node computation."""
    n = arg1.shape[0]
    for i in range(n):
        out[i] = arg1[i] + arg2[i]


# 注册：category 对应 submodule 槽位名，stage 对应管道阶段名
components.register("my_category", "MY_TYPE",
    KernelSpec(fn=my_kernel, signature="my_signature",
               stage="compute_something"))
```

**Step 2：在 `kernels/__init__.py` 中导入**（确保内核在运行时被注册）

```python
from rubimpact.kernels import my_kernel  # noqa: F401
```

**Step 3：（可选）创建 YAML 描述符** `src/rubimpact/descriptors/<path>/MY_TYPE.yaml`

```yaml
# <path> 由 category 名自动推断
# 描述符仅用于文档和验证，不影响功能
```

**Step 4：（可选）添加测试**

```python
def test_my_kernel():
    from rubimpact.kernels.my_kernel import my_kernel
    # ... test pure function behavior
```

### 关键契约

- 函数签名固定：`fn(inputs..., out)`，out 是最后一个参数
- 禁止在 @njit 循环内调用 `np.zeros()`/`np.empty()`（禁止堆分配）
- `signature` 字符串用于编译期检查
- `stage` 字符串对应 `PipelineStage.name`

---

## 3. L2：添加协议化模块

协议化模块继承 `Module`，通过 `get_pipeline_protocol()` 声明管道贡献。

### 步骤

**Step 1：创建模块文件** `src/rubimpact/modules/<name>.py`

```python
"""<Module name> — <description>."""
import numpy as np
from rubimpact.infra.databus import DataBus
from rubimpact.core.registry import components, ModuleSpec
from rubimpact.core.module_base import Module, PipelineStage, PipelineProtocol
from rubimpact.core.param_utils import required_scalar


class MyModule(Module):
    """One-line purpose."""

    def configure(self, cfg: dict) -> None:
        """Parse INP config, resolve kernels, extract params."""
        sm = cfg.get("submodules", {})

        # 物理参数 → required_scalar()
        self.my_param = required_scalar(
            self.db, cfg, "my_param",
            source="*MY_KEYWORD in INP file")

        # 可选子模块 → 允许默认
        sub_cfg = sm.get("my_submodule", {})
        self.sub_type = sub_cfg.get("TYPE", "DEFAULT")

        # 可选数据结构（如涂层数据）→ .get() + 回退
        self.my_data = self.db.get("some.data")

    def get_pipeline_protocol(self) -> PipelineProtocol:
        """Declare pipeline stages and frozen params."""
        return PipelineProtocol(
            stages=[
                PipelineStage(
                    name="compute_something",
                    kernel_ref="my_category/MY_TYPE",
                    depends_on=[],
                ),
                # 可选阶段（如磨损）：
                PipelineStage(
                    name="optional_step",
                    kernel_ref="other_category/OTHER_TYPE",
                    depends_on=["compute_something"],
                    optional=True,
                ),
            ],
            params={
                "my_param": self.my_param,
                "my_data": self.my_data,
            },
        )


# ── Builder ──

def _build_my_module(db, ctx):
    module = MyModule(db, ctx)
    module.configure(ctx.get("my_cfg", {}))
    return module


# ── Register ──

components.register("my_slot", "MY_TYPE",
    ModuleSpec(builder=_build_my_module, protocol="MyProtocol"))
```

**Step 2：在 `modules/__init__.py` 中导入**

```python
try:
    from rubimpact.modules import my_module
except ImportError:
    my_module = None
```

**Step 3：在编排器中添加子模块槽位**

在 `orchestrators/force_assembler.py`（或其他编排器）的 `configure()` 中添加新模块的构建逻辑：

```python
# 在 ForceAssembler.configure() 的 "其他力模块" 循环中：
# 新模块会自动被识别——如果 TYPE 注册在 ComponentRegistry 中
```

### 关键契约

- 所有物理参数使用 `required_scalar()`——两级查找，缺失报错
- 可选子模块（如 `wear_law`）允许 `cfg.get(key, default)` 回退
- 涂层/网格等可选数组使用 `db.get(key)` + 合理零值回退
- `get_pipeline_protocol()` 返回的 `params` 会在管道编译期冻结为闭包变量
- 新增 TYPE 自动进入 `ComponentRegistry.list_combinations()` 测试矩阵——**无需修改测试文件**

---

## 4. L1：添加编排器

编排器（Orchestrator）协调多个 L2 模块，负责构建期（`configure()`）解析子模块 TYPE 并组合管道。

### 现有的编排器

- `ForceAssembler` — 力组装（contact_force + friction_force）
- `ModelAssembler` — INIT 层调度（ExternalData → Casing → ... → ROM）

### 添加新编排器

**Step 1：创建编排器文件** `src/rubimpact/orchestrators/<name>.py`

```python
"""<Orchestrator name> — <description>."""
from rubimpact.infra.databus import DataBus
from rubimpact.core.module_base import Module
from rubimpact.core.registry import components
from rubimpact.core.pipeline_factory import PipelineFactory


class MyOrchestrator(Module):
    """Orchestrates <purpose>."""

    def configure(self, cfg: dict) -> None:
        sm = cfg.get("submodules", {})

        # 解析必需子模块
        sub_cfg = sm.get("my_submodule")
        if sub_cfg is None:
            raise ValueError("MY_ORCHESTRATOR requires 'my_submodule'")
        sub_type = sub_cfg.get("TYPE")
        if sub_type is None:
            raise ValueError(f"my_submodule requires TYPE. "
                             f"Registered: {components.list_types('my_submodule')}")

        spec = components.resolve_module("my_submodule", sub_type)
        if spec is None:
            raise ValueError(f"Unknown my_submodule TYPE: {sub_type}")
        self._module = spec.builder(self.db, {"cfg": sub_cfg})

        # 组装管道
        self._pipeline = PipelineFactory.build(
            modules={"slot": self._module},
            protocol="MyProtocol",
        )

    def run(self, *args):
        """Execute the orchestrated pipeline."""
        return self._pipeline(*args)


# Register
components.register_class("MY_ORCHESTRATOR", MyOrchestrator)
```

### 关键契约

- 编排器自身也是 `Module` 子类——有 `configure(cfg)` + `get_pipeline()`
- 子模块 TYPE 通过 `components.resolve_module()` 分派——**零 isinstance**
- 管道通过 `PipelineFactory.build()` 自动组合——**不手写管道函数**
- 所有 TYPE 差异在 `configure()` 时解决——`run()` 零分支、零 TYPE 判断

---

## 5. INIT 层：添加初始化模块

INIT 模块在模拟启动前执行一次，通过 `ModelAssembler` 调度。

### 步骤

**Step 1：创建初始化文件** `src/rubimpact/init/<name>.py`

```python
"""*<KEYWORD> module — <description>."""
from rubimpact.infra.databus import DataBus
from rubimpact.core.module_base import Module
from rubimpact.core.registry import components
from rubimpact.core.param_utils import required_scalar


class MyInitModule(Module):
    """One-line purpose."""

    def configure(self, cfg: dict) -> None:
        # 提取 INP 参数（全部必需）
        self.param1 = required_scalar(self.db, cfg, "param1",
                                      source="*MY_KEYWORD in INP file")
        self.param2 = float(cfg["param2"])

        # 写入 DataBus
        self.db.set("my_module.result", ...)


# Register
components.register_class("MY_KEYWORD", MyInitModule)
```

**Step 2：添加 YAML 描述符** `src/rubimpact/descriptors/<keyword>.yaml`

```yaml
keyword: MY_KEYWORD
category: MY_INIT_CATEGORY
lifecycle:
  required: false       # true = INP 中必须声明
  order_after: [CASING]  # 在 CASING 之后执行
submodules:
  my_subslot:
    category: my_category
    cardinality: 1
    candidates: [TYPE_A, TYPE_B]
params:
  param1:
    type: float
    required: true
    description: "My parameter"
```

### 关键契约

- 物理参数全部 `required_scalar()` 或直接 `cfg[key]`
- 执行顺序由 YAML 描述符的 `lifecycle.order_after` 声明——不硬编码
- 通过 DataBus 读写与其他 INIT 模块通信

---

## 6. 添加 TYPE 变体

为已有模块添加新的 TYPE 变体（如为摩擦模块添加新摩擦模型）。

### 无需修改的文件

- ❌ `registry.py`
- ❌ `pipeline_factory.py`
- ❌ `force_assembler.py`
- ❌ `model_assembler.py`

### 只需修改/创建

1. **新内核**：`kernels/<category>.py` 中添加 `@njit` 函数 + 注册
2. **（可选）新模块变体**：如需要不同的 `configure()` 行为，添加新 ModuleSpec builder
3. **描述符**：`descriptors/<path>/<TYPE>.yaml` 描述新 TYPE 的参数
4. **测试自动覆盖**：`list_combinations()` 自动将新 TYPE 纳入 L2 对比测试矩阵

### 示例：添加新的摩擦模型

```python
# kernels/friction.py
@njit(cache=True, fastmath=True)
def my_new_friction_kernel(F_n, v_rel, params, out):
    """My custom friction model."""
    out[0] = params[0] * F_n * (1.0 - np.exp(-abs(v_rel) / params[1]))

components.register("friction", "MY_NEW_MODEL",
    KernelSpec(fn=my_new_friction_kernel, signature="my_friction",
               stage="compute_friction_force"))
```

```python
# modules/friction_force.py 中添加 builder（如需要不同 cfg 解析）
def _build_friction_my_new(db, ctx):
    module = MyNewFrictionModule(db, ctx)
    module.configure(ctx.get("friction_cfg", {}))
    return module

components.register("friction_force", "MY_NEW_MODEL",
    ModuleSpec(builder=_build_friction_my_new, protocol="FrictionForceProtocol"))
```

就这样——新 TYPE 自动出现在 `test_all_force_assembler_combinations()` 中。

---

## 7. 添加新 INP 关键字

### 可选的 INIT 关键字

```yaml
# descriptors/<keyword>.yaml
keyword: MY_NEW_KEYWORD
category: MY_INIT_CATEGORY
lifecycle:
  required: false       # 不声明 = 跳过此模块
  order_after: [CASING]
```

**步骤**：
1. 创建 `init/<name>.py`，注册到 `components.register_class("MY_NEW_KEYWORD", MyClass)`
2. 创建 `descriptors/<name>.yaml`，声明 `required: false`
3. 在 `model_assembler.py` 的调度逻辑中——**如果使用 PipelineRegistry，无需修改**
4. 更新 `KEYWORD_MANUAL.md`

### 必需的 RUNTIME 子模块

```yaml
# 如果是已有编排器的子模块
keyword: MY_CONTACT_FORCE
category: contact_force
lifecycle:
  required: true       # 必须在 INP 中声明
```

**步骤**：
1. 创建 `modules/<name>.py`
2. 在对应的编排器中添加子模块解析逻辑（如果是新编排器）
3. 如果属于已有编排器——确认该编排器的子模块循环能发现新 TYPE
4. 更新 `KEYWORD_MANUAL.md`

---

> **相关文档**：[架构指南](ARCHITECTURE.md) · [代码更改原则](CODE_CHANGE_PRINCIPLES.md) · [关键字手册](KEYWORD_MANUAL.md)
