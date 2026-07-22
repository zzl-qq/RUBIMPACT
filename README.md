# RUBIMPACT v1.0.0

**RUBIMPACT** = **RUB**-**IMPACT** Generic Simulation Framework

旋转机械碰摩（rub-impact）接触−磨损耦合仿真的泛用计算平台。当前内置模型为 **RMBOPCL**（Reduced-order Model for Blade-Off Plastic Coating Law），模拟叶片脱落事件中叶片叶尖与可磨耗涂层机匣之间的碰摩−磨损过程。

---


## 设计原则

| 原则 | 说明 |
|------|------|
| **模块化** | 每个物理/数值功能独立成模块，通过 DataBus 通信 |
| **可替换** | 同类模块共享接口，换模型只需替换对应模块 |
| **声明式** | 用户通过 Abaqus 风格关键字组合模型，无需编写胶水代码 |
| **可校验** | 框架在组装时验证模块间兼容性，提前发现配置冲突 |
| **无幻数** | 所有物理参数必须来自 INP 输入文件，无硬编码默认值。结构性可选模块（不声明 = 不使用）允许合理零行为 |

---

## 快速开始

```bash
pixi install
```

### 运行仿真

```bash
# 默认单核（安全）
pixi run rubimpact submit input/demo_contact.inp

# 多核加速（线程数自动绑定物理核心，超出则截断）
pixi run rubimpact submit input/demo_contact.inp --cores 4
```

> **核数说明：** `--cores` 控制 MKL PARDISO 稀疏求解器的线程数。
> 上限自动检测**物理核心**（非逻辑处理器），因为超线程共享 FPU 和缓存，
> 对稠密线性代数通常无收益甚至有害。超出物理核心数时会自动截断并提示。

输出在 `<ModelName>/output/` 目录。

---

## 文档导航

| 文档 | 内容 | 适合 |
|------|------|------|
| **[关键字手册](docs/KEYWORD_MANUAL.md)** | 每个关键字的 INP 语法 + 数学公式 + 参数表 | 编写仿真输入 · 理解模型 |
| [用户指南](docs/GETTING_STARTED.md) | 安装、运行、输出解读、故障排除 | 首次使用 |
| [架构指南](docs/ARCHITECTURE.md) | 框架设计、执行流程、DataBus、模块注册表 | 二次开发 |
| [代码更改原则](docs/CODE_CHANGE_PRINCIPLES.md) | 参数提取规则、禁止模式、提交前检查项 | 修改代码 |
| [模块添加指南](docs/MODULE_ADDITION_GUIDE.md) | L1/L2/L3/INIT 各层模块添加流程 | 扩展新功能 |
| [参考文献](docs/REFERENCES.md) | 学术引用 | 论文写作 |

---

## 框架边界

**负责**：全阶矩阵导入 → ROM 降阶 → 离心刚化插值 → 涂层网格 → 每步接触检测/本构响应/力组装/磨损更新 → HDF5 流式输出

**不负责**：全阶 FE 模型生成、后处理与可视化、参数扫描/批处理（可通过脚本循环调用 CLI）

---

## 依赖

- Python 3.12+ (pixi)
- NumPy, SciPy, Numba (JIT)
- pypardiso (Intel MKL 稀疏求解器)
- h5py (HDF5 输出)

---

## 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| **1.0.1** | 2026-07-22 | 消除所有硬编码默认值——物理参数强制来自 INP，缺失即报错；新增 `param_utils` 参数提取工具；新增代码更改原则和模块添加指南 |
| **1.0.0** | 2026-07-18 | 初始发布：间隙公式修正、LOB 机匣、PEN 穿透量、模块注册表、动态松弛软启动、应变加固、磨损公式改用名义厚度、文档重组 |
