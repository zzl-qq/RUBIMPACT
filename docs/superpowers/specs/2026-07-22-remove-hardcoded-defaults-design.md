# 消除硬编码默认值 — 设计文档

**日期**: 2026-07-22
**状态**: 已批准

## 1. 问题

项目中分布着约 15 处硬编码的物理/数值默认值（如 `E=200e3` MPa、`Y=500` MPa、`A_cell=1.0`），当 INP 输入文件缺少对应关键字时，代码静默使用这些默认值继续运行，导致计算结果可能完全错误且难以定位问题。

所有物理参数值必须从 INP 文件的关键字段中提取，缺失时应该明确报错而非静默回退。结构性可选模块（`*COATING` 不声明、`wear_law` 不指定）允许合理的零行为回退。

## 2. 设计原则

- **物理参数无静默回退**：任何物理参数找不到来源时，抛出清晰的 `ValueError`
- **结构性选择允许回退**：模块存在性（`*COATING` 不声明、`wear_law` 不指定）是结构性选择，不是物理参数，允许合理的零行为
- **两级查找**：上游模块写入 DataBus → 直接读取 INP cfg → 报错。保留 DataBus 作为模块间共享层，cfg 作为直接调用的备用路径
- **`required_scalar`/`required_array` 不提供 `default` 参数**——函数名叫 `required` 就不该有默认值

## 3. 新增工具模块

`src/rubimpact/core/param_utils.py`

## 4. 改动清单

见 `docs/CODE_CHANGE_PRINCIPLES.md` 中的完整规则。

## 5. 测试验证

- 157/157 测试全部通过
- `demo_penalty.inp`（无 `*COATING`、无 `*CONSTITUTIVE` 的 PENALTY 场景）完整运行成功
