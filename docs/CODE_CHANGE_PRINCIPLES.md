# 代码更改原则

> 修改本项目的强制性规则。违反任一条即为 **bug**。

---

## 1. 参数提取：物理参数 vs 结构性选择

### 物理参数 → 必须来自 INP，缺失报错

物理参数（材料属性、几何尺寸、工况参数）**绝不允许硬编码默认值**。用 `required_scalar()` 提取：

```python
from rubimpact.core.param_utils import required_scalar

# ✅ 正确：两级查找，缺失报错
self.E = required_scalar(self.db, cfg, "E", db_key="const_params",
                         source="*CONSTITUTIVE keyword in INP file")

# ❌ 错误：静默回退到 200e3，用户永远不知道 E 填错了
self.E = float(cfg.get("E", 200e3))
```

**物理参数清单**（非穷举）：
- 材料：E, Y, K_plas, k_penalty, mu, mu_s, mu_d, v_s
- 几何：R0, h_coat, L, n_theta, n_x, slope, d0, N_lobe
- 工况：Omega, h, T_f, n_modal
- DR：max_steps, tolerance, relaxation, force_tol
- 阻尼：alpha, beta

### 结构性选择 → 允许合理默认

模块**存在性**（`*COATING` 不声明、`*WEAR_LAW` 不指定）是结构性选择，不是物理参数。允许合理的零行为回退：

```python
# ✅ 正确：涂层是可选的，不存在 = 无涂层，间隙公式不含 h_coat
ch = self.db.get("coating.h")
if ch is None:
    ch = np.zeros((1, 1), dtype=np.float64)  # 无涂层时的合理零值

# ✅ 正确：wear_law 是可选的子模块，不指定 = 不启用磨损
wear_cfg = const_params.get("wear_law") or cfg.get("wear_law") or {"TYPE": "NONE"}

# ❌ 错误：把可选模块当必需，阻塞了无涂层的合法场景
ch = required_array(self.db, "coating.h", source="*COATING")
```

### 判断标准

| 问自己 | 物理参数 | 结构性选择 |
|--------|---------|-----------|
| 缺了这个值，计算结果会错吗？ | 是（E=200e3 vs E=3500 → 结果差几十倍） | 否（没涂层 = 间隙公式不含 h_coat 项，正确） |
| 用户需要知道这个值吗？ | 是，必须显式声明 | 否，默认行为就是"不用这个功能" |
| 举例 | E, Y, k_penalty, mu, R0, Omega | *COATING 不声明, wear_law 不指定, circum_shape 不指定 |

---

## 2. 禁止的模式

### ❌ 禁止：裸 `.get(key, magic_number)`

```python
# 任何带数字默认值的 .get() 都是潜在 bug
value = cfg.get("E", 200e3)          # 200e3 从哪来的？
value = cfg.get("max_steps", 10000)   # 10000 从哪来的？
value = cfg.get("tolerance", 1e-10)   # 1e-10 从哪来的？
```

### ❌ 禁止：回退链到硬编码

```python
# 三级回退 = 最后一级默默生效时用户不知道
self.E = const_params.get("E", cfg.get("E", 200e3))
#                                    ↑ 这个 200e3 是 bug
```

### ❌ 禁止：函数参数默认值伪装物理参数

```python
# 如果 A_cell 是物理参数，不能有默认值
def calculate_force(delta, A_cell=1.0):  # ← 1.0 从哪来的？
    ...
```

### ✅ 允许：纯计算常量

```python
# 这些是数学常量，不是物理参数
TWO_PI = 2.0 * np.pi
EPS = 1e-12
CONSTRAINED_DIAG = 1e30  # 约束 DOF 识别阈值
```

### ✅ 允许：结构性回退

```python
# 不声明 *COATING = 没有涂层数据 = zeros 是正确的
coating_h = db.get("coating.h")
if coating_h is None:
    coating_h = np.zeros((1, 1), dtype=np.float64)
```

---

## 3. 新增/修改代码的检查项

每次提交前检查：

- [ ] 是否有新的 `.get(key, number)` 模式？→ 改为 `required_scalar()` 或确认是有意的结构性回退
- [ ] 是否新增了 `if grid else 1.0` 这类回退？→ 确认 1.0 是纯计算常量还是隐含的物理默认值
- [ ] 是否新增了 `cfg.get("param", default)` 且 default 是数字？→ 审查动机
- [ ] 参数来源是否单一？（DataBus → cfg → error，不允许多级回退到硬编码）
- [ ] 如果 INP 不声明某个关键字，程序能正常跑吗？（不声明 ≠ 报错，除非该关键字是必需的）

---

## 4. 相关工具

| 工具 | 位置 | 用途 |
|------|------|------|
| `required_scalar()` | `rubimpact.core.param_utils` | 物理参数提取（两级查找） |
| `required_array()` | `rubimpact.core.param_utils` | DataBus 数组读取 |
| `DataBus.get()` | `rubimpact.infra.databus` | 可选键读取（允许 None） |
| `cfg.get()` | — | INP 配置读取（仅结构性可选参数） |

---

> **相关文档**：[架构指南](ARCHITECTURE.md) · [模块添加指南](MODULE_ADDITION_GUIDE.md) · [关键字手册](KEYWORD_MANUAL.md)
