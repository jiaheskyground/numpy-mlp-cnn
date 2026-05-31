# 自研深度学习框架 — 架构设计文档

## 一、整体架构

```
train.py  ← 训练脚本（调用所有模块）
    │
    ├── dlframe/data/dataloader.py   ← DataLoader（mini-batch 迭代）
    ├── dlframe/nn/                  ← 网络层模块
    │   ├── module.py                ← Module 基类
    │   ├── linear.py                ← Linear 全连接层
    │   ├── activation.py            ← 激活函数层（ReLU, Sigmoid, Tanh, Softmax）
    │   ├── init.py                  ← 参数初始化（Xavier, He）
    │   ├── container.py             ← Sequential 容器
    │   └── loss.py                  ← 损失函数（MSELoss, CrossEntropyLoss）
    ├── dlframe/optim/               ← 优化器模块
    │   ├── sgd.py                   ← SGD / SGD+Momentum
    │   └── adam.py                  ← Adam
    └── dlframe/                     ← 核心
        ├── tensor.py                ← Tensor 类 + 反向传播图遍历
        ├── autograd.py              ← Function 基类 + 各类运算子类
        └── parameter.py             ← Parameter（可训练参数，继承 Tensor）
```

依赖关系：`tensor.py` → `autograd.py` → `parameter.py` → `nn/module.py` → `nn/*` → `optim/*`

---

## 二、逐模块类与变量说明

### 2.1 自动微分引擎（dlframe/）

#### 2.1.1 Tensor 类 (`tensor.py`)

核心数据结构，包装 NumPy 数组并追踪计算图。

| 变量/属性 | 类型 | 含义 |
|-----------|------|------|
| `data` | `np.ndarray` | 存储该张量的实际数值，所有前向计算基于此 |
| `grad` | `np.ndarray \| None` | 存储该张量的梯度（损失对该张量的偏导数），形状与 `data` 一致。叶节点初始为 None，backward 后累积填充 |
| `requires_grad` | `bool` | 是否需要计算梯度。用户创建的输入/参数设为 True，中间结果由操作自动推断 |
| `_grad_fn` | `callable \| None` | 指向创建该张量的 `Function.backward` 静态方法。叶节点为 None（因为它不是由操作创建的） |
| `_ctx` | `Context \| None` | 前向传播时保存的上下文（中间值），供反向传播使用。由 `Function.forward` 填充 |
| `_parents` | `tuple[Tensor, ...]` | 计算图中该节点的直接前驱（输入张量），用于反向拓扑排序遍历 |
| `shape` | `tuple[int, ...]` | 便捷属性，返回 `self.data.shape` |
| `dtype` | `np.dtype` | 便捷属性，返回 `self.data.dtype` |

**关键方法：**

| 方法 | 签名 | 功能 |
|------|------|------|
| `backward()` | `(self, grad=None) -> None` | 从当前张量出发执行反向传播：拓扑排序 → 逆向调用各节点的 `_grad_fn` → 累积梯度至叶节点 |
| `__add__` | `(self, other) -> Tensor` | `+` 运算符重载，调用 `Add.apply(self, other)` |
| `__radd__` | `(self, other) -> Tensor` | 反向加法（如 `3 + t`） |
| `__matmul__` | `(self, other) -> Tensor` | `@` 矩阵乘法，调用 `MatMul.apply(self, other)` |
| `__mul__` | `(self, other) -> Tensor` | `*` 逐元素乘法，调用 `Mul.apply(self, other)` |
| `__neg__` | `(self) -> Tensor` | 取负，调用 `Neg.apply(self)` |
| `__sub__` | `(self, other) -> Tensor` | `-` 减法，调用 `Sub.apply(self, other)` |
| `__truediv__` | `(self, other) -> Tensor` | `/` 除法 |
| `relu()` | `(self) -> Tensor` | ReLU 激活，调用 `ReLUFunc.apply(self)` |
| `sigmoid()` | `(self) -> Tensor` | Sigmoid 激活 |
| `tanh()` | `(self) -> Tensor` | Tanh 激活 |
| `sum()` | `(self, axis, keepdims) -> Tensor` | 求和归约 |
| `mean()` | `(self, axis, keepdims) -> Tensor` | 均值归约 |
| `reshape()` | `(self, *shape) -> Tensor` | 形状变换 |
| `transpose()` | `(self, axes) -> Tensor` | 转置操作 |
| `__repr__` | `(self) -> str` | 字符串表示，显示 data/ grad/ requires_grad |

#### 2.1.2 Context 类 (`tensor.py`)

前向传播时保存中间值的容器，使反向传播无需重新计算。

| 变量/属性 | 类型 | 含义 |
|-----------|------|------|
| `saved_tensors` | `tuple` | 在前向传播中保存的任意数值/数组/张量，按保存顺序排列 |

| 方法 | 签名 | 功能 |
|------|------|------|
| `save_for_backward(*tensors)` | 保存任意数量的值，供 backward 使用 |
| `get_saved()` | `-> tuple` | 返回所有已保存的值 |

#### 2.1.3 Function 基类 (`autograd.py`)

所有可微分操作的抽象基类。每个子类代表一种运算。

| 方法 | 签名 | 功能 |
|------|------|------|
| `forward(ctx, *inputs)` | `(Context, *np.ndarray) -> np.ndarray` | **静态方法**。接收 NumPy 数组，计算前向结果。需将反向传播依赖的值通过 `ctx.save_for_backward()` 保存 |
| `backward(ctx, grad_output)` | `(Context, np.ndarray) -> tuple[np.ndarray, ...]` | **静态方法**。接收输出梯度，利用 ctx 中保存的值计算各输入的梯度，返回与 `forward` 输入数量相同的梯度元组 |
| `apply(*inputs)` | `(*Tensor\|np.ndarray\|scalar) -> Tensor` | **类方法**。将输入（自动转换非 Tensor）送入 forward，包装结果为 Tensor，记录计算图信息（`_ctx`, `_grad_fn`, `_parents`） |

**Function 子类一览：**

| 子类 | forward 公式 | backward 公式 |
|------|-------------|---------------|
| `Add` | `y = a + b` | `grad_a = grad_y`（按广播规则求和），`grad_b = grad_y`（同上） |
| `MatMul` | `y = a @ b` | `grad_a = grad_y @ b.T`，`grad_b = a.T @ grad_y` |
| `ReLUFunc` | `y = max(0, x)` | `grad_x = grad_y * (x > 0)` |
| `Mul` | `y = a * b` | `grad_a = grad_y * b`，`grad_b = grad_y * a` |
| `Neg` | `y = -x` | `grad_x = -grad_y` |
| `Sub` | `y = a - b` | `grad_a = grad_y`，`grad_b = -grad_y` |
| `Sum` | `y = sum(x, axis)` | `grad_x = broadcast(grad_y, x.shape)` |
| `Reshape` | `y = reshape(x, shape)` | `grad_x = reshape(grad_y, x.shape)` |
| `Transpose` | `y = x.T` | `grad_x = grad_y.T` |
| `SigmoidFunc` | `y = 1/(1+e^{-x})` | `grad_x = grad_y * y * (1-y)` |
| `TanhFunc` | `y = tanh(x)` | `grad_x = grad_y * (1 - y^2)` |
| `LogSoftmax` | `y = log(softmax(x))` | 内部函数，供 CrossEntropyLoss 使用 |

#### 2.1.4 Parameter 类 (`parameter.py`)

继承自 Tensor，用于标记可训练参数。

| 变量/属性 | 类型 | 含义 |
|-----------|------|------|
| （继承 Tensor 全部） | | `requires_grad` 强制为 `True` |

与普通 Tensor 的区别：`Module.parameters()` 通过 `isinstance(p, Parameter)` 筛选可训练参数。

---

### 2.2 神经网络模块（dlframe/nn/）

#### 2.2.1 Module 基类 (`nn/module.py`)

所有网络层的抽象基类，模仿 PyTorch 的 `nn.Module`。

| 变量/属性 | 类型 | 含义 |
|-----------|------|------|
| `_modules` | `OrderedDict[str, Module]` | 子模块字典（如 Sequential 中的各层） |
| `_parameters` | `OrderedDict[str, Parameter]` | 直接属于该模块的可训练参数（如 Linear 的 weight, bias） |
| `_training` | `bool` | 当前模式：True=训练，False=评估（影响 Dropout/BN 行为） |

| 方法 | 签名 | 功能 |
|------|------|------|
| `parameters()` | `-> list[Parameter]` | 递归收集当前模块及所有子模块的 Parameter，返回扁平列表 |
| `zero_grad()` | `-> None` | 将所有参数的 `.grad` 置为 None |
| `train()` | `-> self` | 设置为训练模式（`_training=True`），递归应用到子模块 |
| `eval()` | `-> self` | 设置为评估模式（`_training=False`），递归应用到子模块 |
| `forward(*inputs)` | `-> Tensor` | **子类必须实现**，定义前向计算逻辑 |
| `__call__(*inputs)` | `-> Tensor` | 调用 `forward()`，使实例可像函数一样调用 |
| `__setattr__` 钩子 | | 自动将 Module 实例加入 `_modules`，将 Parameter 实例加入 `_parameters` |
| `_flatten_modules()` | `-> list[Module]` | 按 DFS 顺序展平所有子模块，用于参数收集 |

#### 2.2.2 Linear 全连接层 (`nn/linear.py`)

| 变量/属性 | 类型 | 含义 |
|-----------|------|------|
| `weight` | `Parameter` | 权重矩阵，形状 `(out_features, in_features)` |
| `bias` | `Parameter \| None` | 偏置向量，形状 `(out_features,)`；若 `bias=False` 则为 None |
| `in_features` | `int` | 输入特征维度 |
| `out_features` | `int` | 输出特征维度 |

**forward**: `y = x @ W.T + b`

#### 2.2.3 激活函数层 (`nn/activation.py`)

| 类 | 参数 | forward 公式 | 备注 |
|----|------|-------------|------|
| `ReLU` | 无 | `max(0, x)` | 隐藏层默认激活 |
| `Sigmoid` | 无 | `1/(1+e^{-x})` | 用于二分类输出 |
| `Tanh` | 无 | `(e^x-e^{-x})/(e^x+e^{-x})` | 零均值激活 |
| `Softmax` | `dim: int` | `e^{x_i} / sum(e^{x_j})` | 多分类输出层（配合 CrossEntropyLoss 时内置于 loss 中） |

均无内部参数，`parameters()` 返回空列表。

#### 2.2.4 Sequential 容器 (`nn/container.py`)

| 变量/属性 | 类型 | 含义 |
|-----------|------|------|
| `_modules` | `OrderedDict[str, Module]` | 按顺序存储的子模块，key 为 "0", "1", "2", ... |

**forward**: 按顺序将输入依次通过每个子模块。

#### 2.2.5 参数初始化 (`nn/init.py`)

| 函数 | 公式 | 适用场景 |
|------|------|----------|
| `xavier_uniform_(tensor)` | `U[-sqrt(6/(fan_in+fan_out)), sqrt(6/(fan_in+fan_out))]` | Sigmoid/Tanh 激活 |
| `xavier_normal_(tensor)` | `N(0, sqrt(2/(fan_in+fan_out)))` | Sigmoid/Tanh 激活 |
| `he_uniform_(tensor)` | `U[-sqrt(6/fan_in), sqrt(6/fan_in)]` | ReLU 激活 |
| `he_normal_(tensor)` | `N(0, sqrt(2/fan_in))` | ReLU 激活（推荐） |

`fan_in` 从 `tensor.shape[1]` 计算（权重形状为 `(out, in)`）。

#### 2.2.6 损失函数 (`nn/loss.py`)

| 类 | 参数 | 含义 |
|----|------|------|
| `MSELoss` | `reduction: str = 'mean'` | 均方误差损失 `L = reduce((pred - target)^2)` |
| `CrossEntropyLoss` | `reduction: str = 'mean'` | 交叉熵损失（内置 softmax 以保数值稳定）`L = -mean(log(softmax(pred)) * one_hot(target))` |

**关键梯度公式：**
- MSELoss backward: `2 * (pred - target) / n`（mean reduction）
- CrossEntropyLoss backward: `(softmax(pred) - one_hot(target)) / n`（mean reduction，softmax + CE 组合梯度的简化形式）

---

### 2.3 优化器模块（dlframe/optim/）

#### 2.3.1 SGD 优化器 (`optim/sgd.py`)

| 变量/属性 | 类型 | 含义 |
|-----------|------|------|
| `params` | `list[Parameter]` | 待优化的参数列表 |
| `lr` | `float` | 学习率 $\eta$ |
| `momentum` | `float` | 动量系数 $\beta$（0 表示纯 SGD） |
| `weight_decay` | `float` | L2 正则化强度 $\lambda$（0 表示不加） |
| `_v` | `list[np.ndarray] \| None` | 动量缓存（与 params 一一对应） |

**更新公式**: `v = β*v + g + λ*w; w = w - lr * v`（含 weight decay 的形式）

#### 2.3.2 Adam 优化器 (`optim/adam.py`)

| 变量/属性 | 类型 | 含义 |
|-----------|------|------|
| `params` | `list[Parameter]` | 待优化的参数列表 |
| `lr` | `float` | 学习率 $\eta$，默认 0.001 |
| `betas` | `tuple[float, float]` | $(\beta_1, \beta_2)$，默认 (0.9, 0.999) |
| `eps` | `float` | 数值稳定常数，默认 1e-8 |
| `weight_decay` | `float` | L2 正则化强度，默认 0 |
| `_m` | `list[np.ndarray]` | 一阶矩估计（梯度指数移动平均） |
| `_v` | `list[np.ndarray]` | 二阶矩估计（梯度平方指数移动平均） |
| `_t` | `int` | 时间步计数器 |

**更新公式**: 见开题报告式 (5)(6)

---

### 2.4 数据加载模块（dlframe/data/）

#### 2.4.1 DataLoader 类 (`data/dataloader.py`)

| 变量/属性 | 类型 | 含义 |
|-----------|------|------|
| `X` | `np.ndarray` | 特征数据，形状 `(n_samples, ...)` |
| `y` | `np.ndarray` | 标签数据，形状 `(n_samples,)` |
| `batch_size` | `int` | 每个 mini-batch 的样本数 |
| `shuffle` | `bool` | 每个 epoch 是否打乱数据 |
| `drop_last` | `bool` | 是否丢弃最后不足 batch_size 的批次 |
| `_n_samples` | `int` | 总样本数 |
| `_n_batches` | `int` | 每个 epoch 的批次数 |

**方法：**

| 方法 | 签名 | 功能 |
|------|------|------|
| `__iter__()` | `-> generator` | 按 batch 迭代 (X_batch, y_batch) |
| `__len__()` | `-> int` | 返回每个 epoch 的批次数 |

---

## 三、数据流示例

以 MNIST MLP 训练一步为例：

```
1. DataLoader.__iter__() → (X_batch [32, 784], y_batch [32])
2. model(X_batch):  # Sequential[Linear(784→128), ReLU, Linear(128→64), ReLU, Linear(64→10)]
   → Module.__call__ → Sequential.forward → 逐层调用 forward
   → 每次 Linear.forward 执行 MatMul.apply → 记录 grad_fn
   → 每次 ReLU.forward 执行 ReLUFunc.apply → 记录 grad_fn
   → 最终输出 logits [32, 10]
3. loss = CrossEntropyLoss(logits, y_batch) → 标量 Tensor
4. optimizer.zero_grad() → 清零所有 Parameter.grad
5. loss.backward():
   → 从 loss 出发拓扑排序
   → 逆序调用各节点的 _grad_fn（CrossEntropy → MatMul → ...）
   → 梯度累积到各 Parameter.grad
6. optimizer.step() → 使用 .grad 更新 Parameter.data
```

---

## 四、文件列表与依赖关系

```
dlframe/
├── __init__.py           # 导出 Tensor, Parameter, Function, nn, optim, data
├── tensor.py             # Tensor, Context（无内部依赖）
├── autograd.py           # Function, Add, MatMul, ReLUFunc, Mul, Neg, Sub, Sum, Reshape, Transpose, SigmoidFunc, TanhFunc
│                           （依赖 tensor）
├── parameter.py          # Parameter(Tensor)（依赖 tensor）
├── nn/
│   ├── __init__.py       # 导出 Module, Linear, Sequential, ReLU, Sigmoid, Tanh, Softmax, init, MSELoss, CrossEntropyLoss
│   ├── module.py         # Module 基类（依赖 parameter）
│   ├── linear.py         # Linear(Module)（依赖 module, parameter, autograd）
│   ├── activation.py     # ReLU, Sigmoid, Tanh, Softmax(Module)（依赖 module, autograd）
│   ├── container.py      # Sequential(Module)（依赖 module）
│   ├── loss.py           # MSELoss, CrossEntropyLoss(Module)（依赖 module, autograd）
│   └── init.py           # xavier_uniform_, etc.（纯 NumPy，无框架依赖）
├── optim/
│   ├── __init__.py       # 导出 SGD, Adam
│   ├── sgd.py            # SGD（纯 NumPy，依赖 parameter）
│   └── adam.py           # Adam（纯 NumPy，依赖 parameter）
└── data/
    ├── __init__.py       # 导出 DataLoader
    └── dataloader.py     # DataLoader（纯 NumPy，无框架依赖）
```
