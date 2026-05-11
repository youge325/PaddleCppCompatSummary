# 03 · Sparse 与稀疏张量

> 范围:`is_sparse` / `is_sparse_csr`、`sparse_coo_tensor` / `sparse_csr_tensor` 构造、`_nnz` / `is_coalesced` / `coalesce` / `_values`、Paddle Dense ↔ Sparse 转换工具、稀疏 op 的 dispatch。

## 1. 主题概述

PyTorch 同时支持 dense、COO 稀疏、CSR 稀疏 三种 layout,通过 `Layout::Strided` / `Layout::Sparse` / `Layout::SparseCsr` 区分。compat 层需要让下游(例如某些图神经网络、Embedding 算子库)能用 `at::sparse_coo_tensor(indices, values, sizes)` 之类的 API 直接构造稀疏张量。

Paddle 内部有 `phi::SparseCooTensor` / `phi::SparseCsrTensor`,但它们的 layout 表示、构造接口、kernel 注册都与 dense 路径分开。compat 层的工作包括:

1. **Layout 谓词** — `is_sparse` / `is_sparse_csr` 透过 `phi::Tensor` 的内部类型识别返回。
2. **稀疏构造接口** — `sparse_coo_tensor(...)` / `sparse_csr_tensor(...)` 把 PyTorch 风格的 `(indices, values, sizes)` 翻译成 Paddle 的 `SparseCooTensor::Init(...)`。
3. **稀疏 op** — `_nnz()`、`is_coalesced()`、`coalesce()`、`_values()` 等 COO/CSR 标准操作。
4. **Dense ↔ Sparse 转换** — 提供 `utils/dense_sparse_conversion.h` 工具,让外部 kernel 在两种 layout 间互转。

## 2. PR 汇总表

| PR# | 状态 | 合并日期 | 标题 |
|---:|---|---|---|
| [#77416](https://github.com/PaddlePaddle/Paddle/pull/77416) | CLOSED | — | add DispatchKeySet and Sparse related APIs(后被拆解) |
| [#77581](https://github.com/PaddlePaddle/Paddle/pull/77581) | MERGED | 2026-02-09 | add Sparse related APIs(从 #77416 拆出) |
| [#78037](https://github.com/PaddlePaddle/Paddle/pull/78037) | MERGED | 2026-03-02 | add and refactor some compat APIs(其中含 `coalesce`、`_nnz`、`_values`、`is_coalesced`、稀疏构造重载、dense_sparse_conversion) |
| [#78255](https://github.com/PaddlePaddle/Paddle/pull/78255) | MERGED | 2026-03-19 | Add `pin_memory` support and fix `from_blob`(其中含 `sparse_coo_tensor`/`sparse_csr_tensor` 接入 pin_memory) |
| [#78837](https://github.com/PaddlePaddle/Paddle/pull/78837) | MERGED | 2026-05-01 | Align some other APIs(其中含 `sparse_coo_tensor`/`sparse_csr_tensor` 行为对齐) |
| [PFCCLab#23](https://github.com/PFCCLab/PaddleCppAPITest/pull/23) | MERGED | 2026-03-01 | add sparse related API tests |

## 3. 关键 PR 深度分析

### 3.1 `is_sparse` / `is_sparse_csr` 与构造接口 — #77581(+839/-37 行,18 个文件)

#### 背景

`#77416` 试图一次性补齐 `is_sparse` / `is_sparse_csr` / `is_quantized` / `is_meta` / `is_inference` / `is_nested` 6 个 layout 谓词 + 配套 `sparse_coo_tensor`/`sparse_csr_tensor` 构造接口 + 在 Paddle 内核侧加 `Layout` 字段。导师 review 时提出:

- `is_quantized` / `is_meta` / `is_inference` / `is_nested` 暂时没人用,可以延后
- Paddle 内核的 `Layout` 修改与 compat 接口添加耦合度低,应拆分

因此 #77416 关闭,Sparse 部分拆为 #77581 单独上,DispatchKeySet 留给 #77399(仍 OPEN)。

#### 改动

- **`TensorBase.h`** — 加 `is_sparse()` / `is_sparse_csr()` 谓词。
- **`ATen/Utils.{h,cpp}`** — 加 `_PD_GetSparseType()` 内部 helper 区分 COO / CSR。
- **`ops/sparse_coo_tensor.h`**(+69) — `at::sparse_coo_tensor(indices, values, sizes, opts)` 实现:
  - 检查 `indices.dtype() == kLong`、`indices` shape 为 `(sparse_dim, nnz)`
  - 构造 `phi::SparseCooTensor`,调用 `SetMember(non_zero_indices, non_zero_elements, sparse_dim, is_coalesced)`
  - 包装回 compat 层 `Tensor`
- **`ops/sparse_csr_tensor.h`**(+98) — 同理,接收 `(crow_indices, col_indices, values, sizes, opts)`,构造 `phi::SparseCsrTensor`。
- **`ops/empty.h`** / **`empty_like.h`** / **`zeros.h`** / **`zeros_like.h`** — 接入 `opts.layout()`,sparse 路径走 `SparseCooTensor::Init()` 而非 dense 路径。
- **Paddle 内核侧**:
  - `phi/core/sparse_coo_tensor.cc`(+4/-2)、`sparse_csr_tensor.cc`(+4/-1):暴露 `dims_` 给 compat 读取。
  - `phi/core/tensor_meta.h`(+1/-1):导出 `Layout` 字段。
  - `phi/infermeta/unary.cc`(+10):为稀疏路径加 shape 推断分支。
  - `phi/kernels/sparse/cpu/sparse_utils_kernel.cc`(+9)、`sparse/gpu/matmul_kernel.cu`(+2)、`sparse/gpu/sparse_utils_kernel.cu`(+9):补 kernel 注册。
- **测试**:`c10_layout_test.cc`(+522 行)从 0 开始一次写齐 COO/CSR 构造、shape 校验、`Layout` 谓词。

#### 效果

下游可以用 `at::sparse_coo_tensor` / `at::sparse_csr_tensor` 直接构造稀疏 Tensor,通过 `t.is_sparse()` / `t.is_sparse_csr()` 路由分支。`c10_layout_test.cc` 26 个 case 全 pass。

### 3.2 `coalesce` / `_nnz` / `_values` / `is_coalesced` — #78037 的稀疏部分

`#78037` 是综合 PR(+2917 行),稀疏相关的部分:

- **`ops/coalesce.h`**(+34) — `tensor.coalesce()`:把 COO Tensor 的 indices 排序并合并重复元素。内部调用 Paddle 的 `phi::sparse::CoalesceKernel`。
- **`ops/_nnz.h`**(+44) — `tensor._nnz()`:返回 COO/CSR 的 non-zero 元素数。COO 时返回 `indices.size(1)`,CSR 时返回 `values.numel()`。
- **`ops/is_coalesced.h`**(+33) — `tensor.is_coalesced()`:查询 COO Tensor 的 coalesce flag。
- **`ops/_values.h`**(+46) — `tensor._values()`:返回 COO/CSR 的 values 张量(以 dense Tensor 形式)。
- **`utils/dense_sparse_conversion.h`**(+49) — 提供 `_PD_DenseToSparseCoo()` / `_PD_SparseCooToDense()` / 等工具函数,供 op 实现侧统一转换。
- **`ops/sparse_csr_tensor.h`**(+37/-9) — 在原有 #77581 基础上补全 CSR 构造的多种重载(支持 `dtype`/`device`/`layout`/`pin_memory` 四参组合)。

后续在 #78837 中,`sparse_coo_tensor.h`(+3/-6)、`sparse_csr_tensor.h`(+3/-6) 又做了行为对齐(主要是 `indices` dtype 检查更严格、`values` 与 `sizes` 一致性校验对齐 PyTorch 异常消息)。

### 3.3 Dense ↔ Sparse 工具函数 — `utils/dense_sparse_conversion.h`(在 #78037 中引入,+49 行)

提供 4 个内部函数:

```cpp
// utils/dense_sparse_conversion.h
namespace _PD_Internal {
phi::SparseCooTensor _PD_DenseToSparseCoo(const phi::DenseTensor& dense);
phi::SparseCsrTensor _PD_DenseToSparseCsr(const phi::DenseTensor& dense);
phi::DenseTensor    _PD_SparseCooToDense(const phi::SparseCooTensor& coo);
phi::DenseTensor    _PD_SparseCsrToDense(const phi::SparseCsrTensor& csr);
}
```

这些 helper 只在 compat 内部使用,**不暴露到 `at::` 命名空间**,因为 PyTorch 中 dense ↔ sparse 转换是 op 形式(`tensor.to_sparse()`、`tensor.to_dense()`),而非工具函数。compat 层之所以保留这层 helper,是为了在 `_values` / `_nnz` / `coalesce` 等 op 实现里复用同一份转换逻辑。

测试侧 `ATen_coalesce_test.cc`(+115)、`ATen_nnz_test.cc`(+96)、`ATen_values_test.cc`(+112)、`compat_dense_sparse_conversion_test.cc`(+61) 共 +384 行单测。

### 3.4 Sparse Tensor 与 `pin_memory` 的交叉 — #78255 中的相关部分

`#78255` 主要做 `pin_memory` 与 `from_blob`,但里面有一项不太显眼的工作:**为 `sparse_coo_tensor`(+43/-19)与 `sparse_csr_tensor`(+28/-14)接入 `pin_memory` 参数**。

PyTorch 中:

```cpp
at::sparse_coo_tensor(indices, values, sizes,
                       at::TensorOptions()
                         .device(at::kCPU)
                         .pinned_memory(true));
```

这种用法在多 GPU 数据并行中常见 — 把稀疏 indices/values 准备在 pinned host memory,再异步 H2D 拷贝到 GPU。compat 层在 #78255 中保证了这个路径走通:把 `opts.pinned_memory()` 透传到底层 `phi::SparseCooTensor` 的 `Place`,通过 `utils/pinned_place.h` 取 `phi::GPUPinnedPlace`。

## 4. 其他相关工作简注

| PR# | 标题 | 简注 |
|---:|---|---|
| [#77416](https://github.com/PaddlePaddle/Paddle/pull/77416) (CLOSED) | add DispatchKeySet and Sparse related APIs | 历史首次尝试,与 DispatchKeySet 耦合被关闭。其完整代码迁移路径:`is_sparse`/`is_sparse_csr` → #77581;DispatchKeySet → #77399 + #78070;`c10_layout_test.cc`(+522 行)→ #77581 中保留。 |

## 5. 与 PaddleCppAPITest 的对应关系

| 主仓 PR | 测试仓 PR | 测试文件最终路径 |
|---|---|---|
| #77581 Sparse | [#23](https://github.com/PFCCLab/PaddleCppAPITest/pull/23) | `test/ATen/ops/SparseTensorTest.cpp`、`SparseTensorExtraTest.cpp`(经 #58 重组) |
| #78037 中的 coalesce/_nnz/_values | [#40](https://github.com/PFCCLab/PaddleCppAPITest/pull/40) | `test/ATen/ops/CoalesceTest.cpp`(+150 行) |

在测试仓 #46 的 100% MATCH 表里 `SparseTensorTest`、`SparseTensorExtraTest`、`CoalesceTest` 全部 MATCH。

## 6. 已知差异(留待后续)

虽然测试仓 mismatch 表已经清空稀疏部分,但仍有两个**已知差异**未对齐:

1. **`coalesce()` 的 indices 排序顺序** — PyTorch 用 lexicographic 排序(行优先),Paddle 用稳定排序但对 indices 的扁平化顺序略有不同。当输入有重复 indices 时,输出 values 元素的顺序可能不一致,但合并结果数值相同 — `mismatch_api_record.md` 中标注为"语义等价、字节非等价"。
2. **`to_sparse()` / `to_dense()` 反向转换** — PyTorch 支持 `tensor.to_sparse(sparse_dim)`,compat 层暂未实现 — 详见 `doc/ATen/ops/mismatch_api_record.md`。

这些差异不影响 DeepEP / FastDeploy 等当前下游,优先级较低。
