# 01 · 基础类型与 Tensor

本章覆盖 compat 层最底层、最核心的两类工作:

- **基础类型对齐**:`ScalarType`、`TypeMeta`、`SymInt`、`Layout`、`Storage`、`DispatchKey{,Set}`、`pointer / intrusive_ptr`
- **Tensor 接口与操作**:`TensorBase` / `Tensor` 上的查询、引用计数、形状操作、resize 与 storage 对齐、`from_blob` / `pin_memory`、`TensorAccessor`
- **Sparse 与稀疏张量**:`is_sparse{,_csr}`、`sparse_coo_tensor` / `sparse_csr_tensor` / `coalesce` / `_nnz`、Paddle DenseTensor ↔ Sparse 转换

## 子文档

| # | 文件 | 主要内容 |
|---|---|---|
| 01 | [01-基础类型对齐.md](01-基础类型对齐.md) | ScalarType / TypeMeta / SymInt / Layout / Storage / DispatchKey / pointer |
| 02 | [02-Tensor接口与操作.md](02-Tensor接口与操作.md) | Tensor.reset / defined / weak_use_count / MaybeResetHolder / TensorAccessor / squeeze / unsqueeze / flatten / narrow / select / split / resize_ / as_strided / from_blob / pin_memory |
| 03 | [03-Sparse与稀疏张量.md](03-Sparse与稀疏张量.md) | Sparse APIs / _nnz / is_coalesced / coalesce / dense-sparse 转换 |

## 涉及 PR

**Paddle 主仓**(27 个):#77127、#77176、#77185、#77270、#77301、#77303、#77319、#77388、#77399、#77416、#77498、#77544、#77581、#77614、#77713、#78037、#78099、#78244、#78255、#78257、#78266、#78554、#78576、#78580、#78581、#78609、#78633、#78807、#78809、#78826、#78837

**PaddleCppAPITest**(8 个):#9、#13、#14、#17、#18、#19、#21、#23、#25、#26、#34、#56

## 关键架构决策

1. **TypeMeta 取代裸 ScalarType**(#78257):上游 PyTorch 在 `c10/util/typeid.h` 中以 `TypeMeta` 作为 dtype 的运行时句柄,`ScalarType` 是其上的枚举投影。早期 compat 层只用 `ScalarType` 是不够的(`TensorOptions::dtype()` 需要 `TypeMeta`),因此引入完整的 `c10::util::typeid.{h,cpp}`(+744 行)+ `ScalarTypeToTypeMeta.h` 桥接。

2. **SymInt 从别名提升为类**(#78807):早期 `c10::SymInt = int64_t`,但 PyTorch 的 `SymInt` 是一个支持符号表达的包装类。当 `IntArrayRef` / `SymIntArrayRef` 互相隐式转换时,简单别名会引入类型歧义。后期改为独立 `class SymInt` + `SymIntArrayRef = ArrayRef<SymInt>`,通过 `operator int64_t()`、`guard_int()`、`maybe_as_int()` 维持源码兼容。

3. **Storage 与 holder 同步**(#78826):Paddle `DenseTensor::Holder()` 在外部扩展 kernel 里被替换时,compat 层的 `TensorBase::storage()` 必须重新读取最新 holder,否则下游通过 `storage().data()` 拿到悬空指针 — 由 `MaybeResetHolder` 解决。

4. **Tensor 方法实现移到 `ops/`**(#77713):上游 PyTorch 是 `torchgen` 从 `native_functions.yaml` 生成 `TensorBody.h`,通过 `MethodOperators.h` 串接 ops 实现。compat 层手动模拟这套机制,Tensor 方法的内联实现从 `TensorBody.h` 搬到 `ops/*.h`,降低头文件耦合并避免 ODR 冲突。

按文档顺序阅读即可。
