# 02 · Tensor 接口与操作

> 范围:`TensorBase` / `Tensor` 的查询接口、引用计数、形状操作、`resize_` / `as_strided`、`from_blob` / `pin_memory`、`TensorAccessor`、`MaybeResetHolder` 等。

## 1. 主题概述

`at::Tensor` 是 PyTorch C++ API 的核心抽象。它是一个 `c10::intrusive_ptr<TensorImpl>` 的轻量包装,通过成员函数暴露**几百个**算子(`sum`、`reshape`、`narrow`、`select`、`flatten`、`resize_`、`as_strided`、`item`、`pin_memory` …)。

Paddle 的对应物是 `phi::DenseTensor`(+ 部分 `SelectedRows`、`SparseCooTensor`、`SparseCsrTensor`)。compat 层在 `TensorBase` 内部持有 `shared_ptr<DenseTensor> impl_`,所有方法实现都是"把 `at::*` 入参翻译成 Paddle 风格,调用 `phi::*` kernel,再把结果包装回 `Tensor`"。

挑战在三处:

1. **算子重载数量爆炸** — `narrow` 有 `narrow(int dim, int64_t start, int64_t length)` / `narrow(int dim, SymInt start, SymInt length)` / `narrow_copy(...)` / `narrow_copy_symint(...)` 4 个变体,且要支持成员调用和自由函数调用。compat 层最初把所有实现写在 `core/TensorBody.h`,后来发现头文件膨胀 + ODR 风险,在 #77713 中搬到 `ops/`。
2. **存储与跨平台的微妙差异** — `resize_` 在 PyTorch 中会按 storage 对齐重新分配(`numel * dtype_size`);Paddle `DenseTensor::Resize` 只调整 dim 不重分配。当下游(DeepEP)用 `tensor.resize_(...)` 跟着 `tensor.data_ptr()` 写入,如果对齐错就直接 SegFault。#78609、#78576、#78633 处理这一系列细节。
3. **外部 Holder 替换** — 用户在自定义 kernel 中可能调用 `DenseTensor::set_holder(...)` 替换底层内存(典型场景:从外部分配池接管 storage),compat 层的 `Storage` 对象必须感知这种替换,否则 `tensor.storage().data()` 拿到的是旧指针 — #78826 用 `MaybeResetHolder` 修复。

## 2. PR 汇总表

### 2.1 Tensor 基础接口

| PR# | 状态 | 合并日期 | 标题 |
|---:|---|---|---|
| [#77127](https://github.com/PaddlePaddle/Paddle/pull/77127) | CLOSED | — | add reset API(被 #77319 替代) |
| [#77319](https://github.com/PaddlePaddle/Paddle/pull/77319) | MERGED | 2026-01-13 | Add `Tensor.reset` API(重交) |
| [#77498](https://github.com/PaddlePaddle/Paddle/pull/77498) | MERGED | 2026-02-02 | add TensorAccessor related API |
| [#78809](https://github.com/PaddlePaddle/Paddle/pull/78809) | MERGED | 2026-04-30 | Align TensorBase::weak_use_count with PyTorch |
| [#78826](https://github.com/PaddlePaddle/Paddle/pull/78826) | MERGED | 2026-05-02 | Fix MaybeResetHolder |
| [PFCCLab#9](https://github.com/PFCCLab/PaddleCppAPITest/pull/9) | MERGED | 2026-03-21 | Add test for `defined` and `reset` and Complement tests |
| [PFCCLab#25](https://github.com/PFCCLab/PaddleCppAPITest/pull/25) | MERGED | 2026-03-01 | add TensorAccessor related tests |

### 2.2 形状操作

| PR# | 状态 | 合并日期 | 标题 |
|---:|---|---|---|
| [#77270](https://github.com/PaddlePaddle/Paddle/pull/77270) | MERGED | 2026-01-15 | complement squeeze and unsqueeze API |
| [#77544](https://github.com/PaddlePaddle/Paddle/pull/77544) | MERGED | 2026-01-31 | add flatten and narrow related API |
| [#77614](https://github.com/PaddlePaddle/Paddle/pull/77614) | MERGED | 2026-03-03 | add select and split related API |
| [#77713](https://github.com/PaddlePaddle/Paddle/pull/77713) | MERGED | 2026-02-12 | Move Tensor method implementation to ops folder |
| [#78037](https://github.com/PaddlePaddle/Paddle/pull/78037) | MERGED | 2026-03-02 | add and refactor some compat APIs |
| [#78099](https://github.com/PaddlePaddle/Paddle/pull/78099) | MERGED | 2026-03-10 | Align some compat APIs with libtorch |
| [#78266](https://github.com/PaddlePaddle/Paddle/pull/78266) | MERGED | 2026-03-23 | Rename test `compat_squeeze_test` → `ATen_squeeze_test` |
| [#78837](https://github.com/PaddlePaddle/Paddle/pull/78837) | MERGED | 2026-05-01 | Align some other APIs(`chunk`/`expand`/`index`/`values`) |
| [PFCCLab#17](https://github.com/PFCCLab/PaddleCppAPITest/pull/17) | MERGED | 2026-03-01 | add squeeze and unsqueeze API test |
| [PFCCLab#24](https://github.com/PFCCLab/PaddleCppAPITest/pull/24) | MERGED | 2026-01-27 | Rewrite test files |
| [PFCCLab#26](https://github.com/PFCCLab/PaddleCppAPITest/pull/26) | MERGED | 2026-02-03 | add flatten and narrow related tests |
| [PFCCLab#40](https://github.com/PFCCLab/PaddleCppAPITest/pull/40) | MERGED | 2026-03-21 | add some compat API tests(sum/t/t_/transpose_/view_as/item) |

### 2.3 resize_ / as_strided / from_blob / pin_memory

| PR# | 状态 | 合并日期 | 标题 |
|---:|---|---|---|
| [#78255](https://github.com/PaddlePaddle/Paddle/pull/78255) | MERGED | 2026-03-19 | Add `pin_memory` support and fix `from_blob` |
| [#78554](https://github.com/PaddlePaddle/Paddle/pull/78554) | MERGED | 2026-04-02 | Align resize api |
| [#78576](https://github.com/PaddlePaddle/Paddle/pull/78576) | MERGED | 2026-04-07 | Add `TORCH_WARN` macro and fix resize_ |
| [#78609](https://github.com/PaddlePaddle/Paddle/pull/78609) | MERGED | 2026-04-10 | Fix `resize_` with storage alignment |
| [#78632](https://github.com/PaddlePaddle/Paddle/pull/78632) | CLOSED | — | Skip ATen resize test on Mac(后被 Windows PR 中的条件编译覆盖) |
| [#78633](https://github.com/PaddlePaddle/Paddle/pull/78633) | MERGED | 2026-04-10 | Fix `as_strided` to align `resize_` |
| [#78552](https://github.com/PaddlePaddle/Paddle/pull/78552) | MERGED | 2026-04-02 | Fix arange default dtype |
| [#78551](https://github.com/PaddlePaddle/Paddle/pull/78551) | MERGED | 2026-04-02 | Align device related APIs |

---

## 3. 关键 PR 深度分析

### 3.1 Tensor 方法实现从 `core/` 搬到 `ops/` — #77713

#### 背景

PyTorch 的代码生成器(`torchgen/gen.py`)读 `aten/src/ATen/native/native_functions.yaml`,把每个 op 拆成:

- 自由函数 `at::narrow(tensor, dim, start, length)` 在 `ATen/Functions.h`
- Tensor 成员 `tensor.narrow(dim, start, length)` 在 `ATen/core/TensorBody.h`
- 算子结构体 `at::_ops::narrow` 在 `ATen/ops/narrow_ops.h`,负责实际分发

`TensorBody.h` 内的成员实现只是**调用 `at::_ops::narrow::call(...)`**。这种生成结构让单个 op 的实现集中在一个地方,而 `TensorBody.h` 保持苗条。

早期 compat 层直接把成员实现内联在 `core/TensorBody.h`,导致这个文件不断膨胀(到 #77713 之前已经膨胀到包含 reshape、sum、abs、empty、arange、cat、from_blob、full、ones、zeros、empty_like、empty_strided、zeros_like 等十几个 op 的实现)。

#### 改动(+668/-247 行,28 个文件)

把成员实现从 `core/TensorBody.h` 拆到 `ops/<op>.h`:

```cpp
// 旧 core/TensorBody.h:
inline Tensor Tensor::narrow(int64_t dim, int64_t start, int64_t length) const {
  // ~30 行实现
}

// 新 core/TensorBody.h:
inline Tensor Tensor::narrow(int64_t dim, int64_t start, int64_t length) const;
// 实现迁移到 ops/narrow.h

// 新 ops/narrow.h:
inline Tensor Tensor::narrow(...) const { ... }
inline Tensor narrow(const Tensor& self, ...) { ... }  // 自由函数也在这里
```

涉及 op:`flatten`/`unflatten`、`index`、`narrow`/`narrow_copy`、`permute`、`squeeze`、`unsqueeze`、`slice`、`view`、`reshape`、`transpose`、`sum`、`abs`。`core/TensorBody.h` 从 +33/-192,即净减 159 行成员实现。

#### 效果

- 单个 op 增删互不干扰,header 编译时间下降。
- 后续添加 op 时,所有 5 个变体(int / SymInt / Dimname / 自由函数 / 成员函数)集中在一个 `ops/<op>.h`,便于 review。
- `ATen_reshape_test.cc`(+54 行)在该 PR 落地,确认搬迁不破坏行为。

### 3.2 `flatten`/`unflatten`/`narrow` — #77544(+500 行)

#### 背景

`narrow` 是个易错算子:

```python
narrow(int dim, int64_t start, int64_t length)             # 整数版
narrow_copy(int dim, int64_t start, int64_t length)         # 整数版,返回拷贝
narrow_symint(int dim, SymInt start, SymInt length)         # SymInt 版
narrow_copy_symint(int dim, SymInt start, SymInt length)    # SymInt 版,返回拷贝
narrow(int dim, const Tensor& start, int64_t length)        # Tensor start
```

`flatten`/`unflatten` 同样有 SymInt 重载和 `Dimname` 重载(后者不在 compat 范围内,见 PR 备注)。

PR 描述里特意提了**未来扩展方向**:

> 我觉得如果要兼容的算子少的话,直接在 `paddle/phi/api/include/compat/ATen/core/TensorBody.h` 实现就行,如果多的话,也可以在编译时执行 python 脚本将公共 API 封装 `paddle/phi/api/include/compat/ATen/ops/**.h` 写入 `paddle/phi/api/include/compat/ATen/core/TensorBody.h`。

— 这一思考后来在 #77713 中落地为"实现搬到 ops/"。

#### 改动

`TensorBody.h`(+88 行)添加 7 个声明;`ATen_flatten_test.cc`(+202 行)+ `ATen_narrow_test.cc`(+208 行)各覆盖 SymInt / 非 SymInt 路径、负 dim、越界 length。配套 [PFCCLab#26](https://github.com/PFCCLab/PaddleCppAPITest/pull/26)(`FlattenTest.cpp` +170、`NarrowTest.cpp` +202 行)。

### 3.3 `select`/`split`/`tensor_split` 与 detach/reciprocal/masked_select — #77614(+2328 行)

#### 背景

split 系算子有非常多变体:

```cpp
std::vector<Tensor> split(int64_t split_size, int64_t dim);
std::vector<Tensor> split(const Tensor& self, IntArrayRef split_sizes, int64_t dim);
std::vector<Tensor> tensor_split(int64_t sections, int64_t dim);
std::vector<Tensor> tensor_split(IntArrayRef indices, int64_t dim);
std::vector<Tensor> tensor_split(const Tensor& tensor_indices, int64_t dim);
std::vector<Tensor> unsafe_split(...);  // 不做 in-place 检查
std::vector<Tensor> split_with_sizes(...);
std::vector<Tensor> unsafe_split_with_sizes(...);
std::vector<Tensor> hsplit/vsplit/dsplit(...);  // 沿固定 axis
```

外加 `select`(取单个切片,降维)、`masked_select`(布尔掩码)、`detach`/`detach_`、`reciprocal`/`reciprocal_`。

#### 改动

- 15 个 `ops/*.h` 文件新增,共 +700 余行接口实现。
- 5 个 `test/cpp/compat/ATen_*_test.cc` 测试文件 +1500 余行单测(autograd / memory / select / split)。
- 顺带在 `c10/core/ScalarType.h` 加 1 行 dtype 枚举对齐。

#### 效果

测试仓 #46 的 100% diff 表中 `SplitTest`、`SelectTest`、`SliceTest`、`DetachTest`、`ReciprocalTest`、`PermuteTest`、`MiscTensorTest` 全部 MATCH。

### 3.4 `from_blob` + `pin_memory` 的真正含义 — #78255(+1346/-281 行,27 个文件)

#### 背景

`from_blob(void* data, IntArrayRef sizes, Deleter deleter, TensorOptions opts)` 是 PyTorch 用来 **"接管已有内存创建 Tensor"** 的关键接口。它在生产环境中常见于:用户从 CUDA driver 拿到 stream-allocated 内存,直接包装成 Tensor 进入网络;或者从 NumPy / OpenCV / 自定义分配器拿到 buffer。

行为契约:

1. 创建的 Tensor 的 `device` **由 ptr 推断**,而不是 `opts.device()`。早期 compat 层用 `opts.device()`(若不指定就默认 CPU),导致从 GPU 指针包装时报错 device mismatch。
2. 若 `opts.pinned_memory() == true`,内部 storage 必须挂 `phi::PinnedPlace`,而不是 `phi::CPUPlace`,否则后续 `cudaMemcpyAsync` 拿不到 pinned 速率。
3. `deleter` 在 Tensor 析构时被调用。

#### 改动

- **`ops/from_blob.h`**(+160/-39) — 不再用 `opts.device()`,改用 `paddle::from_blob` 内部对指针的推断(GPU 指针自动得到 `GPUPlace`)。当 `opts.device()` 显式提供时仍尊重用户意图。
- **`utils/pinned_place.h`**(+41) — 新增工具函数 `get_pinned_place()`,从 `TensorOptions::pinned_memory()` 取 `phi::GPUPinnedPlace`。
- **所有创建型 op**(`empty`、`empty_like`、`arange`、`eye`、`full`、`ones`、`zeros`、`new_empty`、`new_full`、`new_ones`、`new_zeros`、`sparse_coo_tensor`、`sparse_csr_tensor`、`zeros_like`)— 全部接入 `pin_memory` 参数,通过 `pinned_place.h` 取得正确的 `Place`。共 +400 余行修改。
- **`TensorBody.h::pin_memory()`**(+59/-15) — 对齐 PyTorch 行为:在 CUDA 设备上 pin 当前 host 内存(若不在 pinned place,会做一次 copy)。
- **`ops/abs.h`**(+15) — 因为 PyTorch 的 abs 对 transpose 后的非连续 tensor 保持非连续,Paddle 则强制 contiguous(详情见 #78099 PR 描述),compat 层无法行为完全对齐,这里加 `TORCH_WARN` 提示用户。
- **测试**:`ATen_cuda_test.cc`(+101)、`ATen_empty_test.cc`(+101)、`ATen_from_blob_test.cc`(+184)、`ATen_pin_memory_creation_test.cc`(+192) — 共 +578 行单测。

#### 效果

DeepEP 一类基于 `from_blob` 包装外部内存的代码可以原样编译。

### 3.5 `resize_` 与 storage 对齐 — #78554、#78576、#78609、#78633

这一族 PR 是 **DeepEP / DeepGEMM 编译适配的重头戏**,占 4 个 PR 共 +330 余行。

#### #78554 起步对齐(+166/-6,2 个文件)

`tensor.resize_(IntArrayRef new_size, std::optional<MemoryFormat> memory_format)` — 两个签名都要支持,`memory_format` 参数透传给 storage。补回归测试 3 个 case(基础 resize、带 memory_format、异常路径)。

#### #78576 修复 DenseTensor 接口不匹配(+86/-4)

DeepEP 编译时报错:

```
error: 'class phi::DenseTensor' has no member named 'offset'
error: 'class phi::DenseTensor' has no member named 'mutable_data'
```

`resize_` 实现用了 `dense_tensor->offset()` / `dense_tensor->mutable_data()`,这两个是较新版本 Paddle 才有的接口。改用 `dense_tensor->Holder()->size()` + 显式计算偏移、用 `dense_tensor->AllocateFrom(...)` 替代 `mutable_data`,避免对 Paddle 内部接口的耦合。

附:`TORCH_WARN` 宏(+81 行)在该 PR 一并落地 — 因为对齐 PyTorch warning 接口的工作量很小,且 `resize_` 在某些边界条件(如 `new_size != old_size` 且无重分配)需要 warn,顺带做掉。

#### #78609 storage 真正对齐(+66/-21)

PR 描述:"根据报告的 `resize_` 接口兼容差异添加测试"。

实质性改动:`resize_` 时若新 `numel * dtype_size <= old_storage_bytes`,只更新 dims、不重分配;若超出,重新 `AllocateFrom` 并 `memcpy` 旧数据。这套行为完全对齐 PyTorch `at::native::resize_impl_cpu_/cuda_`。

`slice.h`(+3)新增一行,确保 `at::slice` 返回的 view 在被 `resize_` 时正确处理(slice 是 storage 的 view,不允许任意 resize)。

#### #78633 `as_strided` 跟随 `resize_`(+20)

PyTorch 的 `as_strided` 创建对 storage 不同 offset / stride 的 view。当原 Tensor 被 `resize_` 后,`as_strided` 创建的 view 是否还有效?对齐 PyTorch 语义:**只要新 storage 仍覆盖 `as_strided` 的访问范围,view 仍有效**;否则触发 storage 失效检查。

加 14 行 `c10_storage_test.cc` 覆盖。

#### #78632 → 平台问题留尾(CLOSED)

PR 描述只有一句:"Mac 编译器的行为略有不同,先跳过测试,防 pr 阻塞"。后来导师建议合并到 Windows PR 一起处理,该 PR 关闭,Mac 的条件编译合到 #78670。

### 3.6 `MaybeResetHolder` — 外部 Holder 替换感知 — #78826(+85/-1)

#### 背景

`phi::DenseTensor` 是 Paddle 的核心张量类型,有一个 `set_holder(...)` 方法允许外部 kernel 替换底层 `Holder`(`shared_ptr<Allocation>`)。典型场景:

```cpp
// 外部 kernel:从自定义分配器拿一段内存,替换到 tensor 上
auto* dt = static_cast<phi::DenseTensor*>(tensor.impl_.get());
dt->set_holder(my_custom_holder);
```

compat 层的 `Storage` 对象在构造时**缓存了**首次读取的 holder。set_holder 之后,`tensor.storage().data()` 返回的是缓存的旧指针,**与实际不一致**。

#### 改动

在 `TensorBase.h` 添加 `MaybeResetHolder()` helper(+24/-1):

```cpp
class TensorBase {
  mutable std::shared_ptr<phi::Allocation> cached_holder_;
  
  void MaybeResetHolder() const {
    auto current = impl_->Holder();
    if (cached_holder_ != current) {
      cached_holder_ = current;
      // 更新 storage_ 的内部状态
    }
  }
  
  const Storage& storage() const {
    MaybeResetHolder();
    return storage_;
  }
};
```

#### 验证

新增 `ATen_resize_custom_kernel_test.cc`(+60 行),写一个伪外部 kernel 替换 holder,验证 `tensor.storage().data()` 反映了新 holder 的指针。这是 compat 层与 Paddle 外部扩展能力解耦的关键修复。

### 3.7 `TensorAccessor` — #77498(+506 行,5 个文件)

#### 背景

`Tensor.packed_accessor32<T, N>()` / `packed_accessor64<T, N>()` 是 PyTorch 中**在 device 端访问 Tensor 数据**的标准方式。它将 Tensor 的 `data_ptr` + `sizes` + `strides` 打包成一个**编译期已知维数**的轻量结构,可以直接传到 CUDA kernel:

```cpp
__global__ void mykernel(at::PackedTensorAccessor32<float, 2> a) {
  int i = threadIdx.x;
  int j = threadIdx.y;
  a[i][j] += 1.0f;
}
mykernel<<<...>>>(t.packed_accessor32<float, 2>());
```

`PackedTensorAccessor32` 使用 `int32_t` 存 stride/size(减小寄存器压力),`PackedTensorAccessor64` 用 `int64_t`(适合大 Tensor)。

#### 改动

- **`ATen/core/TensorAccessor.h`**(+124) — 实现 `TensorAccessor<T, N>` 与 `PackedTensorAccessor<T, N, PtrTraits, index_t>` 模板族(N 维递归继承)。
- **`TensorBase.h`**(+115) — 在 `TensorBase` 上加 `generic_packed_accessor<T, N>()` / `packed_accessor32<T, N>()` / `packed_accessor64<T, N>()`,以及 `is_non_overlapping_and_dense()` / `has_names()`。
- **`ATen_TensorAccessor_test.cc`**(+219) — 22 个测试 case,覆盖 1D/2D/3D、32/64 索引、`__restrict__` 指针。

后续 #78662 中,把 `TensorAccessor` 的实现进一步搬到 `torch/headeronly/core/TensorAccessor.h`(+399 行),让纯 header-only kernel 库直接 include 即用。

测试仓配套 [PFCCLab#25](https://github.com/PFCCLab/PaddleCppAPITest/pull/25) 的 `TensorAccessorTest.cpp`(+99) 覆盖 packed accessor 在 device 端的内存连续性、stride 一致性。

## 4. 其他 PR 简注

| PR# | 标题 | 简注 |
|---:|---|---|
| [#77270](https://github.com/PaddlePaddle/Paddle/pull/77270) | complement squeeze and unsqueeze API | `TensorBody.h` +53 行,完善 `squeeze`/`unsqueeze` 接口的所有重载;`compat_squeeze_test.cc` +207 行(后在 #78266 中改名为 `ATen_squeeze_test`)。 |
| [#77319](https://github.com/PaddlePaddle/Paddle/pull/77319) | Add `Tensor.reset` API(重交)| 因为 #77127 强推失误,代码丢失,重交。新增 `Tensor.reset()` 接口(置空 impl_)+ 32 行单测。PR 描述附带强推失误的复盘:"幸好是在 Windows 上操作,Linux 还有备份,但是 PR 没了"。 |
| [#78037](https://github.com/PaddlePaddle/Paddle/pull/78037) | add and refactor some compat APIs | 大综合 PR(+2917/-106,39 个文件):新增 `sum`/`t`/`t_`/`transpose_`/`view_as`/`coalesce`/`is_variable`/`item`,解耦 `Utils.h` 依赖,补全 `at::tensor` 构造函数重载,引入 `Device → TensorOptions` 隐式转换,重构 `item` 行为对齐 PyTorch。 |
| [#78099](https://github.com/PaddlePaddle/Paddle/pull/78099) | Align some compat APIs with libtorch | 对齐 `arange`、`narrow`、`sum`;`abs` 因 Paddle/PyTorch 底层实现差异(Paddle 强制 contiguous,PyTorch 保留 stride 非连续)无法完全对齐,留作已知差异。 |
| [#78551](https://github.com/PaddlePaddle/Paddle/pull/78551) | Align device related APIs | `c10::Device`/`DeviceType` 补 `PrivateUse1`、`is_xpu`/`is_ipu`/`is_mps`/`is_privateuseone` 谓词、`supports_as_strided`、`set_index`、`std::hash<Device>`;`Device.cpp` 严格字符串解析对齐 PyTorch(`cuda:01` 等非法格式正确拒绝);为规避 Windows `ERROR` 宏污染,内部状态枚举改名为 `kStart`/`kError` 等。 |
| [#78552](https://github.com/PaddlePaddle/Paddle/pull/78552) | Fix arange default dtype | `at::arange(5)` 不指定 dtype 时应返回 `kLong`(PyTorch 行为),而早期 compat 层返回 `kFloat`,被 DeepGEMM 卡住。修复 `is_integral` 路径,整数输入推断 `kLong`,浮点输入跟随默认浮点 dtype。 |
| [#78555](https://github.com/PaddlePaddle/Paddle/pull/78555) | Align misc apis | 13 个文件 +421/-115:`c10::Allocator` 加 `CaptureId_t`/`MempoolId_t`/`MempoolIdHash`、`DataPtr::mutable_get()`、`SetAllocator`/`GetAllocator`、`InefficientStdFunctionContext`;`CUDAGuard` 补 `original_device()`;`CUDAStream` 补 `priority_range`/`pack3`/`unpack3`/`getStreamFromExternal`/`query`/`synchronize`/`hash`;新增 `torch/version.h`。 |
| [#78580](https://github.com/PaddlePaddle/Paddle/pull/78580) | Delete useless code and rename test files | 37 个文件 +294/-288:统一测试文件命名为 `ATen_*` / `c10_*`,清理预处理无用代码,抑制未使用变量警告。 |
| [#78837](https://github.com/PaddlePaddle/Paddle/pull/78837) | Align some other APIs | 拆自 #78707:对齐 `chunk`(+33/-23,新增分块边界检查)、`expand`(+29/-75,修复 -1 维度推断)、`index`、`_values`、`sparse_coo_tensor`/`sparse_csr_tensor`;`c10_layout_test.cc` +32 行 SparseCooTensorInferSize 案例。 |
| [PFCCLab#9](https://github.com/PFCCLab/PaddleCppAPITest/pull/9) | Add test for defined and reset and Complement tests | +2775/-569 行 17 文件:补 `reset`/`defined` 测试,改写 AbsTest/ArangeTest 等 8 个 op 测试,迭代修补 `compatibility-testing` skill(该 skill 由 @Le-soleile #38 创建);PR 描述提了"SymInt 是否要对齐"的开放问题(后由 #78807 回答)。 |
| [PFCCLab#24](https://github.com/PFCCLab/PaddleCppAPITest/pull/24) | Rewrite test files | +281/-167 行,重写 4 个测试文件(TensorTest / FromBlobTest / ReshapeTest / SumTest),把零散的 `EXPECT_*` 整理为结构化测试组。 |

## 5. 与 PaddleCppAPITest 的对应关系

| 主仓 PR(主题) | 测试仓配套 PR | 测试文件最终路径(经 #58/#59 重组) |
|---|---|---|
| #77270 squeeze/unsqueeze | [#17](https://github.com/PFCCLab/PaddleCppAPITest/pull/17) | `test/ATen/ops/SqueezeTest.cpp`、`test/ATen/ops/UnsqueezeTest.cpp` |
| #77498 TensorAccessor | [#25](https://github.com/PFCCLab/PaddleCppAPITest/pull/25) | `test/ATen/core/TensorAccessorTest.cpp` |
| #77544 flatten/narrow | [#26](https://github.com/PFCCLab/PaddleCppAPITest/pull/26) | `test/ATen/ops/FlattenTest.cpp`、`test/ATen/ops/NarrowTest.cpp` |
| #77614 select/split | (#46 整合多个 op 测试) | `test/ATen/ops/SelectTest.cpp`、`SplitTest.cpp`、`DetachTest.cpp`、`ReciprocalTest.cpp`、`MaskedSelectTest.cpp` |
| #77713 Tensor methods → ops/ | [#24](https://github.com/PFCCLab/PaddleCppAPITest/pull/24) | 重写 4 个测试文件验证搬迁不破坏行为 |
| #78037 综合 compat APIs | [#40](https://github.com/PFCCLab/PaddleCppAPITest/pull/40) | `CoalesceTest.cpp`、`ItemTest.cpp`、`SumTest.cpp`、`TTest.cpp`、`ToTest.cpp`、`ViewAsTest.cpp` |
| #78255 from_blob/pin_memory | (在主仓 `ATen_from_blob_test.cc` 与 `ATen_pin_memory_creation_test.cc` 覆盖) | `test/ATen/ops/FromBlobTest.cpp`(测试仓侧,经 #9 重写) |
| #78809 weak_use_count | (在 #58 mismatch 复现中验证) | `test/ATen/core/TensorUtilTest.cpp` |
| #78826 MaybeResetHolder | (在主仓 `ATen_resize_custom_kernel_test.cc` 覆盖) | (主仓侧专属,测试仓无需对照) |
