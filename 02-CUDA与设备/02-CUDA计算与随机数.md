# 02 · CUDA 计算与随机数

> 范围:`at::cuda::blas::gemm<T>` / `getCurrentCUDABlasHandle`、`at::Generator` + `at::CUDAGeneratorImpl`、`PhiloxCudaState` + `philox::unpack`、`CUDAContextLight`、`OpMathType`/`AccumulateType`、`UniqueVoidPtr`、Allocator。

## 1. 主题概述

PyTorch 中,**在 device 上执行 GEMM** 是通过 `at::cuda::blas::gemm<T>(...)` — 内部会:

1. 通过 `at::cuda::getCurrentCUDABlasHandle()` 拿到当前 device 的 `cublasHandle_t`,该 handle 与当前 stream 绑定。
2. 根据 `T`(`float`/`double`/`half`/`bfloat16`)选择 `cublasSgemm` / `Dgemm` / `Hgemm` / `Sgemm` + casting 等。
3. 通过 `AccumulateType<T>` / `OpMathType<T>` 选择中间累加精度(half 输入用 float 累加)。

**device 上的随机数**通过 `at::Generator` + `at::cuda::CUDAGeneratorImpl` 管理,kernel 内部从 `PhiloxCudaState` 取出 4-tuple 种子,用 `at::cuda::philox::unpack` 解包后送进 `curand_*` 或 Paddle 的 `phi::Philox4x32_10` 实现。

这一章节涉及的工作横跨 **6 个核心 PR**(`CUDABlas` + `Generator` + `Philox` + 综合对齐),是 **DeepEP / DeepGEMM / FlashMLA** 等数学计算密集库能编译过的关键依赖。

## 2. PR 汇总表

| PR# | 状态 | 合并日期 | 标题 |
|---:|---|---|---|
| [#78060](https://github.com/PaddlePaddle/Paddle/pull/78060) | MERGED | 2026-03-26 | Add CUDABlas related APIs |
| [#78070](https://github.com/PaddlePaddle/Paddle/pull/78070) | MERGED | 2026-03-22 | Add `Generator` related APIs |
| [#78072](https://github.com/PaddlePaddle/Paddle/pull/78072) | MERGED | 2026-03-23 | Add `Philox` related APIs |
| [PFCCLab#46](https://github.com/PFCCLab/PaddleCppAPITest/pull/46) | MERGED | 2026-03-27 | add CUDABlas related API tests |
| [PFCCLab#49](https://github.com/PFCCLab/PaddleCppAPITest/pull/49) | MERGED | 2026-03-21 | add Generator related API tests |

## 3. 关键 PR 深度分析

### 3.1 `CUDABlas` — 完整的 cuBLAS 兼容层 — #78060(+3394/-380 行,45 个文件)

这是项目中**单 PR 体量最大**的之一(后只被 #78670 与 #78662 超过),原因是引入 cuBLAS 抽象**同时**触发了 `CUDAContext` / `Allocator` / `UniqueVoidPtr` / `Storage` / `Device` / `OpMathType` 等周边类型的大范围补全。

#### 背景

`at::cuda::blas::gemm<T>` 的实现需要:

```
at::Tensor 输入
   ↓ data_ptr<T>(), strides(), sizes()
at::cuda::getCurrentCUDABlasHandle()
   ↓ 拿到 cublasHandle_t,绑定当前 stream
cublasSgemm / Dgemm / Hgemm / SgemmEx / GemmEx
   ↓ 实际计算
at::Tensor 输出
```

PyTorch 在 `ATen/cuda/CUDABlas.{h,cpp}` 中定义 gemm 模板。compat 层需要等价实现,内部走 Paddle 的 cublas wrapper(`phi::backends::dynload::cublas`)。

#### 改动

**1. `ATen/cuda/CUDABlas.{h,cpp}`** (+69 + +201 行)

```cpp
namespace at::cuda::blas {

template <typename T>
void gemm(char transa, char transb,
          int64_t m, int64_t n, int64_t k,
          T alpha,
          const T* a, int64_t lda,
          const T* b, int64_t ldb,
          T beta,
          T* c, int64_t ldc);

// 显式实例化
template void gemm<float>(...);
template void gemm<double>(...);
template void gemm<at::Half>(...);
template void gemm<at::BFloat16>(...);

}  // namespace at::cuda::blas
```

实现内部:

```cpp
template <>
void gemm<float>(...) {
  auto handle = getCurrentCUDABlasHandle();
  PADDLE_RETRY_CUDA_SUCCESS(phi::dynload::cublasSgemm(
      handle, opa, opb, m, n, k,
      &alpha, a, lda, b, ldb, &beta, c, ldc));
}
```

**2. `ATen/cuda/CUDAContextLight.{h,cpp}`**(+130 + +214 行)新建

`CUDAContext.h` 与 `CUDAContextLight.h` 在 PyTorch 中有分工:`CUDAContext.h` 是"重"头(include cusparse、cublasLt 等所有 handle),`CUDAContextLight.h` 只暴露 `getCurrentCUDABlasHandle` / `getCurrentCUDAStream` / `getDeviceProperties` 这些常用接口,避免 include 传染。compat 层照这个分工把它们拆开。

`CUDAContextLight.cpp` 实现:

```cpp
namespace at::cuda {

cublasHandle_t getCurrentCUDABlasHandle() {
  auto* ctx = phi::GetMutableGPUContext();
  return ctx->cublas_handle();  // Paddle 已为每个 GPUContext 维护一个 cublasHandle
}

cusparseHandle_t getCurrentCUDASparseHandle() {
  auto* ctx = phi::GetMutableGPUContext();
  return ctx->cusparse_handle();
}

const cudaDeviceProp& getDeviceProperties(int64_t device) {
  return phi::backends::gpu::GetDeviceProperties(device);
}

}  // namespace at::cuda
```

**3. `ATen/OpMathType.h`**(+69 行)新建

PyTorch 的"运算时累加类型"映射:

```cpp
template<typename T> struct OpMathType { using type = T; };
template<> struct OpMathType<at::Half>     { using type = float; };
template<> struct OpMathType<at::BFloat16> { using type = float; };
```

这是混合精度训练的语义基础 — 输入 fp16,内部累加用 fp32。

**4. `c10/util/UniqueVoidPtr.h`**(+139 行)新建

`c10::UniqueVoidPtr` 是 `DataPtr` 的核心,一个带自定义 deleter 的 `unique_ptr<void>`。在 PyTorch 中由 `c10::DataPtr` 持有,用于让 caching allocator 接管内存生命周期。

**5. `c10/core/Allocator.h`**(+106/-33 行)

补全 `Allocator` 与 `DataPtr` 接口:

```cpp
struct DataPtr {
  UniqueVoidPtr ptr_;
  Device device_;
  
  void* get() const noexcept;
  void* get_context() const noexcept;
  void* mutable_get() const;
  // ... 等约 20 个方法
};

struct Allocator {
  virtual DataPtr allocate(size_t nbytes) const = 0;
  virtual DeleterFnPtr raw_deleter() const = 0;
  // ...
};
```

**6. `c10/core/Device.{h,cpp}`**(+44/-12 + +24/-5 行)

`Device` 与 `DeviceType` 显著扩展:

```cpp
enum class DeviceType : int8_t {
  CPU = 0, CUDA = 1, HIP = 2, MKLDNN = 3, OPENGL = 4, OPENCL = 5,
  IDEEP = 6, XLA = 7, Vulkan = 8, Metal = 9, XPU = 10, MPS = 11,
  Meta = 12, HPU = 13, VE = 14, Lazy = 15, IPU = 16, MTIA = 17,
  PrivateUse1 = 18, COMPILE_TIME_MAX_DEVICE_TYPES,
};
```

`Device::is_cuda()` / `is_xpu()` / `is_privateuseone()` 等谓词,`set_index()`,`std::hash<Device>` 特化。

**7. `c10/core/Storage.h`**(+289/-139 行)

Storage 重写 — 在 #77176 基础上,补全 `set_resizable` / `nbytes` / `mutable_unsafe_get_storage_impl` / `is_pinned` / `unique` 等接口。

**8. `c10/cuda/CUDAFunctions.h`**(+6/-5) — 加 `device_count()` / `is_available()`。

**9. 测试**(8 个 cc 文件,总计 +1100 余行)

- `ATen_CUDABlas_test.cc`(+254) — 14 个 case 覆盖 Sgemm/Dgemm/Hgemm + 不同 trans 组合 + alpha/beta 边界
- `ATen_CUDAContext_test.cc`(+241) — handle 一致性、device properties 查询、stream/handle 绑定
- `c10_Device_test.cc`(+124) — Device 完整覆盖
- `c10_ScalarType_test.cc`(+76) — ScalarType 与 cuBLAS data type 映射
- `c10_storage_test.cc`(+795/-93) — 大重写,覆盖 30+ 新 Storage case
- `cuda_test_utils.h`(+63 行) — 测试侧 helper(后续 #78580 中删除,内联到各测试)

#### 效果

DeepEP / DeepGEMM 的 cuBLAS 调用全部可编译;测试仓 [#46](https://github.com/PFCCLab/PaddleCppAPITest/pull/46) 的 100% MATCH 表中 `CUDABlasTest` MATCH(`paddle_CUDABlasTest` vs `torch_CUDABlasTest`)。

### 3.2 `Generator` + `CUDAGeneratorImpl` — #78070(+3743 行,13 个文件)

#### 背景

PyTorch 的随机数体系:

```
at::Generator (持有 GeneratorImpl 的 intrusive_ptr)
├── at::CPUGeneratorImpl       (CPU 路径,内部用 mt19937)
├── at::cuda::CUDAGeneratorImpl (CUDA 路径,内部用 PhiloxCudaState)
```

用户拿到 `Generator`,通过 `at::manual_seed(seed)` / `at::seed()` 设置种子,然后 `at::randn(sizes, gen)` 等 op 接收 `Generator` 参数。

`at::get_generator_or_default<at::CUDAGeneratorImpl>(gen, default_gen)` 是 op 内部的标准用法 — 如果用户传入的是有效 Generator 就用它,否则用默认 Generator。

#### 改动

PR 描述附带了一张 5255×5000 的 Mermaid 图,详细展示了 `Generator` ↔ `CUDAGeneratorImpl` ↔ `Philox` 的依赖关系。

**1. `ATen/core/Generator.h`**(+196 行)新建

```cpp
namespace at {

class Generator final {
 public:
  Generator() = default;
  Generator(c10::intrusive_ptr<c10::GeneratorImpl> gen_impl);
  
  bool defined() const;
  c10::GeneratorImpl* unsafeGetGeneratorImpl() const;
  c10::GeneratorImpl* operator->() const;
  
  Generator clone() const;
  void set_current_seed(uint64_t seed);
  uint64_t current_seed() const;
  uint64_t seed();
  
  void set_state(const c10::TensorImpl& new_state);
  c10::intrusive_ptr<c10::TensorImpl> get_state() const;
  
  Device device() const;
  
  template <typename T>
  T* get() const { return static_cast<T*>(unsafeGetGeneratorImpl()); }
  
 private:
  c10::intrusive_ptr<c10::GeneratorImpl> impl_;
};

}  // namespace at
```

**2. `ATen/cuda/CUDAGeneratorImpl.h`**(+197 行)新建

```cpp
namespace at::cuda {

class CUDAGeneratorImpl : public c10::GeneratorImpl {
 public:
  CUDAGeneratorImpl(DeviceIndex device_index = -1);
  
  void set_current_seed(uint64_t seed) override;
  uint64_t current_seed() const override;
  uint64_t seed() override;
  
  PhiloxCudaState philox_cuda_state(uint64_t increment);
  
  static DeviceType device_type() { return DeviceType::CUDA; }
  
 private:
  uint64_t seed_;
  uint64_t philox_offset_per_thread_;
};

namespace detail {

const Generator& getDefaultCUDAGenerator(DeviceIndex device_index = -1);
Generator createCUDAGenerator(DeviceIndex device_index = -1);

}  // namespace detail
}  // namespace at::cuda
```

`getDefaultCUDAGenerator(device_index)` 是一个 process-wide 单例,在 op 内部最常用。

**3. `c10/core/GeneratorImpl.h`**(+147 行)新建

抽象基类:

```cpp
namespace c10 {

struct GeneratorImpl : public c10::intrusive_ptr_target {
  virtual ~GeneratorImpl() = default;
  
  virtual void set_current_seed(uint64_t seed) = 0;
  virtual uint64_t current_seed() const = 0;
  virtual uint64_t seed() = 0;
  virtual Device device() const = 0;
};

}  // namespace c10
```

**4. `c10/core/DispatchKey.h`**(+598 行)、`DispatchKeySet.h`(+735 行)、`c10/util/intrusive_ptr.h`(+442 行)新建

`Generator` 通过 `intrusive_ptr<GeneratorImpl>` 持有 — 这迫使 compat 层把完整的 `intrusive_ptr` 和 `DispatchKey{,Set}` 同时落地。详见 [01-基础类型与Tensor/01-基础类型对齐.md](../01-基础类型与Tensor/01-基础类型对齐.md) 第 3.5、3.6 节。

**5. `paddle/phi/core/generator.cc`**(+5 行)

Paddle 内核侧暴露 `phi::Generator::set_current_seed(uint64_t)` 等接口,供 compat 层桥接。

**6. 测试**(5 个 cc 文件,总计 +1500 余行)

- `c10_DispatchKeySet_test.cc`(+439)、`c10_DispatchKey_test.cc`(+248) — DispatchKey 完整覆盖
- `c10_generator_impl_test.cc`(+189) — Generator/Impl 基础
- `c10_intrusive_ptr_lifecycle_test.cc`(+183) — intrusive_ptr 引用计数生命周期
- `cuda_generator_test.cc`(+358) — `getDefaultCUDAGenerator` 单例性、`set_current_seed` 后采样可复现

#### 效果

`at::randn(sizes, at::cuda::detail::getDefaultCUDAGenerator())` 一类常见 op 用法可编译且行为对齐。测试仓 [#49](https://github.com/PFCCLab/PaddleCppAPITest/pull/49) 配套 GeneratorTest +279 行,与 libtorch `diff` 一致。

### 3.3 `Philox` + `PhiloxCudaState` — #78072(+119/-61 行,5 个文件)

#### 背景

CUDA kernel 内部生成 stateless 随机数的标准方式是 **Philox 计数器随机数生成器**(Salmon et al. 2011)。每个 kernel 实例从 `PhiloxCudaState` 拿出种子和 counter,用 4×32-bit Philox 算法直接计算第 N 个随机数,无需全局状态。

PyTorch 的接口:

```cpp
__global__ void mykernel(at::PhiloxCudaState philox) {
  auto seeds = at::cuda::philox::unpack(philox);  // -> std::tuple<uint64_t, uint64_t>
  curandStatePhilox4_32_10_t state;
  curand_init(std::get<0>(seeds), thread_id, std::get<1>(seeds), &state);
  float4 r = curand_uniform4(&state);
}
```

`PhiloxCudaState` 是一个 16 字节的轻量结构,可以直接传到 kernel。

#### 历史问题

PR 描述提到一个有趣的事实:**compat 层早就有一个 `PhiloxCudaState.h`,但位置在 `compat/c10/cuda/`**(六个月前为 Paddle 内部 DeepEP fork 加的,带一个 `_PD_Internal_GetDefaultPhiloxCudaState` 函数)。这个旧文件**与上游 PyTorch 路径不一致** — 上游放在 `ATen/cuda/PhiloxCudaState.h`。

`#78072` 的工作:

1. 在 `ATen/cuda/PhiloxCudaState.h` 提供与上游同名同路径的版本。
2. 旧的 `c10/cuda/PhiloxCudaState.h` 保留(暂时不删,避免 Paddle 内部用户编译失败)。
3. 新建 `ATen/cuda/PhiloxUtils.cuh`(+45 行),提供 `at::cuda::philox::unpack(state)` device-callable 函数。

PR 描述里作者主动提了这个问题:"目前同时存在 `compat/c10/cuda/PhiloxCudaState.h` 和 `compat/ATen/cuda/PhiloxCudaState.h`,前者是六个月前加的兼容接口…后者对齐 libtorch 的路径,**是否要删掉其中一个文件**?" — 后续这两个文件会在 #78580 / #78549 中清理。

#### 改动

- **`ATen/cuda/PhiloxCudaState.h`**(+11/-15) — 改写为与上游一致的纯 struct + helper:

  ```cpp
  struct PhiloxCudaState {
    uint64_t* seed_;
    uint64_t* offset_;
    bool captured_;
    uint64_t offset_intragraph_;
  };
  ```

- **`ATen/cuda/PhiloxUtils.cuh`**(+45 行)— device-callable `unpack`:

  ```cpp
  __host__ __device__ std::tuple<uint64_t, uint64_t>
  unpack(at::PhiloxCudaState arg) {
    if (arg.captured_) {
      return std::make_tuple(static_cast<uint64_t>(*arg.seed_),
                              static_cast<uint64_t>(*arg.offset_));
    } else {
      return std::make_tuple(static_cast<uint64_t>(arg.offset_intragraph_),
                              static_cast<uint64_t>(0));
    }
  }
  ```

- **`ATen/cuda/CUDAGeneratorImpl.h`**(+5/-46) — 旧版的 inline `philox_cuda_state` 实现搬到 cpp / 用新版替换。

- **`ATen_philox_test.cc`**(+57 行)— 6 个 case 覆盖 `unpack` 在 host / device 路径的一致性、captured / non-captured 两种模式。

#### 效果

DeepEP 的 dropout / random fused kernel 可以直接使用 `at::cuda::philox::unpack`,与 PyTorch 等价。

## 4. 其他相关 PR 简注

(本主题大多与第 1 章节"CUDA 流事件与 Context"耦合,详见该章节的"其他 PR 简注"。)

## 5. 与 PaddleCppAPITest 的对应关系

| 主仓 PR | 测试仓 PR | 测试文件最终路径(经 #58 / #59 重组) |
|---|---|---|
| #78060 CUDABlas | [#46](https://github.com/PFCCLab/PaddleCppAPITest/pull/46) | `test/ATen/cuda/CUDABlasTest.cpp`(+575 行) |
| #78070 Generator | [#49](https://github.com/PFCCLab/PaddleCppAPITest/pull/49) | `test/ATen/cuda/GeneratorTest.cpp`(+279 行,#59 中更新 +68 行) |
| #78072 Philox | (在 #46 中 CUDABlasTest 内间接覆盖) | `doc/ATen/cuda/philox.md`(由 #46 引入 +91 行,后由 #58 移到 `doc/ATen/cuda/` 子目录) |
| #78070 DispatchKey/Set 顺带落地 | [#59](https://github.com/PFCCLab/PaddleCppAPITest/pull/59) | `test/c10/core/DispatchKeyTest.cpp`(+370)、`DispatchKeySetTest.cpp`(+363) |

注:测试仓 #49 PR 描述里提到"没对齐的地方和 Device 别名有关,等晓春修复" — 这里"晓春"指的是同期营员 [@Le-soleile](https://github.com/Le-soleile),他负责 Device 别名相关工作。
