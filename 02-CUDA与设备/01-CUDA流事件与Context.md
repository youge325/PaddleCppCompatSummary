# 01 · CUDA 流事件与 Context

> 范围:`c10::cuda::CUDAStream`、`c10::Event`、`c10::cuda::CUDAGuard{,OptionalCUDAGuard}`、`at::cuda::CUDAContext{,Light}`、`record_stream`、与 `phi::GPUContext` 的双向同步、`thread-local` 语义。

## 1. 主题概述

PyTorch 在 `c10/cuda/` 与 `ATen/cuda/` 下定义了一组关键 CUDA 抽象,**几乎所有 GPU C++ 库都依赖它们**:

- `c10::cuda::CUDAStream` — CUDA stream 的轻量包装,支持 `pack3()` / `unpack3()` 跨进程传递、`getStreamFromPool()` 池化分配、`getStreamFromExternal()` 接管外部 stream。
- `c10::cuda::getCurrentCUDAStream()` / `setCurrentCUDAStream()` — 线程局部的"当前流"语义。
- `c10::Event` — 跨设备的事件抽象(CPU 路径占位、CUDA 路径走 `cudaEvent_t`)。
- `c10::cuda::CUDAGuard` / `OptionalCUDAGuard` — RAII 设备切换 guard,析构时恢复原 device。
- `at::cuda::CUDAContext` / `CUDAContextLight` — 当前 device 查询、`cudaStream_t` / `cusparseHandle_t` / `cuBlasHandle_t` 获取。
- `tensor.record_stream(stream)` — 通知内存分配器:这个 stream 上还有未完成的 op,不要早释放该 Tensor 的 storage。

Paddle 对应的抽象是 `phi::backends::gpu::*`(`SetDeviceId`、`GpuDeviceGuard`)和 `phi::GPUContext`(全局 device context,内部有当前 stream)。compat 层的工作:在 `paddle/phi/api/include/compat/c10/cuda/` 与 `paddle/phi/api/include/compat/ATen/cuda/` 下提供一套**与上游同名、同语义**的接口,内部走 Paddle 后端。

**最大的工程挑战是 "stream 状态的双向同步"** — Python 用户在 `paddle.set_cuda_stream(s)` 之后,从 C++ 通过 `c10::cuda::getCurrentCUDAStream()` 必须立刻看到这条流;反之 C++ 内做 `c10::cuda::setCurrentCUDAStream(s)` 后,Python 侧的 `paddle.cuda.current_stream()` 也必须返回 `s`。这层同步在本章涉及 **3 个 PR**(#78652、#78808、#78902),其中 #78902 还在 review。

## 2. PR 汇总表

| PR# | 状态 | 合并日期 | 标题 |
|---:|---|---|---|
| [#78060](https://github.com/PaddlePaddle/Paddle/pull/78060) | MERGED | 2026-03-26 | Add CUDABlas related APIs(其中含 `CUDAContext{,Light}` 起步) |
| [#78143](https://github.com/PaddlePaddle/Paddle/pull/78143) | MERGED | 2026-03-21 | Complement `record_stream` and related Stream/Event implementation |
| [#78549](https://github.com/PaddlePaddle/Paddle/pull/78549) | MERGED | 2026-05-01 | Remove deepep legacy APIs(拆自 #78484) |
| [#78550](https://github.com/PaddlePaddle/Paddle/pull/78550) | MERGED | 2026-04-03 | Fix flashmla compile(拆自 #78484,含 Stream 行为修复) |
| [#78553](https://github.com/PaddlePaddle/Paddle/pull/78553) | MERGED | 2026-04-02 | Align event api(拆自 #78484) |
| [#78555](https://github.com/PaddlePaddle/Paddle/pull/78555) | MERGED | 2026-04-03 | Align misc apis(拆自 #78484,含 CUDAStream 扩展) |
| [#78584](https://github.com/PaddlePaddle/Paddle/pull/78584) | MERGED | 2026-04-04 | Fix `CUDAContext.h` to align with Pytorch |
| [#78652](https://github.com/PaddlePaddle/Paddle/pull/78652) | MERGED | 2026-04-14 | Sync c10 CUDA stream state with Paddle's GPUContext stream |
| [#78808](https://github.com/PaddlePaddle/Paddle/pull/78808) | MERGED | 2026-05-01 | Align cuda compat(`torch::cuda::synchronize` + `CUDAGuard`) |
| [#78902](https://github.com/PaddlePaddle/Paddle/pull/78902) | OPEN | — | use thread-local stream in `c10::cuda::getCurrentCUDAStream` |
| [PFCCLab#47](https://github.com/PFCCLab/PaddleCppAPITest/pull/47) | MERGED | 2026-03-21 | add `RecordStreamTest.cpp` |

## 3. 关键 PR 深度分析

### 3.1 `record_stream` + Stream / Event 起步实现 — #78143(+903/-45 行,16 个文件)

#### 背景

PyTorch 中的 `tensor.record_stream(stream)` 在 CUDA caching allocator 体系下有具体含义:**告诉分配器,这个 Tensor 的 storage 在 `stream` 上还有未完成的访问,允许该 Tensor 被销毁但不立即归还 storage**。

DeepEP(高性能 MoE 通信库)在 `csrc/deep_ep.cpp` 中重度使用 `record_stream` 跨 stream 协调通信与计算,触发了 compat 层第一个 stream 相关 PR。

PR 描述附带了 Mermaid 流程图(8192×4645 像素),拆解 `record_stream` 在 PyTorch 中的调用路径:`Tensor::record_stream` → `Storage::record_stream` → `StorageImpl::record_stream` → caching allocator hook。compat 层不实现 caching allocator(Paddle 内部有自己的 `phi::Allocator`),所以 `record_stream` 在 compat 层退化为一个 no-op,但**接口必须存在**才能让 DeepEP 编译过。

#### 改动

- **`ops/record_stream.h`**(+70 行) — `record_stream(Tensor, Stream)` 与 `Tensor::record_stream(Stream)`,内部调用 `phi::stream::Stream::RecordEvent` 做最小同步,在 CUDA caching allocator 不存在时是 no-op,在 CUDA 存在时记录一个 event。
- **`c10/core/Stream.h`**(+108 行)+ **`Stream.cpp`**(+91 行) — `c10::Stream` 类:`device()` / `device_index()` / `stream()` / `id()`,`operator==/!=`,`pack3()`/`unpack3()` 序列化。
- **`c10/cuda/CUDAStream.h`**(+199/-16 行) — 完整的 `c10::cuda::CUDAStream`,包含 `Stream` 父类的所有方法 + CUDA 专属的 `query()` / `synchronize()` / `priority()` / `cudaStream_t stream()`。
- **`c10/core/Event.h`**(+18/-18 行) — `c10::Event` 类雏形(后续 #78553 完整对齐)。
- **`ATen/cuda/CUDAEvent.h`**(+148 行) — `at::cuda::CUDAEvent`:包装 `cudaEvent_t`,提供 `record(Stream)` / `block(Stream)` / `query()` / `elapsed_time(other)` / `synchronize()`。
- **`c10/core/DeviceType.h`**(+13 行) — 为 stream/event 路径补 `DeviceType::HIP` / `XPU` 占位。

#### 测试

- `ATen_record_stream_test.cc`(+77 行) — 10 个 case,覆盖跨 stream `record_stream`、deleter 时序、storage 生命周期。
- `c10_Stream_test.cc`(+132 行) — 22 个 case,覆盖 `Stream` 构造、`pack3` 往返、device_index 范围、operator 比较。
- 测试仓 [#47](https://github.com/PFCCLab/PaddleCppAPITest/pull/47) `RecordStreamTest.cpp`(+247 行)对照 libtorch 跑同样代码,`diff` 为空。

### 3.2 `c10::Event` 完整跨平台对齐 — #78553(+440/-111 行,5 个文件)

#### 背景:`#78484` 的拆分

`#78484` 是一个综合性的对齐 PR,涉及 `Event` / `Stream` / `Allocator` / `Device` / `ScalarType` / `CUDAGuard` 等多个主题,体量 +900/-513。导师 review 时建议拆成 6 个独立 PR(#78549、#78550、#78551、#78552、#78553、#78554、#78555),便于各自 review 与回滚。`#78484` 因此 CLOSED。

`#78553` 是其中专门处理 `c10::Event` 的拆分 PR。

#### 历史问题

`c10::Event` 之前的状态:

1. 被 `#ifdef PADDLE_WITH_CUDA` 包起来,**非 CUDA 构建直接编译失败**(`ATen/cuda/CUDAEvent.h` 在 CPU 构建下也试图 include `c10::Event`)。
2. 缺少 `EventFlag` 枚举(`PYTORCH_DEFAULT` / `BACKEND_DEFAULT` / `INVALID`),构造函数只支持 `Event(DeviceType)` 一种签名。
3. 没有移动语义(默认拷贝是危险的,因为内部持有 `cudaEvent_t` 句柄)。
4. CPU 路径异常行为不一致 — PyTorch 在 CPU device 上调用 `record()` 应抛 "Backend doesn't support events",compat 层之前直接 segfault。

#### 改动

`c10/core/Event.h`(+293/-109 行)— 把 `Event` 重构为跨平台类:

```cpp
namespace c10 {

enum class EventFlag {
  PYTORCH_DEFAULT,
  BACKEND_DEFAULT,
  INVALID,
};

class Event final {
 public:
  Event(DeviceType device_type,
        EventFlag flag = EventFlag::PYTORCH_DEFAULT);
  ~Event() = default;

  // 禁拷贝、允许 move
  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;
  Event(Event&&) = default;
  Event& operator=(Event&&) = default;

  // 属性
  Device device() const noexcept;
  DeviceType device_type() const noexcept;
  DeviceIndex device_index() const noexcept;
  EventFlag flag() const noexcept;
  bool was_marked_for_recording() const noexcept;

  // 记录与同步
  void record(const Stream& stream);
  void recordOnce(const Stream& stream);
  void block(const Stream& stream) const;
  bool query() const;
  void synchronize() const;
  double elapsedTime(const Event& event) const;
  void* eventId() const;
};

}  // namespace c10
```

CUDA 构建下,各方法 dispatch 到 `cudaEvent_*`(或 `hipEvent_*`,DCU);CPU 构建下,所有方法直接抛 `STD_TORCH_CHECK(false, "Backend doesn't support events")`。

`c10/cuda/CUDAStream.h`(+12/-2)— 在 CUDA 构建下为 `Event` 提供 `block(CUDAStream)` 与 `record(CUDAStream)` 的重载。

#### 测试

`c10_Event_test.cc`(+120 行)— 18 个 case,覆盖 `EventFlag` 三种枚举、move-constructed Event 行为、`elapsed_time` 计算、CPU 路径异常抛出。

### 3.3 `CUDAContext.h` 自包含 — #78584(+211/-140 行,6 个文件)

#### 问题

DeepEP 在 `csrc/deep_ep.cpp` 中只 `#include <ATen/cuda/CUDAContext.h>`,假定该头文件能提供 `at::cuda::getCurrentCUDABlasHandle()` 等所需的全部声明。但 compat 层早期版本的 `ATen/cuda/CUDAContext.h` 内部声明 `getCurrentCUDAStream()` 时没 include `c10/cuda/CUDAStream.h`,DeepEP 编译直接报"undeclared identifier `c10::cuda::CUDAStream`"。

#### 改动

- **`ATen/cuda/CUDAContext.h`**(+2 行) — 在文件顶部 `#include <c10/cuda/CUDAStream.h>`,让该头自包含。
- **`compat/CMakeLists.txt`**(+2/-1) — 将 `c10/cuda/CUDAStream.cpp` 加入编译,因为 #78584 一并把 `CUDAStream.h` 中大量 inline 实现拆到 cpp。
- **`c10/cuda/CUDAStream.cpp`**(+182 行) — 新文件,承载 `CUDAStream.h` 中拆出的实现。
- **`c10/cuda/CUDAStream.h`**(+20/-138 行) — 净减 118 行,只保留声明 + 必要内联。
- **`c10/cuda/CUDAException.h`**(+4 行) — 补 `CUDAException` 的 `__attribute__((visibility("default")))` 之类的导出注解。

#### 效果

PR 描述里附了 Mermaid 流程图(8192×4375),展示 `getStreamFromPool` 在 PaddleCppAPITest 中**与 libtorch 完全对齐**的调用路径。

测试仓 [#59](https://github.com/PFCCLab/PaddleCppAPITest/pull/59) 把 `cuda_stream.md` / `cuda_context_light.md` 文档更新到 +111/-48,详细描述这一同步逻辑。

### 3.4 ★ CUDA Stream ↔ GPUContext 双向同步 — #78652(+37/-46 行,3 个文件)

这个 PR 体量很小但是**架构上最关键的一次修正**,触发它的 issue 是 [FastDeploy#7344](https://github.com/PaddlePaddle/FastDeploy/pull/7344)。

#### 问题

之前 compat 层用自己维护的 thread-local storage 记录"当前 CUDA stream":

```cpp
// 旧实现(简化)
namespace c10::cuda {
thread_local CUDAStream current_stream = default_stream();

CUDAStream getCurrentCUDAStream(DeviceIndex device_index) {
  return current_stream;
}

void setCurrentCUDAStream(CUDAStream stream) {
  current_stream = stream;
}
}  // namespace c10::cuda
```

FastDeploy 在 Python 层调用 `paddle.set_cuda_stream(s)`,后端只会改 `phi::GPUContext` 的 stream,**compat 层的 TLS 一无所知**。下游 C++ 库再调 `c10::cuda::getCurrentCUDAStream()` 拿到的还是默认 stream — Python/C++ 视角不一致,推理结果错乱。

#### 改动

```cpp
// 新实现:从 GPUContext 读、回写
CUDAStream getCurrentCUDAStream(DeviceIndex device_index) {
  // 优先与 Paddle 当前流保持一致
  cudaStream_t paddle_stream = phi::GetMutableGPUContext()->stream();
  return CUDAStream::wrap_external(paddle_stream, device_index);
}

void setCurrentCUDAStream(CUDAStream stream) {
  // 更新 compat 状态时,同步回 GPUContext
  cudaStream_t s = stream.stream();
  
  // 使用 TLS 持有的 phi::CUDAStream wrapper 回写 GPUContext,
  // 避免直接篡改外部 stream 对象
  thread_local std::unique_ptr<phi::CUDAStream> tls_wrapper;
  tls_wrapper = std::make_unique<phi::CUDAStream>(s);
  phi::GetMutableGPUContext()->SetStream(tls_wrapper.get());
}
```

#### 测试

`c10_Stream_test.cc`(+12/-23) — 新加两个 case:

1. `setCurrentCUDAStream` 后 compat 与 Paddle 看到同一条流
2. 仅 Paddle 侧 `SetStream(s)`,`getCurrentCUDAStream()` 也能返回正确流

#### 副作用与后续(→ #78902)

这个 PR 解决了"双向同步",但**牺牲了 thread-local 语义**:

- 新线程会**继承主线程的 current stream**(因为 `GPUContext` 是全局对象),违反 PyTorch 的"每个线程独立的 current stream"约定,可能引发 multi-threaded inference 中 stream 之间的串扰。
- `setCurrentCUDAStream` 内部用了 `phi::GPUContext::SetStream`,而后者会 `cudaStreamDestroy` 旧 stream。当旧 stream 来自 compat 的 `getStreamFromPool()`(compat 持有所有权)时,`cudaStreamDestroy` 会导致 stream handle 失效,后续重复使用即 SegFault。

这就是 **#78902 OPEN 状态**的原因 — 见 3.6 节。

### 3.5 `torch::cuda::synchronize` 与 `CUDAGuard` 重构 — #78808(+209/-37 行,4 个文件)

#### 背景:多卡场景下的 device leak

`#78707` 是一个综合 PR,提出了 `ScalarType` 对齐 + `weak_use_count` 对齐 + `torch::cuda::synchronize` + `CUDAGuard` 修复。被拆为 4 个独立 PR(#78806、#78807、#78808、#78809、#78837),`#78808` 是其中的"cuda compat"分支。

问题场景:

```cpp
// 用户代码
c10::cuda::set_device(0);
{
  c10::cuda::CUDAGuard g(1);
  torch::cuda::synchronize(2);   // 同步 device 2
  // 期望:析构 g 后回到 device 0
}
// Bug: 实际在 device 2 上,因为 synchronize 隐式改了 device
```

`torch::cuda::synchronize(other_device)` 在 PyTorch 中**必须**先 push device 0 的 guard,切到 `other_device` 调 `cudaDeviceSynchronize`,然后 pop guard 回 device 0。compat 层早期实现直接 `cudaSetDevice(other) + cudaDeviceSynchronize() + cudaSetDevice(prev)`,如果 `prev` 在并发线程被改,就漏了。

#### 改动

**1. `torch/csrc/api/include/torch/cuda.cpp`**(+20/-15) — 重写 `synchronize`:

```cpp
void synchronize(int64_t device_index) {
  if (device_index == -1) {
    // 同步当前 device
    c10::cuda::device_synchronize();
    return;
  }
  
  // 使用 CUDAGuard 保证不污染当前 device
  c10::cuda::CUDAGuard guard(static_cast<DeviceIndex>(device_index));
  c10::cuda::device_synchronize();
  // guard 析构,回到 original device
}
```

**2. `c10/cuda/CUDAGuard.h`**(+31/-18) — 重构 `CUDAGuard`:

```cpp
class CUDAGuard {
 public:
  explicit CUDAGuard(DeviceIndex device_index) {
    original_device_ = phi::backends::gpu::GetCurrentDeviceId();
    phi::backends::gpu::SetDeviceId(device_index);
  }
  
  ~CUDAGuard() {
    phi::backends::gpu::SetDeviceId(original_device_);
  }
  
  void set_device(DeviceIndex device_index) {
    phi::backends::gpu::SetDeviceId(device_index);
  }
  
  void set_index(DeviceIndex device_index) {
    set_device(device_index);
  }
  
  DeviceIndex original_device() const { return original_device_; }
  DeviceIndex current_device() const {
    return phi::backends::gpu::GetCurrentDeviceId();
  }
  
 private:
  DeviceIndex original_device_;
};
```

直接调 `phi::backends::gpu::SetDeviceId`,**移除对 `paddle::platform::CUDADeviceGuard` 的依赖**(后者在某些路径下会自己管理一个 device stack,与 `CUDAGuard` 期望的 RAII 行为有冲突)。

**3. `torch/csrc/api/include/torch/cuda.h`**(+4/-4) — 把 `synchronize`、`device_count`、`is_available` 都加 `PADDLE_API` 注解,解决 Windows 链接错误。

**4. CPU-only 构建条件编译** — 用 `#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)` 包裹,保证 CPU-only 编译过。

#### 测试

`ATen_CUDAContext_test.cc`(+154 行)— 4 组测试:

- `torch::cuda::synchronize(other_device)` 不泄漏当前 device
- `CUDAGuard` 多次 `set_device` 后析构正确回到 original
- `OptionalCUDAGuard::reset()` 正确清理
- 多卡 stress test:并发 100 次 `synchronize`/`CUDAGuard` 不串扰

### 3.6 ★ 把 thread-local 找回来 — #78902(OPEN,+307/-21 行,3 个文件)

#### 问题

如 3.4 节所述,#78652 把所有 stream 状态搬到 `GPUContext`,丢了 thread-local 语义。具体两个 bug:

1. **新线程继承串扰**:线程 A `setCurrentCUDAStream(s1)` 之后启动线程 B,B 调 `getCurrentCUDAStream()` 拿到的是 `s1` 而不是 `default_stream`。PyTorch 语义是每个线程独立。
2. **stream destroy SegFault**:`setCurrentCUDAStream(stream)` 内部用 `GPUContext::SetStream`,后者会 `cudaStreamDestroy` 旧 stream。如果旧 stream 来自 `getStreamFromPool()`(compat 持有所有权),destroy 后 compat 池里的句柄失效,后续重用即崩溃。

#### 改动思路

```cpp
// CUDAStream.cpp(+53/-19)
namespace c10::cuda {

// 重新引入 thread-local current stream
thread_local std::optional<cudaStream_t> tls_current_stream;

CUDAStream getCurrentCUDAStream(DeviceIndex device_index) {
  if (tls_current_stream.has_value()) {
    return CUDAStream::wrap(*tls_current_stream, device_index);
  }
  // fallback to GPUContext stream(保持与 Paddle Python 一致)
  return CUDAStream::wrap(phi::GetMutableGPUContext()->stream(), device_index);
}

void setCurrentCUDAStream(CUDAStream stream) {
  tls_current_stream = stream.stream();
  // 不再 destroy 旧 stream
  // GPUContext 上不再同步写,避免 destroy 问题
}
}  // namespace c10::cuda
```

这样新线程 `tls_current_stream` 为空,fallback 到 `GPUContext` 的 default stream(其语义与 PyTorch 在新线程上读到 default stream 一致)。Paddle Python 层做 `paddle.set_cuda_stream(s)` 改的是 `GPUContext`,新线程依然能看到。同时**`setCurrentCUDAStream` 不再触碰 GPUContext**,杜绝 destroy 问题。

#### 测试

`c10_Stream_test.cc`(+254 行) — 大批量回归:

- 多线程并发 `setCurrentCUDAStream` / `getCurrentCUDAStream`,每个线程互不影响
- `setCurrentCUDAStream(pool_stream)` 后,pool stream 在重置后仍可被 `getStreamFromPool()` 重发
- 死锁回归:并发 `synchronize` + `setCurrentCUDAStream`,无 hang

#### 状态

PR 描述里明确说明了 #78652 引入的两个回归,并阐述这个 PR 的修复方向。截至 2026-05-11 仍在 review,等待导师 [@BingooYang](https://github.com/BingooYang) 确认。

## 4. 其他 PR 简注

| PR# | 标题 | 简注 |
|---:|---|---|
| [#78060](https://github.com/PaddlePaddle/Paddle/pull/78060) | Add CUDABlas related APIs | 详见 [`02-CUDA计算与随机数.md`](02-CUDA计算与随机数.md);其中含 `CUDAContextLight.{h,cpp}` 新建,与本章 CUDAContext 重构相关。 |
| [#78549](https://github.com/PaddlePaddle/Paddle/pull/78549) | Remove deepep legacy APIs | 拆自 #78484:移除 `record_stream`、`Event`、`Stream` 中曾经为 DeepEP 兼容旧接口加的临时 forwarder(DeepEP 已经迁移完成,可以清理)。 |
| [#78550](https://github.com/PaddlePaddle/Paddle/pull/78550) | Fix flashmla compile | 拆自 #78484:`torch/cuda.h` 不再把 `is_available` 重复导出到 `at::cuda`,统一通过 `torch::cuda::is_available()` 入口;`Stream` 比较运算符语义对齐;`native_handle()` 在 CPU 设备上抛 `NotImplementedError`。 |
| [#78551](https://github.com/PaddlePaddle/Paddle/pull/78551) | Align device related APIs | 拆自 #78484:`Device.h`/`DeviceType.h` 增强(详见 [`01-基础类型与Tensor/02-Tensor接口与操作.md`](../01-基础类型与Tensor/02-Tensor接口与操作.md) 中"其他 PR 简注"段),其中包含 `is_xpu`/`is_ipu`/`is_mps`/`is_privateuseone`,与本章 CUDA 状态查询相邻。 |
| [#78555](https://github.com/PaddlePaddle/Paddle/pull/78555) | Align misc apis | 拆自 #78484:`CUDAStream.h` 新加 `priority_range`/`pack3`/`unpack3`/`getStreamFromExternal`/`query`/`synchronize`/`hash<CUDAStream>`;新增 `torch/version.h`(供第三方库检测 compat 层版本)。 |

## 5. 与 PaddleCppAPITest 的对应关系

| 主仓 PR | 测试仓 PR | 测试文件最终路径 |
|---|---|---|
| #78143 record_stream | [#47](https://github.com/PFCCLab/PaddleCppAPITest/pull/47) | `test/c10/cuda/RecordStreamTest.cpp` |
| #78553 c10::Event | (在 #58 mismatch 复现中验证) | `test/c10/core/EventCompatTest.cpp`(经 #59 重组) |
| #78584 CUDAContext 自包含 | [#59](https://github.com/PFCCLab/PaddleCppAPITest/pull/59) | `doc/c10/cuda/cuda_stream.md`、`doc/ATen/cuda/cuda_context_light.md` |
| #78652 GPUContext 同步 | (#47 RecordStreamTest 间接覆盖) | — |
| #78808 CUDAGuard + synchronize | (在 #58 mismatch 复现中验证) | `test/ATen/cuda/CUDAContextTest.cpp` |
| #78902 thread-local stream(OPEN) | [#60](https://github.com/PFCCLab/PaddleCppAPITest/pull/60) | `test/c10/cuda/CUDAStreamDeadlockAlignmentTest.cpp`(+206 行,专门覆盖死锁与多线程串扰回归) |
