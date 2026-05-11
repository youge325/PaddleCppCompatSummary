# 02 · CUDA 与设备

本章覆盖 compat 层中**最有工程挑战**的两类工作:

- **CUDA 流事件与 Context**:`c10::cuda::CUDAStream`、`c10::Event` 与 `at::cuda::CUDAContext{,Light}`、`record_stream`、`getCurrentCUDAStream` ↔ `phi::GPUContext` 双向同步、`thread-local` 语义
- **CUDA 计算与随机数**:`at::cuda::blas::gemm<T>`、`getCurrentCUDABlasHandle`、`at::Generator` + `at::CUDAGeneratorImpl`、`PhiloxCudaState` + `philox::unpack`
- **平台与设备适配**:Windows compat 测试、Mac 平台条件编译、DCU(HIP)native API 替换、XPU device 索引解析

## 子文档

| # | 文件 | 主要内容 |
|---|---|---|
| 01 | [01-CUDA流事件与Context.md](01-CUDA流事件与Context.md) | Stream / Event / CUDAContext{,Light} / record_stream / GPUContext 同步 / thread-local stream |
| 02 | [02-CUDA计算与随机数.md](02-CUDA计算与随机数.md) | CUDABlas / Generator / Philox / DispatchKey for CUDA |
| 03 | [03-平台与设备适配.md](03-平台与设备适配.md) | Windows / Mac / DCU (HIP) / XPU |

## 涉及 PR

**Paddle 主仓**(15 个):#78060、#78070、#78072、#78143、#78549、#78550、#78551、#78553、#78584、#78595、#78632(closed)、#78647、#78652、#78670、#78808、#78902(open)

**PaddleCppAPITest**(3 个):#46(CUDABlas)、#47(RecordStream)、#49(Generator)

## 关键架构决策

1. **CUDA Stream 必须与 `phi::GPUContext` 双向同步**(#78652、#78902):FastDeploy 在 Python 层 `paddle.set_cuda_stream(s)` 之后,C++ 侧通过 `c10::cuda::getCurrentCUDAStream()` 必须看到同一条流;反之亦然。早期 compat 层维护自己的 TLS,导致两侧不同步,引发推理结果错乱。#78652 改为从 `GPUContext` 直接读、回写;#78902 进一步把 TLS 重新引入(避免新线程继承主线程 stream 与 `cudaStreamDestroy` 的所有权混乱)。

2. **`c10::Event` 必须跨平台**(#78553):上游 PyTorch 的 `c10::Event` 是真正的跨设备抽象(CPU/CUDA/MPS/XPU),CPU 路径直接抛 "Backend doesn't support events"。compat 层早期把 `c10::Event` 整个用 `#ifdef PADDLE_WITH_CUDA` 包起来,非 CUDA 构建直接编译失败。#78553 把它解开,加上 placeholder。

3. **`CUDAGuard` 必须无副作用**(#78808):PyTorch 的 `CUDAGuard` 在析构时必须**精确恢复** `original_device()`。compat 层早期用 `paddle::platform::CUDADeviceGuard` 间接实现,会有"如果 set_device 期间又被切了 device"的边界情况。#78808 改为直接调用 `phi::backends::gpu::SetDeviceId`,在 `set_index`/`reset`/`~CUDAGuard` 路径都显式调一次。

4. **CUDA native API 通过宏切换 HIP**(#78595):支持 DCU(海光)设备意味着把 `cudaStream_t` / `cudaEvent_t` / `cudaError_t` / `cudaDeviceSynchronize` 等替换为 `hipStream_t` 等,用 `#ifdef PADDLE_WITH_HIP` 切分,但保持 compat 层接口签名不变。

5. **Linux 上以 ABI 检查兜底**(#78831,在 [`03-接口结构与基础设施/02-CI与编译.md`](../03-接口结构与基础设施/02-CI与编译.md) 中详述):本章节内大量改动都涉及 `paddle/phi/api/include/compat/c10/cuda/` 下的导出符号,Linux ABI checker 保证后续 PR 不会误删这些符号。

按文档顺序阅读即可。
