# 护航计划 Paddle C++ API 生态兼容建设 — 工作总结

> 营员:[youge325](https://github.com/youge325) · 项目:第十期飞桨黑客松护航计划集训营(提前批)项目四 · 导师:[@BingooYang](https://github.com/BingooYang) · 集训期:**2026-01-05 ~ 2026-04**(实际工作延续至 2026-05-11)

## 项目目标一句话

在 Paddle 仓库的 `paddle/phi/api/include/compat/` 下构建一套 **libtorch C++ API 兼容层**(ATen / c10 / torch 命名空间齐备),使现有基于 PyTorch C++ API 的代码(FastDeploy、DeepEP、DeepGEMM、FlashMLA、hybrid_ep、paddlecodec 等)能在最小改动下编译并链接到 Paddle,同时在配套仓库 [PFCCLab/PaddleCppAPITest](https://github.com/PFCCLab/PaddleCppAPITest) 中以 GoogleTest **逐 API 对齐 PyTorch 行为**、补齐文档、记录 mismatch。

## 成果速览(2026-05-11)

| 仓库 | 提交 | 合入 | OPEN | CLOSED | 主要贡献 |
|---|---:|---:|---:|---:|---|
| [PaddlePaddle/Paddle](https://github.com/PaddlePaddle/Paddle) | 58 | 50 | 3 | 5 | compat 层核心 ATen/c10/torch 接口(新增 89 个文件)、CUDA 流/事件/Context、CUDABlas、Generator/Philox、Sparse、DCU/XPU/Windows 适配、Linux ABI 兼容性 CI |
| [PFCCLab/PaddleCppAPITest](https://github.com/PFCCLab/PaddleCppAPITest) | 22 | 22 | 0 | 0 | 全套 GoogleTest 单测(覆盖 ATen / c10 / torch 三命名空间)、按 namespace 的 doc 体系、`mismatch_api_record.md` 全套、`add-compat-api`/`fix-compat-api`/`compat-doc-authoring`/`compatibility-testing` 四个 Agent skill |
| **合计** | **80** | **72** | **3** | **5** | — |

整体合入率:**72/80 = 90.0%**;5 个 CLOSED 全部是被后续 PR 拆解/替代,无功能性丢失(详见 [`05-未合并PR分析.md`](05-未合并PR分析.md))。

## 阅读路径

| 顺序 | 章节 | 主题 |
|---|---|---|
| 1 | [00-项目背景与总览.md](00-项目背景与总览.md) | 项目背景、目标、技术架构、时间线 |
| 2 | [01-基础类型与Tensor/](01-基础类型与Tensor/README.md) | 基础类型(ScalarType / TypeMeta / SymInt / Layout / Storage / DispatchKey / pointer)、Tensor 接口与操作、Sparse |
| 3 | [02-CUDA与设备/](02-CUDA与设备/README.md) | CUDA 流事件 Context、CUDABlas / Generator / Philox、平台与设备适配(Windows / Mac / DCU / XPU) |
| 4 | [03-接口结构与基础设施/](03-接口结构与基础设施/README.md) | libtorch 入口与宏、Linux ABI CI 与编译相关 |
| 5 | [04-PaddleCppAPITest仓库/](04-PaddleCppAPITest仓库/README.md) | 测试仓贡献(测试用例 / 文档 / mismatch 复现) |
| 6 | [05-未合并PR分析.md](05-未合并PR分析.md) | OPEN/CLOSED 状态与替代关系 |
| 7 | [06-完整PR列表.md](06-完整PR列表.md) | 全部 80 个 PR 的总表与主题串联速查表 |

## 关键技术亮点

- **设计上**与 PyTorch 上游头文件树严格对齐:`compat/ATen/{,core,cuda,native,ops}/`、`compat/c10/{core,util,cuda,macros}/`、`compat/torch/{csrc,headeronly}/`、`compat/utils/`,使下游可直接 `#include <ATen/...>` / `<c10/...>` / `<torch/...>`。
- **架构上**关键桥接全部走 `phi::GPUContext` / `phi::DenseTensor` / `phi::DataType`,而不是单独复制一份内核;CUDA stream/event 由 compat 层与 `phi::GPUContext` 双向同步(#78652、#78902),`TensorBase::storage()` 同步 compat storage holder(#78826)。
- **质量保障**:在 Paddle 仓内提供单元测试(`test/cpp/compat/c10_*_test.cc`、`ATen_*_test.cc`、`torch/*_test.cc`),在测试仓 PaddleCppAPITest 提供**等价对照实验**(同一份代码分别链接 Paddle compat 与 libtorch,二者输出 `diff` 必须为空)。
- **CI 防回退**:新增 Linux 动态符号 ABI 兼容性检查(#78831),对 `c10::`/`at::`/`torch::`/`caffe2::` 命名空间下的导出符号删除直接拦截,保护兼容承诺。

## 工作量数据

- **代码行数**:在 Paddle 仓累计 +27,000 / -3,800 行左右(以合入 PR 统计),其中 compat 层贡献(`paddle/phi/api/include/compat/`)占绝大多数;PaddleCppAPITest 仓累计 +20,000 行左右,主要是 GoogleTest 用例和 doc。
- **PR 节奏**:1 月 4 PR/周、2 月 6 PR/月、3 月 8 PR/月、4 月 18 PR/月(密集对齐期)、5 月维护期 3 PR — 详细时间线见 [`00-项目背景与总览.md`](00-项目背景与总览.md)。

---

> 文档约定:文中所有的 PR 编号都加链接(`#XXXXX`);提及文件路径时使用反引号或 `[路径](relative)` 形式;时间均为 UTC 绝对日期;命名空间、类型名、宏名、函数名保留英文原貌。
