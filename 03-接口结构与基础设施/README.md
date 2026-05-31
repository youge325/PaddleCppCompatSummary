# 03 · 接口结构与基础设施

本章覆盖三类**工程治理类**工作:

- **libtorch 入口与宏**:`torch/torch.h` 顶级入口、`TORCH_WARN` / `STD_CHECK` / `STD_TORCH_CHECK` 宏、`torch::headeronly` 命名空间下的纯 header 接口、代码搬迁(`Utils.h` → `Utils.cpp`、Tensor methods → `ops/`)
- **CI 与编译相关**:Linux ABI 兼容性检查、`flashmla` 编译修复、`paddlecodec` 编译适配、`deepep` legacy API 清理、FA 编译线程数调优、文件改名

这些工作单独看不"性感",但它们决定了 compat 层能否长期可维护、能否兜住后续 PR 的回退、能否快速接入新下游。

## 子文档

| # | 文件 | 主要内容 |
|---|---|---|
| 01 | [01-libtorch入口与宏.md](01-libtorch入口与宏.md) | torch/torch.h / TORCH_WARN / STD_CHECK / torch::headeronly / 代码搬迁 |
| 02 | [02-CI与编译.md](02-CI与编译.md) | Linux ABI 兼容性检查 / flashmla / paddlecodec / deepep cleanup / FA 编译 |

## 涉及 PR

**Paddle 主仓**(11 个):#77177、#77694(open)、#77854、#78244、#78549、#78550、#78576、#78580、#78641、#78662、#78831

## 关键架构决策

1. **ABI 检查作为最后一道防线**(#78831):compat 层向外承诺 `c10::`/`at::`/`torch::`/`caffe2::` 命名空间下的符号稳定。Static Check 直接接 `readelf --dyn-syms` 比对 base wheel 与 PR wheel,protected 符号被删就直接 fail 并要求导师手动 approve。这是项目能持续生长而不破坏下游的核心保障。

2. **下游编译触发的 hotfix 优先**:`flashmla`(#78550)、`paddlecodec`(#78641)、`DeepEP`(含 hybrid-ep 分支,#77854/#78576/#78584/#78549)等每次下游报编译错,都会迅速产出一个 PR 修复 — compat 层的功能不是"猜测出来",而是"被下游编译需求倒逼出来"。

3. **宏命名严格对齐**:`TORCH_WARN` / `STD_CHECK` / `STD_TORCH_CHECK` / `TORCH_CHECK_EQ/NE/LT/LE/GT/GE` 全部沿用 PyTorch 上游命名,保证下游代码无需改宏即可编译。

4. **headeronly 化解 include 传染**:某些纯计算类型(`ScalarType` 枚举、`DeviceType` 枚举、`TensorAccessor` 模板)被搬到 `torch/headeronly/`,让纯 header-only 库可以只引用这部分,而不带上 Paddle 的 cpp 依赖。

按文档顺序阅读即可。
