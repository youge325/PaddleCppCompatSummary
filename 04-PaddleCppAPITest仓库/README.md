# 04 · PaddleCppAPITest 仓库

> [PFCCLab/PaddleCppAPITest](https://github.com/PFCCLab/PaddleCppAPITest) — Paddle C++ API 兼容层的**对照测试 / 文档 / mismatch 记录**仓库。31 个 PR,25 个 MERGED,6 个 OPEN,作者均为 [@youge325](https://github.com/youge325)。

## 1. 仓库定位

PaddleCppAPITest 与 Paddle 主仓的关系:

```
Paddle 主仓                             PaddleCppAPITest 仓
────────────────────                    ────────────────────
paddle/phi/api/include/compat/    ───► <linked-as>libtorch
    (compat 实现头/源)                       │
                                            ▼
                                    test/ATen/, c10/, torch/
                                    (GoogleTest 测试代码)
                                            │
                                            ├── 同一份测试代码
                                            │     ├──→ link Paddle compat → 输出 paddle_*Test.log
                                            │     └──→ link 真 libtorch    → 输出 torch_*Test.log
                                            │
                                            └── diff 两份 log,对齐 = MATCH
```

测试仓的**核心价值**:同一份 C++ 测试代码,分别链接 Paddle compat 和真正的 libtorch,验证两边输出完全一致 — **行为对齐而非接口对齐**。这是比"能编译过"更强的对齐保障。

## 2. 子文档

| # | 文件 | 主要内容 |
|---|---|---|
| 01 | [01-测试用例.md](01-测试用例.md) | 按 ATen / c10 / torch 命名空间分类的 GoogleTest 用例;基础设施 PR |
| 02 | [02-文档贡献.md](02-文档贡献.md) | tensor_base、TypeMeta、cuda_stream 等文档;按 namespace 的目录组织;4 个 Agent skill |
| 03 | [03-mismatch复现.md](03-mismatch复现.md) | #58 reproduce some mismatch points;`mismatch_api_record.md` 体系;`unmatch_` 前缀测试 |
| 04 | [04-API映射表.md](04-API映射表.md) | #61/#63/#73 API 映射基础设施、目录重组、Tensor 体系架构图、GPUContext 依赖分析 |

## 3. 31 个 PR 速览

按时间顺序:

| 序 | PR# | 创建日 | 类型 | 简述 |
|---:|---|---|---|---|
| 1 | [#8](https://github.com/PFCCLab/PaddleCppAPITest/pull/8) | 2025-12-18 | 基础设施 | 修复 cmake 在新版下 `ctest` 找不到目标文件 |
| 2 | [#9](https://github.com/PFCCLab/PaddleCppAPITest/pull/9) | 2025-12-28 | 测试用例 | 测试 defined/reset + 重写 8 个 op 测试 |
| 3 | [#13](https://github.com/PFCCLab/PaddleCppAPITest/pull/13) | 2026-01-04 | 测试用例 | layout 测试 |
| 4 | [#14](https://github.com/PFCCLab/PaddleCppAPITest/pull/14) | 2026-01-04 | 测试用例 | storage 测试 |
| 5 | [#15](https://github.com/PFCCLab/PaddleCppAPITest/pull/15) | 2026-01-05 | 文档 | 首批 4 篇文档:allocator/layout/storage/tensor_base |
| 6 | [#17](https://github.com/PFCCLab/PaddleCppAPITest/pull/17) | 2026-01-08 | 测试用例 | squeeze/unsqueeze 测试 |
| 7 | [#18](https://github.com/PFCCLab/PaddleCppAPITest/pull/18) | 2026-01-11 | 测试用例 | SymInt 测试 |
| 8 | [#19](https://github.com/PFCCLab/PaddleCppAPITest/pull/19) | 2026-01-11 | 测试用例 | ScalarType 测试 |
| 9 | [#21](https://github.com/PFCCLab/PaddleCppAPITest/pull/21) | 2026-01-17 | 测试用例 | pointer/intrusive_ptr 测试 |
| 10 | [#23](https://github.com/PFCCLab/PaddleCppAPITest/pull/23) | 2026-01-22 | 测试用例 | Sparse 测试 |
| 11 | [#24](https://github.com/PFCCLab/PaddleCppAPITest/pull/24) | 2026-01-24 | 测试重构 | 重写 4 个测试文件结构 |
| 12 | [#25](https://github.com/PFCCLab/PaddleCppAPITest/pull/25) | 2026-01-25 | 测试用例 | TensorAccessor 测试 |
| 13 | [#26](https://github.com/PFCCLab/PaddleCppAPITest/pull/26) | 2026-01-27 | 测试用例 | flatten/narrow 测试 |
| 14 | [#34](https://github.com/PFCCLab/PaddleCppAPITest/pull/34) | 2026-02-06 | 文档 | 更新 tensor_base 文档 |
| 15 | [#40](https://github.com/PFCCLab/PaddleCppAPITest/pull/40) | 2026-02-27 | 测试 + skill | sum/t/transpose_/view_as/coalesce/item 测试 + 迭代 `compatibility-testing` skill(由 @Le-soleile #38 创建) |
| 16 | [#46](https://github.com/PFCCLab/PaddleCppAPITest/pull/46) | 2026-03-05 | 测试 + 文档 | CUDABlas/CUDAContext 大综合 PR(+4111/-655,97 文件) + `compat-doc-authoring` skill |
| 17 | [#47](https://github.com/PFCCLab/PaddleCppAPITest/pull/47) | 2026-03-05 | 测试 + 文档 | RecordStream 测试 + stream 三篇文档 |
| 18 | [#49](https://github.com/PFCCLab/PaddleCppAPITest/pull/49) | 2026-03-07 | 测试 + 文档 | Generator 测试 + 3 篇文档 |
| 19 | [#56](https://github.com/PFCCLab/PaddleCppAPITest/pull/56) | 2026-03-13 | 文档 | TypeMeta 文档 |
| 20 | [#58](https://github.com/PFCCLab/PaddleCppAPITest/pull/58) | 2026-03-27 | 重组 + mismatch | reproduce some mismatch points + 目录重组到 ATen/ c10/ torch/ |
| 21 | [#59](https://github.com/PFCCLab/PaddleCppAPITest/pull/59) | 2026-04-04 | 文档 + CI | cuda_stream 等文档大更新 + 3 个 add/fix/compat skill + GitHub Actions |
| 22 | [#60](https://github.com/PFCCLab/PaddleCppAPITest/pull/60) | 2026-04-29 | 测试 + CI | `PADDLE_WITH_CUSTOM_KERNEL` 宏接入 + 复现 MaybeResetHolder + CUDAStreamDeadlockAlignmentTest |
| 23 | [#61](https://github.com/PFCCLab/PaddleCppAPITest/pull/61) | 2026-05-18 | 文档 | 综合 C++ API mapping docs、alias mapping、skill refinements |
| 24 | [#62](https://github.com/PFCCLab/PaddleCppAPITest/pull/62) | 2026-05-19 | 文档 | skill 文档补充 Step 0 环境搭建和 Step 7 PR workflow(OPEN) |
| 25 | [#63](https://github.com/PFCCLab/PaddleCppAPITest/pull/63) | 2026-05-25 | 文档 | GPUContext 架构图 + API mapping table 更新 |
| 26 | [#64](https://github.com/PFCCLab/PaddleCppAPITest/pull/64) | 2026-05-28 | 测试 + 文档 | broadcast_to 跨框架测试 + mapping doc 更新(OPEN) |
| 27 | [#65](https://github.com/PFCCLab/PaddleCppAPITest/pull/65) | 2026-05-28 | 测试用例 | vstack 行为对齐测试(OPEN) |
| 28 | [#66](https://github.com/PFCCLab/PaddleCppAPITest/pull/66) | 2026-05-28 | 测试用例 | take 行为对齐测试(OPEN) |
| 29 | [#67](https://github.com/PFCCLab/PaddleCppAPITest/pull/67) | 2026-05-29 | 测试用例 | svd 行为对齐测试(OPEN) |
| 30 | [#68](https://github.com/PFCCLab/PaddleCppAPITest/pull/68) | 2026-05-29 | 测试用例 | repeat_interleave 行为对齐测试(OPEN) |
| 31 | [#73](https://github.com/PFCCLab/PaddleCppAPITest/pull/73) | 2026-05-31 | 文档 + 测试重构 | Tensor 体系架构图 + 测试拆分 + skill Step 0/7(MERGED) |

## 4. 工作量与覆盖率

`#59` PR 描述附带的覆盖率截图(3308×2066 像素)显示**项目结束时**测试仓的覆盖率:

- 大多数 namespace 覆盖率 > 80%
- `ATen/core` 中涉及 jit 编译的文件覆盖率较低(`ivalue.h` / `jit_type.h`)— 这是 PyTorch jit 路径,Paddle 暂不需要
- `c10/cuda` 中涉及 device 切换的 `CUDAGuard` 类未完全覆盖 — 本地只有单卡
- `utils/` 中 `Undefined` 数据类型覆盖率有提升空间

`#60` 中的覆盖率截图显示在引入 `PADDLE_WITH_CUSTOM_KERNEL` 路径后,**`MaybeResetHolder` 相关分支也被覆盖**,coverage 略有上升。

## 5. 关键架构决策

1. **单层 CMake + `GLOB_RECURSE`**:测试仓的 [`CMakeLists.txt`](https://github.com/PFCCLab/PaddleCppAPITest/blob/develop/CMakeLists.txt) 用一个 `file(GLOB_RECURSE TEST_SRCS test/*.cpp)`,所有测试统一编译为一个可执行,无需每个子目录维护 CMakeLists。代价是新增测试时需要触发一次 cmake reconfigure。

2. **测试同时按 namespace 镜像 libtorch**:在 #58 中把扁平的 `test/*Test.cpp` 重组为 `test/ATen/.../`、`test/c10/.../`、`test/torch/.../`,与 libtorch 的源码树**完全一致**。这样下游用户阅读 libtorch 文档可以**直接对照测试用例**。

3. **`unmatch_` 前缀标记已知差异**:不能对齐的测试不删除,而是改名为 `unmatch_<原名>.cpp`(如 `test/torch/csrc/api/include/torch/unmatch_PythonTest.cpp`),在 CI 中被跳过,但**保留作为已知差异的可执行记录**。

4. **`mismatch_api_record.md` 按 namespace 散布**:在每个 doc 子目录下放一份 `mismatch_api_record.md`,记录该 namespace 下与 PyTorch 不对齐的接口、原因、规避方案。详见 [`03-mismatch复现.md`](03-mismatch复现.md)。

5. **4 个 Agent skill 沉淀工程经验**:`compatibility-testing` / `compat-doc-authoring` / `add-compat-api` / `fix-compat-api` — 每个 skill 是一份 markdown SOP,记录怎么写测试 / 怎么写文档 / 怎么加 API / 怎么修 API。后续营员和 LLM 助手按这些 skill 工作即可。(测试仓还有一个 `cpp-api-compatible` skill,由同期营员 [@Le-soleile](https://github.com/Le-soleile) 在 PR #57 中创建,不在本总结范围。)

按文档顺序阅读即可。
