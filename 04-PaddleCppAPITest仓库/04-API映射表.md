# 04 · API 映射表

> 范围:PaddleCppAPITest 仓库 `doc/mapping/` 目录下的 C++ API 跨框架映射文档、`doc/gpucontext_dependency_analysis.md` 架构分析,以及 `doc/ATen/core/` 下的 Tensor 体系架构图。对应主仓 compat 层接口的系统性差异梳理。

## 1. 主题概述

compat 层接口数量庞大(ATen/c10/torch 共计 1000+ 个 API),逐一人工对齐成本高。为此在 PaddleCppAPITest 仓库建立了**自动化的 C++ API 映射基础设施**:

1. **签名解析器** — 从 libtorch ops 头文件和 Paddle `api.h` 提取 C++ 函数签名
2. **差异分类引擎** — 按优先级自动分类(返回类型 → 参数数量 → 参数类型 → 默认值 → 参数名)
3. **别名映射机制** — 识别跨框架名称不同但功能一致的 API,避免误判为"功能缺失"
4. **人工复核工作流** — 通过 `api-mapping-updater` / `api-mapping-verifier` skill 标准化更新与验证流程

最终产出覆盖 **1096 个实际参与映射**的 API,其中 790 个为功能缺失(尚未实现),306 个已发现具体差异。

## 2. PR 汇总表

| PR# | 状态 | 合并日期 | 标题 | 核心贡献 |
|---:|---|---|---|---|
| [#61](https://github.com/PFCCLab/PaddleCppAPITest/pull/61) | MERGED | 2026-05-19 | comprehensive C++ API mapping docs, alias mapping, and skill refinements | 从零构建映射基础设施、7 类差异文档、18 个别名映射 |
| [#63](https://github.com/PFCCLab/PaddleCppAPITest/pull/63) | MERGED | 2026-05-29 | Add GPUContext related architecture diagram and Update API mapping table | 目录重组到 `doc/mapping/`、GPUContext 依赖分析架构图、2 个 mapping skill |
| [#73](https://github.com/PFCCLab/PaddleCppAPITest/pull/73) | MERGED | 2026-05-31 | docs(skill): add Step 0 env setup and Step 7 PR workflow to add/fix | Tensor 体系架构图(`paddletensor_arch.md`、`torchtensor_arch.md`)、测试重构 |

## 3. #61 从零构建 C++ API 映射基础设施(+10707/-41 行)

### 3.1 核心脚本

**`doc/generate_cpp_api_mapping.py`**(+1310 行):

- 从 libtorch ops 头文件和 Paddle `api.h` 提取 C++ 函数签名
- 类型归一化:统一 `at::Tensor`→`Tensor`、`::std::tuple`→`tuple` 等跨框架别名
- 差异分类引擎:按优先级自动分类
- 参数名语义映射:识别 `self`→`x`、`other`→`y`、`dim`→`axis`、`weight`→`filter` 等常见映射
- **参数映射错位修复**:从索引逐位对比改为**按参数名匹配**,解决 `repeat_interleave` 等参数顺序错位导致的误判

**`doc/cpp_api_mapping_cn.md`**(+1228 行):

映射总览文档,含统计表与分类说明。

### 3.2 七类差异文档体系

| 分类 | 目录 | 数量 | 说明 |
|---|---:|---:|---|
| 仅参数名不一致 | `doc/cpp_args_name_diff/` | 72 | 如 `self`→`x`、`dim`→`axis` |
| 参数默认值不一致 | `doc/cpp_args_default_value_diff/` | 3 | `fill`、`roll`、`tile` |
| 输入参数类型不一致 | `doc/cpp_input_args_type_diff/` | 49 | `int` vs `IntArrayRef` 等 |
| 返回参数类型不一致 | `doc/cpp_output_args_type_diff/` | 29 | 返回值类型差异 |
| Paddle 参数更多 | `doc/cpp_paddle_more_args/` | 46 | Paddle 提供额外参数(如 `conv2d`) |
| PyTorch 参数更多 | `doc/cpp_torch_more_args/` | 22 | PyTorch 提供额外参数(如 `add`) |
| API 别名 | `doc/cpp_api_alias_diff/` | 18 | 名称不同但功能一致 |

### 3.3 API 别名映射机制(新增)

**问题**:脚本原本仅通过严格同名匹配定位对应 API,导致 Paddle 中使用不同名称实现的 API 被误判为"功能缺失"。

**方案**:
- `doc/discover_cpp_api_aliases.py` — 通过去 `_` 前缀、命名风格翻转(`conv_transposeNd`↔`convNd_transpose`)、字符串相似度等启发式规则自动发现候选别名
- `doc/cpp_api_alias_mapping.json` — 18 个高置信度跨框架映射

**已确认的别名映射示例**:

| PyTorch | Paddle | 类型 |
|---|---|---|
| `conv_transpose2d` | `conv2d_transpose` | 命名风格差异 |
| `max_pool2d_with_indices` | `max_pool2d_with_index` | 命名风格差异 |
| `_log_softmax` | `log_softmax` | PyTorch 内部变体(去前缀) |
| `_fft_c2c` | `fft_c2c` | PyTorch 内部变体(去前缀) |
| `grid_sampler` | `grid_sample` | 功能同名不同 |
| `log_sigmoid` | `logsigmoid` | 下划线差异 |
| `range` | `arange` | 语义等价 |

**效果**:
- 功能缺失:808 → **790**(-18)
- 实际参与映射:1078 → **1096**(+18)

### 3.4 Skill 完善

4 个 skill 同步更新:`add-compat-api` / `compat-doc-authoring` / `compatibility-testing` / `fix-compat-api`。

## 4. #63 目录重组与架构图(+80864/-1765 行)

### 4.1 目录重组

将 #61 中散落在 `doc/` 根目录的 mapping 文件统一搬到 `doc/mapping/` 子目录:

```
doc/mapping/
├── README.md                           — 映射体系导览
├── cpp_api_mapping_cn.md               — 映射总览(从 doc/ 根移入)
├── cpp_api_alias_candidates.json
├── cpp_api_alias_mapping.json
├── cpp_api_alias_diff/                 — 18 个别名差异
├── cpp_args_name_diff/                 — 72 个参数名差异
├── cpp_args_default_value_diff/        — 3 个默认值差异
├── cpp_input_args_type_diff/           — 49 个输入类型差异
├── cpp_output_args_type_diff/          — 29 个返回类型差异
├── cpp_paddle_more_args/               — 46 个 Paddle 多参数
└── cpp_torch_more_args/                — 22 个 Torch 多参数
```

### 4.2 GPUContext 依赖分析架构图

**`doc/gpucontext_dependency_analysis.md`**(+330 行):

调研 `phi::GPUContext` 的完整依赖链,绘制从 `c10::cuda::CUDAStream` 到 `phi::GPUContext` 再到 Paddle 内核发射的调用栈。这是 compat 层 stream 同步设计的底层参考文档。

### 4.3 新增 2 个 mapping skill

- `.github/skills/api-mapping-updater/SKILL.md`(+176 行) — 更新 API mapping 的标准流程
- `.github/skills/api-mapping-verifier/SKILL.md`(+193 行) — 验证 mapping 准确性的检查清单

## 5. #73 Tensor 体系架构图与测试重构(+2868/-938 行)

### 5.1 Tensor 体系架构图

- **`doc/ATen/core/paddletensor_arch.md`**(+373 行) — 从 Paddle 用户视角看 Tensor 体系
- **`doc/ATen/core/torchtensor_arch.md`**(+452 行) — 从 libtorch 用户视角看 Tensor 体系

这两篇文档是 #59 中 `doc/PaddleTensor.md` / `doc/TorchTensor.md` 的架构图版本,用 Mermaid 图展示 Tensor 生命周期、storage holder 同步、view 语义等。

### 5.2 测试重构

`test/ATen/core/TensorTest.cpp`(-930 行,大幅精简) — 把原先臃肿的 TensorTest 按功能拆分到各 op 测试文件中,提升模块化:

- `test/ATen/ops/AbsTest.cpp`(+59)
- `test/ATen/ops/AsStridedTest.cpp`(+73)
- `test/ATen/ops/ClampTest.cpp`(+181)
- `test/ATen/ops/ExpandTest.cpp`(+162)
- `test/ATen/ops/MiscTensorTest.cpp`(+106)
- `test/ATen/ops/ResizeTest.cpp`(+86)
- `test/ATen/ops/StdTest.cpp`(+66)
- `test/ATen/ops/VarTest.cpp`(+87)
- ... 等 11 个测试文件新增/扩展

### 5.3 Skill 完善:Step 0 与 Step 7

- `.github/skills/add-compat-api/SKILL.md` / `fix-compat-api/SKILL.md` — 插入 Step 0(环境检测与自动配置)和 Step 7(提交 commit 并创建 PR)
- `references/Step0.md`(+111 行) — 完整命令模板(GPU 探测 / fork 克隆 / libtorch 下载 / wheel 检测)
- `references/Step7.md`(+201 行) — 完整命令模板(新分支 → commit → push → PR + heredoc body 模板)

## 6. 映射统计总览(截至 #63)

| 分类 | 数量 |
|---|---|
| API 完全一致 | 66 |
| 仅 API 调用方式不一致 | 1 |
| 仅参数名不一致 | 72 |
| Paddle 参数更多 | 46 |
| 参数默认值不一致 | 3 |
| PyTorch 参数更多 | 22 |
| 输入参数类型不一致 | 49 |
| 返回参数类型不一致 | 29 |
| API 别名 | 18 |
| 功能缺失 | 790 |
| **实际参与映射** | **1096** |

## 7. 与 Paddle 主仓的对应关系

| 主仓 PR | 测试仓 PR | 关联内容 |
|---|---|---|
| #79173 `broadcast_to` | #64 | mapping doc 更新 |
| #79176 `column_stack` | #74 | mapping doc 更新 |
| 其他新增 API(#79175~#79201) | #65~#72 | 每新增一个 compat API,同步更新 `doc/mapping/` 中的差异记录 |

API 映射表是 compat 层**规模化管理**的基础设施 — 1000+ API 中哪些已实现、哪些有差异、差异在哪一类,都可以通过这个体系快速查询。
