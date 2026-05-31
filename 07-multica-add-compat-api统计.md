# 07 · multica + Claude Code(Kimi-k2.6) 新增 10 个 compat API 统计

> 工具链:multica → Claude Code → `add-compat-api` skill → Paddle 主仓 + PaddleCppAPITest 仓

## 1. API 主统计表

| # | API 名称 | Paddle 主仓 PR | PaddleCppAPITest PR | 大致耗时 | 人工干预次数 | 干预总时长 |
|---|---|---|---|---|---|---|
| 1 | broadcast_to | [#79173](https://github.com/PaddlePaddle/Paddle/pull/79173) | [#64](https://github.com/PFCCLab/PaddleCppAPITest/pull/64) | 30m 1s | 6 | 94m 13s |
| 2 | vstack | [#79175](https://github.com/PaddlePaddle/Paddle/pull/79175) | [#65](https://github.com/PFCCLab/PaddleCppAPITest/pull/65) | 21m 23s | 3 | 20m 59s |
| 3 | column_stack | [#79176](https://github.com/PaddlePaddle/Paddle/pull/79176) | [#74](https://github.com/PFCCLab/PaddleCppAPITest/pull/74) | 21m 32s | 3 | 19m 56s |
| 4 | take | [#79178](https://github.com/PaddlePaddle/Paddle/pull/79178) | [#66](https://github.com/PFCCLab/PaddleCppAPITest/pull/66) | 36m 19s | 2 | 14m 5s |
| 5 | svd | [#79187](https://github.com/PaddlePaddle/Paddle/pull/79187) | [#67](https://github.com/PFCCLab/PaddleCppAPITest/pull/67) | 66m 15s | 2 | 22m 41s |
| 6 | repeat_interleave | [#79189](https://github.com/PaddlePaddle/Paddle/pull/79189) | [#68](https://github.com/PFCCLab/PaddleCppAPITest/pull/68) | 44m 21s | 1 | 3m 46s |
| 7 | cholesky | [#79194](https://github.com/PaddlePaddle/Paddle/pull/79194) | [#69](https://github.com/PFCCLab/PaddleCppAPITest/pull/69) | 42m 24s | 2 | 7m 43s |
| 8 | index_reduce | [#79195](https://github.com/PaddlePaddle/Paddle/pull/79195) | [#70](https://github.com/PFCCLab/PaddleCppAPITest/pull/70) | 36m 21s | 4 | 40m 39s |
| 9 | scatter_reduce | [#79196](https://github.com/PaddlePaddle/Paddle/pull/79196) | [#71](https://github.com/PFCCLab/PaddleCppAPITest/pull/71) | 42m 39s | 3 | 19m 1s |
| 10 | stft | [#79201](https://github.com/PaddlePaddle/Paddle/pull/79201) | [#72](https://github.com/PFCCLab/PaddleCppAPITest/pull/72) | 71m 7s | 3 | 24m 4s |

## 2. 各 API 人工干预内容表

### 2.1 broadcast_to（6 次干预，总时长 94m 13s）

| 干预序号 | 干预运行时长 | 干预内容/原因 |
|---|---|---|
| 1 | 3m 45s | PR Body 没有按照 skill 的规范来写，修复 PR Body |
| 2 | 28m 54s | 使用 fix-compat-api 根据 PR Review 修复接口 |
| 3 | 11m 4s | 使用 fix-compat-api 根据 PR Review 修复接口 |
| 4 | 17m 19s | 对齐测试绕过了尚未对齐的语义，恢复测试 |
| 5 | 6m 47s | 对齐测试绕过了尚未对齐的语义，恢复测试 |
| 6 | 26m 24s | 对齐语义 |

### 2.2 vstack（3 次干预，总时长 20m 59s）

| 干预序号 | 干预运行时长 | 干预内容/原因 |
|---|---|---|
| 1 | 1m 39s | PR Body 没有按照 skill 的规范来写，修复 PR Body |
| 2 | 14m 59s | 使用 fix-compat-api 根据 PR Review 修复接口 |
| 3 | 4m 21s | 使用 fix-compat-api 根据 PR Review 修复接口 |

### 2.3 column_stack（3 次干预，总时长 19m 56s）

| 干预序号 | 干预运行时长 | 干预内容/原因 |
|---|---|---|
| 1 | 1m 2s | PR Body 没有按照 skill 的规范来写，修复 PR Body |
| 2 | 15m 57s | 使用 fix-compat-api 根据 PR Review 修复接口 |
| 3 | 2m 57s | PaddleCppAPITest 仓库的 PR 没有创建在 upstream，而是创建在 fork 仓库 |

### 2.4 take（2 次干预，总时长 14m 5s）

| 干预序号 | 干预运行时长 | 干预内容/原因 |
|---|---|---|
| 1 | 59s | PR Body 没有按照 skill 的规范来写，修复 PR Body |
| 2 | 13m 6s | 使用 fix-compat-api 根据 PR Review 修复接口 |

### 2.5 svd（2 次干预，总时长 22m 41s）

| 干预序号 | 干预运行时长 | 干预内容/原因 |
|---|---|---|
| 1 | 9m 47s | 没有创建 PR |
| 2 | 12m 54s | 使用 fix-compat-api 根据 PR Review 修复接口 |

### 2.6 repeat_interleave（1 次干预，总时长 3m 46s）

| 干预序号 | 干预运行时长 | 干预内容/原因 |
|---|---|---|
| 1 | 3m 46s | 没有创建 PR |

### 2.7 cholesky（2 次干预，总时长 7m 43s）

| 干预序号 | 干预运行时长 | 干预内容/原因 |
|---|---|---|
| 1 | 2m 22s | PR Body 没有按照 skill 的规范来写，修复 PR Body |
| 2 | 5m 21s | 使用 fix-compat-api 根据 PR Review 修复接口 |

### 2.8 index_reduce（4 次干预，总时长 40m 39s）

| 干预序号 | 干预运行时长 | 干预内容/原因 |
|---|---|---|
| 1 | 12m 48s | 没有创建 PR，且 Paddle 仓库缺失测试 |
| 2 | 1m 2s | PR Body 没有按照 skill 的规范来写，修复 PR Body |
| 3 | 13m 48s | 使用 fix-compat-api 根据 PR Review 修复接口 |
| 4 | 13m 1s | 使用 fix-compat-api 根据 PR Review 修复接口 |

### 2.9 scatter_reduce（3 次干预，总时长 19m 1s）

| 干预序号 | 干预运行时长 | 干预内容/原因 |
|---|---|---|
| 1 | 6m 23s | 没有创建 PR |
| 2 | 1m 11s | PR Body 没有按照 skill 的规范来写，修复 PR Body |
| 3 | 11m 27s | 使用 fix-compat-api 根据 PR Review 修复接口（重编译 120m 超时，已丢弃） |

### 2.10 stft（3 次干预，总时长 24m 4s）

| 干预序号 | 干预运行时长 | 干预内容/原因 |
|---|---|---|
| 1 | 2m 17s | PR Body 没有按照 skill 的规范来写，修复 PR Body |
| 2 | 4m 21s | 在 develop 分支操作，没有 checkout 新分支 |
| 3 | 17m 26s | 使用 fix-compat-api 根据 PR Review 修复接口 |

## 3. 总结指标

| 指标 | 数值 |
|---|---|
| 总计 API 数 | 10 |
| 总计干预次数 | 29 次 |
| 平均每个 API 干预次数 | 2.9 次 |
| 总计干预时长 | 267m 7s（约 4h 27m） |
| 平均每个 API 干预时长 | 26m 43s |
| 单次干预平均时长 | 9m 13s |

## 4. 干预原因分布

| 干预原因 | 次数 | 占比 |
|---|---|---|
| PR Body 格式不规范 | 7 | 24.1% |
| 使用 fix-compat-api 根据 PR Review 修复接口 | 14 | 48.3% |
| 没有创建 PR | 4 | 13.8% |
| 对齐测试/语义问题 | 3 | 10.3% |
| 分支/仓库操作错误 | 2 | 6.9% |
| **合计** | **29** | **100%** |
