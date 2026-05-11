# 03 · mismatch 复现

> 范围:PaddleCppAPITest 仓库中**已知不对齐**的接口复现机制 — `mismatch_api_record.md` 体系、`unmatch_` 前缀测试、#58 "reproduce some mismatch points" PR。

## 1. 主题概述

兼容工作不可能做到 **100% 对齐** — 总会有些角落由于 Paddle 与 PyTorch 底层实现差异、运行时语义差异、设备能力差异等导致行为不一致。这些差异需要:

1. **能复现** — 有可执行的 GoogleTest 案例,把差异暴露出来
2. **能记录** — 在 `mismatch_api_record.md` 中文字描述差异、原因、规避方案
3. **能跟踪** — 当 Paddle 主仓 PR 修复差异后,把 `unmatch_*Test.cpp` 改名(去掉 `unmatch_` 前缀),把记录从 mismatch 表中移除

测试仓在这一块的核心 PR 是 [#58 reproduce some mismatch points](https://github.com/PFCCLab/PaddleCppAPITest/pull/58)(+3831/-3252 行,126 文件)。

## 2. `mismatch_api_record.md` 体系

经过 #34 → #46 → #58 → #59 → #60 的多次迭代,最终形态是按 namespace 散布的多份文档:

| 路径 | 行数 | 范围 |
|---|---:|---|
| `doc/mismatch_api_record.md` | 顶层索引 + 跨命名空间汇总 | — |
| `doc/ATen/mismatch_api_record.md` | +159(在 #58) | ATen 顶层 |
| `doc/ATen/core/mismatch_api_record.md` | +123(在 #58) | ATen::core(Generator/TensorBase/TensorAccessor 等) |
| `doc/ATen/cuda/mismatch_api_record.md` | +155(在 #58) | ATen::cuda(CUDABlas / CUDAContext) |
| `doc/ATen/ops/mismatch_api_record.md` | +417(在 #58) | ATen ops(每个 op 一节) |
| `doc/c10/core/mismatch_api_record.md` | +694(在 #58) | c10::core(Storage / Stream / Event / Device / ScalarType / Layout) |
| `doc/c10/cuda/mismatch_api_record.md` | +61(在 #58) | c10::cuda(CUDAStream / CUDAFunctions / CUDAGuard) |
| `doc/c10/util/mismatch_api_record.md` | +204(在 #58) | c10::util(intrusive_ptr / TypeMeta / Exception) |

每份文档的格式:

```markdown
## API: at::SomeFunction

### 期望(PyTorch)
- 描述 PyTorch 行为
- 引用 PyTorch 源码路径

### 现状(Paddle compat)
- 描述当前 compat 层行为
- 给出复现代码片段(指向 test/.../*Test.cpp)

### 原因
- 解释为何不能对齐:实现差异 / 设计取舍 / 优先级问题

### 规避方案
- 用户视角的 workaround

### 跟踪
- 是否计划修复;关联 Paddle 主仓 issue/PR
```

## 3. 关键 PR:#58 reproduce some mismatch points

#### 背景

在测试仓 #46 落地后,经过几次 rebase 与 Agent 自动修复,**很多 mismatch 已经被"隐式抹掉"**(因为 Agent 修改测试代码让它"看起来对齐"了)。但这种"看起来对齐"其实掩盖了底层差异,后续 compat 改动一旦碰到这些点,问题会突然爆发。

PR 描述:

> 复现所有因 rebase 或 Agent 自动修复而消除的差异点

#### 改动(+3831/-3252,126 文件)

**1. 目录重组**:把扁平 `test/*Test.cpp` 重组为 `test/ATen/`、`test/c10/`、`test/torch/`:

```
test/ATen/AccumulateTypeTest.cpp        (从 test/AccumulateTypeTest.cpp 移)
test/ATen/DeviceGuardTest.cpp           (从 test/DeviceGuardTest.cpp 移)
test/ATen/IndexingTest.cpp              (从 test/IndexingTest.cpp 移)
test/ATen/UtilsTest.cpp                 (新增 +129 行)
test/ATen/core/IValueTest.cpp           (改动 +6/-5)
test/ATen/core/LayoutTest.cpp           (移)
test/ATen/core/TensorTest.cpp           (改动 +8/-9)
test/ATen/core/TensorAccessorTest.cpp   (移)
test/ATen/core/TensorUtilTest.cpp       (移)
test/ATen/cuda/CUDABlasTest.cpp         (移)
test/ATen/cuda/CUDAContextTest.cpp      (+124 行,新建,把 #46 中 unmatch_CUDAContextTest 改写为正式)
test/ATen/cuda/CUDADataTypeTest.cpp     (改动 +15/-12)
test/ATen/cuda/GeneratorTest.cpp        (移)
test/ATen/native/RangeUtilsTest.cpp     (移)
test/ATen/ops/*Test.cpp                 (约 30 个 op 测试统一移到 ops/ 子目录)
test/c10/core/AllocatorCompatTest.cpp   (+190/-7)
```

**2. mismatch 文档铺开**:7 个 `mismatch_api_record.md` 在各 namespace 子目录下创建(总计 +1813 行)。

**3. 测试代码的实质改动**:

- `test/ATen/UtilsTest.cpp` 新增 129 行,覆盖原先没测的 `at::AT_ASSERT*` / `at::AT_CHECK*` / `_PD_Internal::*` 路径。
- `test/ATen/cuda/CUDAContextTest.cpp` 新增 124 行,把 `unmatch_CUDAContextTest.cpp` 中部分已经对齐的 case 提升为正式测试。
- `test/c10/core/AllocatorCompatTest.cpp`(+190/-7) — 大幅扩充 Allocator/DataPtr 测试。
- `test/ATen/ops/ArangeTest.cpp`(+64/-1) — 新增 arange 无 dtype 推断 case,复现 #78552 中要修的问题(尚未合入时这是 unmatch,合入后变 match)。

**4. 删除已过时的 unmatch**:
- `test/CUDATest2.cpp`(-306) — 删除老 CUDA 测试,功能已在 `test/c10/cuda/CUDATest2.cpp` 重写
- `test/DeviceTest.cpp`(-77) — 删除老 Device 测试
- `test/c10/core/unmatch_EventTest.cpp`(在 #59 中 -260 行) — Event 已在 #78553 对齐,unmatch 文件删除

**5. ccache 集成** — `cmake/ccache.cmake`(+30 行,新建),加速 CI 编译。

#### 影响

这个 PR **明确了 compat 层的"已知差异边界"**。后续每次 Paddle 主仓有改动,营员都用 `mismatch_api_record.md` 对照检查:

- 改动是否消除了某个 known mismatch → 把对应记录从表中移除
- 改动是否引入新的 mismatch → 新加一条记录,把 test 改名为 `unmatch_*`

## 4. `unmatch_` 前缀测试约定

测试仓使用一个简单约定:**已知不对齐的测试,文件名加 `unmatch_` 前缀**。例:

| 文件 | 状态 |
|---|---|
| `test/torch/csrc/api/include/torch/PythonCompatTest.cpp` | 对齐(MATCH) |
| `test/torch/csrc/api/include/torch/unmatch_PythonTest.cpp` | 不对齐(已知,CI 跳过) |

CMake 中根据文件名前缀决定是否纳入 `ctest`:

```cmake
file(GLOB_RECURSE ALL_TESTS test/*.cpp)
foreach(test_file ${ALL_TESTS})
  get_filename_component(name ${test_file} NAME_WE)
  if(NOT name MATCHES "^unmatch_")
    # 加入 ctest 测试目标
    add_test(NAME ${name} COMMAND $<TARGET_FILE:${name}>)
  endif()
endforeach()
```

这样**未对齐的测试代码仍在 repo 中保留**(作为"如何复现差异"的可执行文档),但不进入回归 CI。当 Paddle 主仓修复后,只需 `git mv unmatch_PythonTest.cpp PythonTest.cpp` 改名,测试自动纳入 CI。

## 5. 已知 mismatch 的几个代表

> 这些是 #58 起步时记录、截至 #59 完成时仍未对齐的接口。完整列表在各 namespace 的 `mismatch_api_record.md`。

### 5.1 `at::abs` 非连续 stride

- **PyTorch**:`tensor.transpose(0,1).abs()` 返回的 Tensor 保持非连续 stride
- **Paddle compat**:Paddle 底层强制 contiguous,返回连续 Tensor
- **影响**:`data_ptr` 存储数据顺序不同,如果下游直接读 `data_ptr` 会拿到不同字节
- **规避**:用 `.contiguous().abs()` 显式 contiguous;或读取时通过 `tensor[i][j]` 而非 `data_ptr` 索引
- **跟踪**:[#78099](https://github.com/PaddlePaddle/Paddle/pull/78099) PR 描述里专门提了这个,作为"已知差异"暂不修复

### 5.2 `at::sparse_coo_tensor` indices 排序

- **PyTorch**:`coalesce()` 后 indices 是 lexicographic 排序
- **Paddle compat**:Paddle 内部用稳定排序但 flatten 顺序略有差异
- **影响**:输出 values 元素顺序不一致,但数值合并结果相同
- **跟踪**:`doc/ATen/ops/mismatch_api_record.md` 标"语义等价、字节非等价"

### 5.3 `Tensor.to_sparse()` / `Tensor.to_dense()`

- **PyTorch**:支持 `tensor.to_sparse(sparse_dim)` 反向转换
- **Paddle compat**:暂未实现,需要补 `_PD_Internal::DenseToSparseCoo`
- **规避**:用 `at::sparse_coo_tensor(...)` 构造稀疏张量;dense ← sparse 用 `tensor._values()` 取出 dense 子张量
- **跟踪**:低优先级,DeepEP / FastDeploy 不使用

### 5.4 `c10::Event` CPU 路径

- **PyTorch**:CPU device 上 `Event::record()` 抛 "Backend doesn't support events"
- **Paddle compat**:#78553 之前是 segfault,之后已对齐抛异常
- **跟踪**:已 RESOLVED,#58 中保留为 unmatch 记录用于回归检查,#59 中 `test/c10/core/unmatch_EventTest.cpp` 被删除

### 5.5 `TensorBase::weak_use_count`

- **PyTorch**:返回真正的 weak refs 数量
- **Paddle compat**:由于 Paddle 用 `shared_ptr` 而非 `intrusive_ptr`,无法完全对齐;#78809 只对齐 "已定义 → ≥1" 的可观察行为
- **跟踪**:`doc/c10/util/mismatch_api_record.md`、`doc/ATen/core/mismatch_api_record.md`

### 5.6 `at::cuda::CUDAStream::priority()`

- **PyTorch**:返回 stream 创建时的 priority(-1 high / 0 default)
- **Paddle compat**:Paddle 的 `phi::CUDAStream` 不保存 priority,返回 default
- **规避**:用 `priority_range()` 查询硬件能力但不依赖具体 stream 的 priority
- **跟踪**:`doc/c10/cuda/mismatch_api_record.md`

## 6. 与 Paddle 主仓的协作模式

mismatch 复现工作 **驱动** Paddle 主仓的多个 PR:

1. **测试仓发现差异** → 在 `mismatch_api_record.md` 记录 → 写 `unmatch_*.cpp` 复现
2. **Paddle 主仓产出修复 PR**(如 #78099 abs、#78553 Event、#78552 arange、#78554 resize、#78584 CUDAContext、#78652/#78902 stream、#78808 CUDAGuard、#78809 weak_use_count、#78826 MaybeResetHolder)
3. **测试仓更新** → 把 `unmatch_*` 改名为正式 `*Test.cpp`,从 mismatch 表移除记录

这一闭环 **在工作期间运行了大约 6 轮**,每次几个差异点。最终 **#46 PR 描述里那张 70+ 行 MATCH 表** 就是这个闭环的成果体现。
