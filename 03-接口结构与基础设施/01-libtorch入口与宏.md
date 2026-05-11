# 01 · libtorch 入口与宏

> 范围:`torch/torch.h` 顶级入口头文件、`TORCH_WARN` / `STD_CHECK` / `STD_TORCH_CHECK` / `TORCH_CHECK_OP` 系列宏、`torch::headeronly` 命名空间下的纯 header 接口、代码组织重构(`Utils.h` → `Utils.cpp`、Tensor methods → `ops/`、文件改名)。

## 1. 主题概述

PyTorch C++ API 的入口约定:

```cpp
#include <torch/torch.h>  // 顶级入口,等价于 "全部 include"
#include <ATen/ATen.h>    // ATen 子命名空间
#include <c10/util/Exception.h>  // 异常宏 TORCH_CHECK / TORCH_WARN
```

下游库根据需要 include 这些头。compat 层提供与 PyTorch 同路径的"门面"头文件,在内部只 include 它真正需要的子头(避免把整个 Paddle build 拖进 include 树)。

宏命名同样需要严格对齐 — PyTorch 用 `TORCH_CHECK(cond, "msg")` 抛 `c10::Error`,用 `TORCH_WARN(...)` 打 warning,用 `STD_CHECK(cond, msg)` 检查标准库一致性。下游代码不会改这些宏名,所以 compat 层必须 1:1 提供。

`torch::headeronly` 是一个有趣的设计 — 把纯计算 / 纯模板 / 纯枚举的接口抽到一个**完全不依赖 cpp 实现**的子命名空间,这样**纯 header 库**(典型如 DeepGEMM 的 C++ template kernel)可以只 include 这部分,而不需要把 compat 层的 cpp 编译产物链进来。

## 2. PR 汇总表

| PR# | 状态 | 合并日期 | 标题 |
|---:|---|---|---|
| [#77854](https://github.com/PaddlePaddle/Paddle/pull/77854) | MERGED | 2026-04-02 | Add `torch/torch.h` |
| [#78244](https://github.com/PaddlePaddle/Paddle/pull/78244) | MERGED | 2026-03-12 | move implementation from `Utils.h` to `Utils.cpp` |
| [#78576](https://github.com/PaddlePaddle/Paddle/pull/78576) | MERGED | 2026-04-07 | Add `TORCH_WARN` macro and fix resize_ |
| [#78580](https://github.com/PaddlePaddle/Paddle/pull/78580) | MERGED | 2026-04-07 | Delete useless code and rename test files |
| [#78641](https://github.com/PaddlePaddle/Paddle/pull/78641) | MERGED | 2026-04-14 | add `STD_CHECK` macro for paddlecodec |
| [#78662](https://github.com/PaddlePaddle/Paddle/pull/78662) | MERGED | 2026-04-15 | Add some headeronly APIs |
| [#78266](https://github.com/PaddlePaddle/Paddle/pull/78266) | MERGED | 2026-03-23 | Rename test `compat_squeeze_test` → `ATen_squeeze_test` |

## 3. 关键 PR 深度分析

### 3.1 `torch/torch.h` — 项目的"hello world" 顶级入口 — #77854(+22 行,1 个文件)

#### 背景

hybrid_ep(多模态算子库)的编译只 `#include <torch/torch.h>`,然后用 `at::Tensor` / `c10::ScalarType` 等接口。它不在乎实现细节,只要这一个顶级入口能解析所有引用。

之前 compat 层没有 `torch/torch.h`,hybrid_ep 编译直接报"找不到该头文件"。

#### 改动

新建 `paddle/phi/api/include/compat/torch/csrc/api/include/torch/torch.h`(+22 行):

```cpp
#pragma once

// 顶级入口:等价于 PyTorch 的 torch/torch.h
#include <ATen/ATen.h>
#include <c10/core/Device.h>
#include <c10/core/ScalarType.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorOptions.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Exception.h>
// ...
```

#### 效果

hybrid_ep 编译过。**这个 PR 是项目早期的标志性里程碑** — 第一次"下游库无改动直接编译过"的实例。

### 3.2 `TORCH_WARN` — #78576(+86/-4 行,2 个文件)

#### 背景

DeepEP 在 `csrc/deep_ep.cpp` 中用 `TORCH_WARN("some message")` 提示 deprecation。compat 层最初没有这个宏,导致 DeepEP 编译报 `'TORCH_WARN' was not declared`。

同时这个 PR 还附带修复了 `resize_` 接口编译错误(对 `phi::DenseTensor::offset()` / `mutable_data()` 的依赖,详见 [`01-基础类型与Tensor/02-Tensor接口与操作.md`](../01-基础类型与Tensor/02-Tensor接口与操作.md) 第 3.5 节)。

#### 改动

`c10/util/Exception.h`(+81 行)中新增:

```cpp
// 类似 PyTorch 的 TORCH_WARN:打印 warning 但不抛异常
#define TORCH_WARN(...)                                         \
  ::c10::_PD_Internal::warn(                                    \
      ::c10::SourceLocation{__func__, __FILE__, (uint32_t)__LINE__}, \
      ::c10::detail::torchCheckMsgImpl(__VA_ARGS__))

#define TORCH_WARN_ONCE(...)                                    \
  do {                                                           \
    static bool warned = false;                                  \
    if (!warned) {                                               \
      TORCH_WARN(__VA_ARGS__);                                   \
      warned = true;                                             \
    }                                                            \
  } while (0)

#define TORCH_WARN_DEPRECATION(...) TORCH_WARN("DEPRECATION: ", __VA_ARGS__)
```

支持变参数、source location 信息、`_ONCE` / `_DEPRECATION` 变体。后端 warning sink 复用 Paddle 的日志体系(`VLOG(0)` + 控制台输出)。

#### 效果

DeepEP 编译通过。后续 #78609 / #78633 中 `resize_` 边界条件的 warning 也用这个宏。

### 3.3 `STD_CHECK` — #78641(+55 行,1 个文件)

#### 背景

paddlecodec(多模态编解码库)使用 `STD_CHECK(condition, "message")` 做参数校验。这是 PyTorch 体系中**简化版的 `TORCH_CHECK`** — 不抛 `c10::Error`,而是抛 `std::runtime_error`,适用于不依赖 c10 异常体系的代码路径。

#### 改动

`c10/util/Exception.h`(+55 行)新增:

```cpp
#define STD_CHECK(cond, ...)                                    \
  do {                                                           \
    if (!(cond)) {                                               \
      throw std::runtime_error(                                  \
          ::c10::detail::torchCheckMsgImpl(__VA_ARGS__));        \
    }                                                            \
  } while (0)

#define STD_CHECK_EQ(a, b, ...)  STD_CHECK((a) == (b), __VA_ARGS__)
#define STD_CHECK_NE(a, b, ...)  STD_CHECK((a) != (b), __VA_ARGS__)
#define STD_CHECK_LT(a, b, ...)  STD_CHECK((a) < (b), __VA_ARGS__)
#define STD_CHECK_LE(a, b, ...)  STD_CHECK((a) <= (b), __VA_ARGS__)
#define STD_CHECK_GT(a, b, ...)  STD_CHECK((a) > (b), __VA_ARGS__)
#define STD_CHECK_GE(a, b, ...)  STD_CHECK((a) >= (b), __VA_ARGS__)
```

`STD_TORCH_CHECK` 在 #78662 中作为这族宏的姊妹引入:

```cpp
#define STD_TORCH_CHECK(cond, ...)  \
  TORCH_CHECK(cond, __VA_ARGS__)  // 别名,语义同 TORCH_CHECK
```

#### 效果

paddlecodec 编译通过。

### 3.4 `torch::headeronly` 命名空间 — #78662(+912/-561 行,8 个文件)

#### 背景

DeepGEMM 等高性能 kernel 库的特性是:**整个库都是 C++ template,放在 header-only**。它们 `#include <c10/core/ScalarType.h>` 后,如果 compat 层把 `ScalarType` 相关实现写在 cpp 里(意味着用户必须 link 到 Paddle 的 .so),就会破坏 header-only 模型。

#### 改动思路

把 `ScalarType` / `DeviceType` / `TensorAccessor` / `Exception` 中**纯枚举 + 模板 + 内联 helper** 的部分,搬到独立的 `torch/headeronly/` 子目录,完全不依赖任何 cpp 实现:

```
paddle/phi/api/include/compat/torch/headeronly/
├── core/
│   ├── DeviceType.h         (+58 行)
│   ├── ScalarType.h         (+316 行)
│   └── TensorAccessor.h     (+399 行)
└── util/
    └── Exception.h          (+85 行,纯宏定义)
```

`c10/core/ScalarType.h`、`c10/core/DeviceType.h`、`ATen/core/TensorAccessor.h` 改为**只做转发**:

```cpp
// c10/core/ScalarType.h
#pragma once
#include <torch/headeronly/core/ScalarType.h>
namespace c10 {
using ::torch::headeronly::ScalarType;
using ::torch::headeronly::isFloatingType;
// ... 等
}  // namespace c10
```

`c10/util/Exception.h` 同理(+1/-54 行,大部分宏定义搬到 headeronly)。

#### 同时引入 `STD_TORCH_CHECK`

PR 描述说"补充 `STD_TORCH_CHECK` 宏"。该宏是 `STD_CHECK` + `TORCH_CHECK` 的合体:

```cpp
#define STD_TORCH_CHECK(cond, ...)  \
  if (!(cond)) { \
    throw ::c10::Error({__func__, __FILE__, (uint32_t)__LINE__}, \
                       ::c10::detail::torchCheckMsgImpl(__VA_ARGS__)); \
  }
```

`headeronly/util/Exception.h`(+85 行)承载完整异常机器的纯 header 实现。

#### 效果

DeepGEMM 等 header-only 库可以只引用 `torch/headeronly/`,不需 link 到 Paddle 主库 — 但代价是,引用了 headeronly 之后,如果再 include `c10/util/Exception.h` 等就有了重复定义风险,后续 #78580 中做了一次清理。

### 3.5 `Utils.h` → `Utils.cpp` 实现搬迁 — #78244(+102/-64 行,3 个文件)

#### 背景

[#78239 review 中 SigureMo 的评论](https://github.com/PaddlePaddle/Paddle/pull/78239#discussion_r2909210128) 提到:`compat/ATen/Utils.h` 中的部分 helper 用了 inline 内联实现,但在多 TU 中可能触发重复定义(ODR 违规)。

具体表现:`Utils.h` 中定义了:

```cpp
// compat/ATen/Utils.h(旧)
inline void some_helper() { /* 大段实现 */ }
```

当多个 cpp 都 include 这个头并各自实例化 `some_helper` 时,链接器会报多重定义。

#### 改动

新建 `compat/ATen/Utils.cpp`(+96 行),把 `Utils.h` 中所有非模板的 inline 实现搬到 cpp。`Utils.h` 只保留**声明 + 模板实现**(+5/-64 行)。

`compat/CMakeLists.txt`(+1) — 把 `Utils.cpp` 加入编译目标。

#### 后续

`#78580`、`#78670`(Windows)中,类似的搬迁还做了几次 — 比如 `DefaultDtype.h` → `DefaultDtype.cpp`(在 #78670 中)、`CUDAStream.h` 大量内联实现搬到 `CUDAStream.cpp`(在 #78584 中)。每次搬迁都是为了消除一个具体的 ODR 风险或减少头文件膨胀。

### 3.6 测试文件改名清理 — #78266、#78580

#### 历史

compat 测试文件早期叫 `compat_basic_test.cc`、`compat_squeeze_test.cc`、`compat_toString_test.cc` — 用 `compat_` 前缀区分。随着测试规模扩大,导师建议:**按所测试 API 的命名空间命名**,即 `ATen_*` 或 `c10_*`,更直观。

#### #78266(+1/-1 行,2 文件)

把 `compat_squeeze_test.cc` 改名为 `ATen_squeeze_test.cc`。仅 1 行 CMakeLists 修改。**这个 PR 只做一个文件的改名,因为后续要改的还有十几个,留给 #78580 集中处理**。

#### #78580(+294/-288 行,37 文件)

集中改名:

| 旧名 | 新名 |
|---|---|
| `compat_basic_test.cc` | `ATen_basic_test.cc` |
| `compat_toString_test.cc` | `ATen_toString_test.cc` |
| `compat_dense_sparse_conversion_test.cc` | `ATen_dense_sparse_conversion_test.cc` |
| `compat_squeeze_test.cc` | (已在 #78266 中改名) |

同时:

- 清理 `c10_storage_test.cc` 中预处理无用代码(+11/-6)
- 抑制未使用变量警告:在 `ATen_clamp_test.cc`、`ATen_index_test.cc` 等十几个测试中加 `(void)var;`
- `ATen_as_strided_test.cc` / `ATen_basic_test.cc` / `ATen_clamp_test.cc` / `ATen_index_test.cc` / `ATen_transpose_test.cc` / `ATen_viewAs_test.cc` / `c10_cuda_generator_test.cc` 等 +6~+12 行各种小修
- `c10_Event_test.cc`(+15/-9)、`c10_Stream_test.cc`(+13/-10) 整理
- `torch_library_dispatch_test.cc`(+9/-9) — 删除 dead code

这些**看起来零散**的清理工作,实际上为后续 PR 提供了**干净的工作面**。从 #78580 开始,所有 compat 测试文件名都遵循 `<namespace>_<subject>_test.cc` 命名约定。

## 4. 与 PaddleCppAPITest 的对应关系

测试仓侧没有专门的"宏测试"或"入口头测试" — 这些都是通过实际使用过程的编译来验证。

但在 #46 / #58 / #59 中,测试仓也做了**类似的命名空间组织重构**:测试文件路径从扁平的 `test/*Test.cpp` 改为 `test/ATen/...` / `test/c10/...` / `test/torch/...`,与本章的 compat 测试改名思路一致。
