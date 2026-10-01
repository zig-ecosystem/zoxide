# AGENTS.md

给在本仓库工作的协作者(人或 agent)的约定。细节一律以 docs/ 原文为准,这里只做索引和硬规则。

## 工作方式

- **测量优先于推理。** 性能结论必须有真机数据;PTX 级结论必须给出 grep 证据。
  本仓库历史上三次连错(见 docs/tma-plan.md 防自欺规则)都来自把推理当测量。
- **判据先立,阈值写进代码或文档**,不接受看到数之后往回凑解释。
- **预期为空的假设预先声明为空**,空结果不得重新包装成成功。
- **反向对照先于绕法。** 新机制的对照必须要求那条具体错误,「没 PASS」不算符合预期。
- **正确性检查和计时放一起**;单独计时的漂移不会被发现。
- **host 与 device 共享的常量/签名放 `src/examples_abi.zig`**,让漂移成为编译错误。
- **A/B 必须形状对齐**(同一编译器、同一展开因子、同一除法路径)。
- **多层机制出问题先二分层**,不要连着猜同一层。
- **设备侧等待必须有界**;GPU 死循环零诊断。

## 文档政策

- **保留原文 + 订正框。** 文档和 announcement 里的错误结论不删除,加带日期的
  「订正」框指向新证据。先例:docs/announcement.md 的 v0.0.4/v0.0.7/v0.0.12 条目。
- ROADMAP.md 的版本表是唯一进度真相源;已发布版本必须有 git tag +
  announcement 条目 + GitHub release asset(tarball 含 zoxide + kernels/ + scripts/)。
- 真机证据进 `docs/verification/`;下次 GPU pod 的执行顺序见
  `docs/verification/next-pod-run.md`。

## 代码

- 设备端库是 freestanding 的(无 host std);能用 Zig 内建就不用 wrapper
  (f16 算术、@atomicRmw),wrapper 只覆盖 Zig 没有语法的东西。
- `src/gen/` 是生成物,不手改;例外必须在文件头注明可复现的转换规则
  (先例:mbarrier u32 收窄)。
- 新 kernel:注册进 build.zig 列表 + examples_abi.zig 签名 + CI 的 PTX grep 断言
  (防上游回退)+ pod-verify 或 next-pod-run 的真机位置,四件一起做。
- 提交信息单行概括 + 必要时正文写证据;版本语义见 ROADMAP。

## 验证

- 本机无 GPU 是常态:验收 = `zig build` + `zig build kernels` + `zig build test`
  + 完整 CI PTX 断言块 + pod-verify 降级模式(SKIP 数变动要同步 ci.yml 的断言)。
- 任何「PTX 级验证通过」的表述必须同时注明真机挂起。
