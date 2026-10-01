# 下一次 GPU pod 运行清单(挂起验证欠账总表)

> 2026-09-30。P0–P3 全部落地,但所有需要真实硬件的验证都挂起。
> 本文是下次拿到 H20 pod(sm_90)时的执行顺序:按「能解锁后续决策」排序,
> 每项写明跑什么、证明什么、失败时去哪查。

## 部署

```sh
zig build -Dtarget=x86_64-linux-gnu.2.28 -Doptimize=ReleaseSmall --prefix dist
# 二进制内嵌全部示例 PTX(a7046c7),只需拷 dist/bin/zoxide 一个文件 + scripts/
./scripts/pod-verify.sh ./zoxide ./kernels   # kernels/ 可选;内嵌 stem 可直接跑
```

## 0. pod-verify 全量(第一道口子)

`./scripts/pod-verify.sh ./zoxide ./kernels`(不带 --quick)。
自动覆盖:4 示例 + wgmma/dev_global/const_bank/const_vs_ldg + **tma_smoke +
tma_s2g_smoke(2f/2g 节)** + sgemm_swz + hgemm_wgmma×3 + intrinsics_smoke ptxas。
任何 FAIL 先按下面对应条目查,不要直接改代码。

## 1. tma_smoke / tma_s2g_smoke(解锁:TMA 线 + prefetch + g2s 全维度)

- 证明:mbarrier+async proxy fence 修复是否生效(tma_smoke 曾挂死 4 次);
  s2g 指令形式 ptxas 是否接受;`.read` drain 语义;prefetch-then-load 路径;
  gather4 手写 asm 的 ptxas 接受性(compile-covered,未真机)。
- 挂死时:`scripts/tma-bisect.sh` 二分 barrier 层 vs 拷贝层;
  kernel 内有 stage marker,stderr 会指出挂在哪一步。
- tma_s2g_smoke 比对的是整张量(tile 内精确 + tile 外 0xAA 哨兵),
  哨兵坏了 = 描述符坐标被硬件忽略。

## 2. hgemm_wgmma4 真机首跑(解锁:`.reg` 绕法代价数据 → tcgen05 路线决策)

- `zoxide bench hgemm_wgmma4 --n 4096 --iters 5 --arch sm_90a`
- 第一件事看 ptxas 报告的 local spill(`--maxrregcount` sweep 如需要):
  有 spill → 直接记录为「.reg 路线代价数据」,不为它调优(plan 原文)。
- 无 spill 且结果精确 → 与 wgmma3(95.3 TFLOPS / 64.4%)对比,
  判定「n16 单指令成本」假设是否被宽 N 摊薄。

## 3. hgemm_tma(H2:ptxas regs)

- H1 已被 PTX 级数据否定(144→135),S3 吞吐测量已取消。
- 只剩顺手观测:`zoxide bench hgemm_tma --n 4096 --iters 5 --arch sm_90a`
  的 ptxas regs 数 vs wgmma3 的 102——下降则支持「TMA 为更宽 wgmma 腾寄存器」
  的使能条件论;结果精确性必须过。

## 4. 新 mma 形状家族(hgemm_bf16 / imma_s8 / imma_s4 / hgemm_sp / imma_sp_s8 / imma_sp_s4 / hgemm_wgmma_bf16 / hgemm_wgmma_sp)

- `zoxide bench <stem> --n 4096 --iters 5`(都是 sm_90 默认可跑,无需 --arch)。
- 全部精确校验(bf16/f16 小整数技巧,int 系零容差),任何布局/元数据解读
  错误都是硬 FAIL,不会给出貌似合理的数字——**FAIL 即信息**,对照各 kernel
  头部注释里标注的「PTX ISA 解读」段落逐条复核。
- 已知最可能错的点(按嫌疑排序):
  1. **hgemm_wgmma_sp 元数据的线程映射**(新增,嫌疑升至首位):映射读自 PTX ISA
     的 Figure 175 **图片**(文本抽取丢失,我渲染 PDF 页面读图),lane 4g+t
     供行 w*16+g 与 g+8、t0 低半/t1 高半 chunk——读图错误无文本可校验;
     另有二层:稀疏 A descriptor 的 stride 语义按"与 dense k16 tile 同字节形"
     推的,若错则全 tile 错。精确校验,硬 FAIL。
  2. hgemm_sp 元数据位序/行配对/贡献线程(纯文档解读,无硬件佐证)
  3. imma_sp_s8 / imma_sp_s4 元数据的行→线程配对(k32-s8 与 k64-s4 的 selector
     都选**线程对** T0/T1,我读作 T0=行 g、T1=行 g+8;换错则整 tile 错,硬 FAIL)
     ——与 (1) 同为纯文档解读,嫌疑并列最高。imma_sp_s4 额外多一层:4:8
     **pair 聚簇**(存活单位是 2 宽子块而非单元素),sub-chunk 索引读错同样硬 FAIL
  4. imma_s4 的 s4 片段 k 分布(k=8t..8t+7,第二寄存器 +32)
  5. imma_s8/imma_s4/imma_sp_s8 的 B 手工收集路径(ld.shared.b8 打包顺序)
  6. hgemm_wgmma_bf16 是**低风险**项:descriptor/fragment/流水线全部与已验证的
     f16 wgmma3 逐字节同构,新内容只有指令后缀;要跑的命令注意 `--arch sm_90a`
- 峰值占比读数口径:bf16 对 148T(dense FP16);imma_s8 对 296 TOPS;
  imma_s4 对 592T(**assumed**,2×INT8,无官方数);hgemm_sp 对 296T(**assumed**);
  imma_sp_s8 对 592 TOPS(**assumed**,sparse = 2× dense INT8);
  imma_sp_s4 不报峰值比(2× 于一个本身假设的 INT4 上限,假设的平方,只报 GOPS);
  hgemm_wgmma_bf16 对 148T(与 FP16 同一份规格表,bf16 同率)。

## 5. launch bounds 的 ptxas 接受性(hgemm_bf16 顺带覆盖)

- hgemm_bf16 的 PTX 现在含 `.maxntid 128;`(kernel 体中部,prologue 之后)。
  ptxas 若拒绝该放置位置,第 4 条的 hgemm_bf16 会以 InvalidPtx 失败——
  修法是把发射位置挪到 entry 头部紧邻处,不是删功能。

## 6. sanitize 包装的真实调用

- `zoxide sanitize --tool memcheck -- ./zoxide run vector_add`
- 验证 exec 形态对真实 compute-sanitizer 可接受(本机只有 fake-binary 测试)。

## 报告格式

每项结果追加到 `docs/verification/`,文件名 `2026-XX-XX-h20-pod-<主题>.md`,
沿用 2026-09-18 那份的结构(环境行、原始输出、结论、意外发现)。
ROADMAP 未发布表格对应行的「证据层级」列随之从 PTX 级升级为真机。
