# TMA 线:判据与分期

写在开工前。这一轮在 constant memory 上连错三次(`.visible` 多余、
constant memory「不可达」、`.const`「更快」),三次都是把推理或文档当成测量。
TMA 的体量比那条线大得多,所以判据先立,且尽量立成**能在没有 GPU 的机器上就否掉**的形式。

## 起点事实(已实测,非推测)

`hgemm_wgmma3` 停在 **64.4% FP16 峰值**(95260 GFLOPS,n=4096,1.443 ms),
102 regs / 18432 B shared / 4 blocks/SM / 25% occupancy。

PTX 成本结构:

| | 条数 |
|---|---|
| `cp.async`(真正的载入) | 14 |
| `wgmma.mma_async`(真正的计算) | 16 |
| **地址运算**(`add.s64` 68 + `shl/and/or` 72 + `mul.wide` 4) | **144** |
| `mov` | 150 |
| 总语句 | 526 |

已被实测**排除**的瓶颈假设(v0.0.9 那轮,五个候选全排除):
流水线排空、全局流量、occupancy 并发、共享内存操作数饥饿。
L2 边界 sweep 明确排除了全局流量。剩余嫌疑收敛到 **wgmma n16 单指令成本**。

寄存器压力已实测极其敏感:98 regs 最优;88 regs 溢出 64 B → 0.55x;
1 个字节的 local spill 就值 7.3% 吞吐。

## 假设与预言

**这一段的作用是:如果预言不成立,就不要继续往下做。**

### H1 — TMA 大幅削减地址运算(可在无 GPU 上验)

TMA 把地址生成搬进硬件:`cp.async.bulk.tensor` 只吃一个描述符指针 + 若干坐标,
不需要在寄存器里算 swizzle 和偏移。

- **预言**:地址运算类指令(add.s64 / shl / and / or / mul.wide)**降 50% 以上**,
  即从 144 降到 70 以下。
- **反向对照**:若降幅不足,说明地址运算不是来自载入路径(可能来自 C tile 写回或
  wgmma 描述符构造),TMA 改的地方不对,**停止**。
- 这一条**不需要 GPU**,是最便宜的门。

### H2 — 地址运算的减少转化为寄存器数下降

- **预言**:`ptxas` 报告的 regs/thread 从 102 下降。任意下降都算,因为 25% occupancy
  下 4 blocks/SM 的台阶在 104 regs 附近(实测 cap 104 → 4 blocks,cap 96 → 5 blocks)。
- **需要 GPU**(ptxas)。
- **注意**:这不等于变快。见 H3。

### H3 — 寄存器下降转化为吞吐(**最可能为假**)

这是要说清楚的一条:**先前的证据不支持 TMA 会提升吞吐**。
全局流量已被排除为瓶颈,剩余嫌疑是 wgmma n16 的单指令成本——TMA 不碰这个。

- **预言(我倾向于空结果)**:吞吐变化在 ±3% 以内。
- 若真的显著提升,说明「n16 单指令成本」这个收敛结论有问题,那是个更重要的发现,
  要回头重查 v0.0.9 的排除逻辑。
- 若显著下降,TMA 的额外同步(mbarrier)成本超过了省下的地址运算,记录并保留 cp.async 版。

**空结果是可接受的结果**,并且要写进文档:那意味着 TMA 的价值是
「代码更短、寄存器更省、为更宽的 wgmma 让路」,而不是它本身更快。

### H4 — cluster multicast 降 L2→SM 流量(**预先声明预期为空**)

同一列的多个 CTA 读同一个 B tile,cluster multicast 可以一次载入广播给整个 cluster。

- **预言:空结果**。L2 边界 sweep 已排除全局/L2 流量作为瓶颈。
- 这条**先不做**。只有 H2 成立且 H3 非空时才有理由回来做它——否则就是在优化
  一个已经证明不是瓶颈的东西(和早先撤销的「M=128 降全局流量」同一个错)。

## 防自欺规则(从本轮三次错误里提炼)

1. **先做反向对照,再写绕法。** `.visible` 那次是反过来的:先写了 pass、单元测试、
   构建集成、跨平台工具,再想起来做对照,结果对照否掉了全部工作。
2. **正确性检查必须和计时放在一起。** `trips` 漂移那次,计时会照常给出完全合理的
   数字,只有正确性检查拦住了它。
3. **对照必须要求那条具体错误**,不能接受「没 PASS 就算符合预期」。第一版
   `.visible` 对照因为文件名 stem 不匹配而在无关位置失败,却报了「as expected」。
4. **主机与设备共享的常量放 `examples_abi.zig`**,让漂移变成编译错误而不是错值。
5. **A/B 必须形状对齐。** `.const` 那次 LLVM 给出 4× 和 8× 两种展开因子,
   不强制对齐就是在测展开器。
6. **判据阈值写进代码**,不留在脑子里,避免看到数之后往回凑解释。
7. **预期为空的假设要预先声明为空**(H3/H4),这样空结果不会被重新包装成成功。

## 分期

### S0 — 前提:host 侧描述符 + device 侧 g2s(无 GPU 可完成)

**订正**:本文初稿写「device 侧 wrapper 齐备,`cp_async_bulk_tensor` 138 条」。
那是错的——我数了前缀匹配却没看**方向**。138 条全是 `s2g`(shared→global)和
`reduce_*`。GEMM 需要的 `g2s`(global→shared)**一条都没有**:

```
$ grep -c g2s src/gen/intrinsics.zig src/gen/instrinsics_asm.zig
0
0
```

LLVM 没有暴露这个方向的 intrinsic,生成器也就没产出。所以和
`wgmma.mma_async` 一样必须**手写 inline asm**。

两个硬阻塞:
1. host: `cuTensorMapEncodeTiled` 绑定数 0
2. device: `cp.async.bulk.tensor.*.shared::cluster.global.tile` 手写 asm

可复用的:`mbarrier_arrive_expect_tx` / `mbarrier_try_wait` / `fence_proxy_async`
共 16 条 asm wrapper 已生成。

产出:绑定 + 类型化描述符构造(维度/元素类型/swizzle 模式做成 enum,
让非法组合变编译错误而不是 `CUDA_ERROR_INVALID_VALUE`)。

### S1 — 最小 TMA smoke(需 GPU,不谈性能)✅ 已实现,待跑

`tma_smoke`:128×128 f16 张量,取 (64,16) 处的 64×8 tile(**故意不取原点**
——忽略坐标的描述符在原点会通过)。共享内存线性读出到 global,host 逐字节比对。

对照用两个描述符跑同一个 kernel:

| swizzle | 预期 |
|---|---|
| `.none` | 共享内存就是行主序 tile,线性读出**必须逐字节相符** |
| `.b128` | 硬件置换 16 字节 chunk,同样的读出**必须不同** |

第二条是对照。两者相同则说明 swizzle 字段没到硬件,那第一条也就证明不了
「描述符在驱动这次拷贝」——和 `const_bank` 里 `pad_before` 补的是同一个洞。

内层 tile 宽度取 64×2 = **128 字节 = 恰好一个 `.b128` 周期**,让对照干净。

`chunk_index ^ row` 的置换模型只**报告不判定**:那是我对 swizzle 的理解,
模型错不该让一个主张在别处的测试失败。

`expect_tx` 的字节数从 `abi.tma_smoke.tile_bytes` 来,并与
`map.tileBytes()` 交叉校验 —— 这个数写错不会 fault:偏小则 wait 在残缺数据上放行,
偏大则永不放行。

**第一次跑挂死了。** 已排查并修两处:

1. **漏了 `fence.proxy.async.shared::cta`**。我写了 `fenceProxyAsync()` 却从没调用。
   `mbarrier.init` 之后必须有它:async proxy(拷贝引擎)是独立的内存消费者,
   不保证观察到已初始化的 barrier。CUTLASS 的顺序也是 init → fence → issue。
   **这是最可能的原因,但不是已确认的原因**——逐条排查过哨兵地址空间、描述符
   对齐、tile 越界、`_` sink、barrier 对齐,都成立。
2. **无界自旋改成有界**(`tryWaitFor`)。这条比 1 更重要:GPU 上死循环只能杀进程、
   零诊断。现在放弃轮询后 kernel 写 `0xBA` 哨兵,host 认出来并报
   「mbarrier never completed」,把「不可诊断的挂死」变成「可报告的失败」。
   探针另加 shell 层 `timeout 60`,区分「in-kernel 轮询耗尽」和「launch 本身不返回」。

教训:**任何设备侧等待在测试里都必须有界**。这和「正确性检查要和计时放在一起」
是同一条原则——失败必须能说出自己是什么。

### S2 — TMA 版 hgemm(需 GPU)
tile 形状与 `hgemm_wgmma3` **完全一致**(m64n128k16 三级流水),只替换载入路径。
H1 在这一步用 PTX 判定(无 GPU 即可),H2 用 ptxas 报告判定。

### S3 — 吞吐测量(需 GPU)
仅在 H1 成立时进行。与 `hgemm_wgmma3` 同 n、同 iters、同正确性检查。
H3 的 ±3% 判据编译进 harness。

### S4 — cluster multicast
**暂不排期**,理由见 H4。

## 与 `hgemm_wgmma4` 的关系

两者攻击**不同**的东西,不要混为一谈:

- `hgemm_wgmma4`(m64n128k16,一条 wgmma 覆盖 N=128)直接攻击 n16 单指令成本
  ——即当前收敛到的那个嫌疑。**它一直未跑**(release `wide-20260922`)。
- TMA 攻击载入路径的地址运算成本,而载入流量已被排除为瓶颈。

所以按证据排序,`hgemm_wgmma4` 的期望价值高于 TMA。TMA 的主要价值更可能是
「腾出寄存器,为更宽的 wgmma 让路」——也就是说它可能是 `wgmma4` 的**使能条件**
而不是独立的优化。这个关系在 S2 的寄存器数上就能看出来。
