//! Compile smoke for the generated cp.async + mbarrier bindings.
//!
//! Zig compiles lazily: a generated wrapper nobody calls is never type-checked,
//! let alone lowered. This file references every wrapper of the six async-copy /
//! barrier families in src/gen (cp_async_copy, cp_async_control,
//! cp_async_mbarrier, mbarrier_basic, mbarrier_extended, counted_barrier — 35
//! decls total) so `zig build kernels` proves they all compile, and the CI
//! grep assertions prove they reach PTX.
//!
//! The calls are formally correct, not a working pipeline: nothing here is a
//! sequence you could run. mbar_smoke covers the semantics; this covers the
//! bindings.
const cuda = @import("cuda");
const gen = cuda.gen;
const asmgen = cuda.asm_gen;
const abi = @import("examples_abi").cpasync_mbar_smoke;

var smem: [64]u8 addrspace(.shared) = undefined;
var bar_mem: [2]u64 addrspace(.shared) = undefined;

inline fn barPtr() [*]addrspace(.shared) u8 {
    return @ptrCast(&bar_mem);
}

// cp.async.wait_group takes an immediate operand in LLVM (immarg); the
// generated wrapper's runtime i32 parameter cannot select. Same workaround as
// hgemm_mma2: a comptime-parameterised shim.
fn cpAsyncWaitGroup(comptime num: u32) void {
    gen.async_copy.cp_async_wait_group(num);
}

/// cp_async_copy (8), cp_async_control (3), cp_async_mbarrier (4).
pub fn cpAsyncSmoke(src: [*]const u8, diag: [*]u32) callconv(.kernel) void {
    const gsrc: [*]addrspace(.global) const u8 = @addrSpaceCast(src);
    const dst: [*]addrspace(.shared) u8 = @ptrCast(&smem);

    // cp_async_copy: ca/cg at 4/8/16 bytes plus the zfill (src-size) variants.
    gen.async_copy.cp_async_ca_4(dst, gsrc);
    gen.async_copy.cp_async_ca_8(dst, gsrc);
    gen.async_copy.cp_async_ca_16(dst, gsrc);
    gen.async_copy.cp_async_cg_16(dst, gsrc);
    gen.async_copy.cp_async_ca_zfill_4(dst, gsrc, 4);
    gen.async_copy.cp_async_ca_zfill_8(dst, gsrc, 8);
    gen.async_copy.cp_async_ca_zfill_16(dst, gsrc, 16);
    gen.async_copy.cp_async_cg_zfill_16(dst, gsrc, 16);

    // cp_async_control.
    gen.async_copy.cp_async_commit_group();
    cpAsyncWaitGroup(0);
    gen.async_copy.cp_async_wait_all();

    // cp_async_mbarrier: with and without the pending-count increment, in the
    // shared-typed and the generic-pointer form.
    gen.async_copy.cp_async_mbarrier_arrive_shared(barPtr());
    gen.async_copy.cp_async_mbarrier_arrive_noinc_shared(barPtr());
    gen.async_copy.cp_async_mbarrier_arrive(@ptrCast(@addrSpaceCast(barPtr())));
    gen.async_copy.cp_async_mbarrier_arrive_noinc(@ptrCast(@addrSpaceCast(barPtr())));

    diag[0] = smem[0];
}

/// mbarrier_basic (5), mbarrier_extended (11), counted_barrier (4).
pub fn mbarGenSmoke(src: [*]const u8, diag: [*]u32) callconv(.kernel) void {
    _ = src;
    const bar = barPtr();
    // The asm wrappers take the mbarrier address as u64 (src/tma.zig's header
    // documents why that signature is a poor fit — shared addresses are 32-bit,
    // which is why tma.zig does not use them). Usable here by passing the
    // zero-extended 32-bit shared offset; only PTX emission is asserted.
    const addr: u64 = @intFromPtr(&bar_mem[0]);

    // mbarrier_basic: NVVM init/arrive/arrive.noComplete/inval + asm test_wait.
    gen.barrier.mbarrier_init(bar, 1);
    const s0 = gen.barrier.mbarrier_arrive(bar);
    const s1 = gen.barrier.mbarrier_arrive_no_complete(bar, 1);
    var acc: u64 = @bitCast(s0 ^ s1);
    acc +%= asmgen.barrier.mbarrier_test_wait(addr, 0);
    gen.barrier.mbarrier_inval(bar);

    // mbarrier_extended.
    asmgen.barrier.nanosleep(32);
    acc +%= asmgen.barrier.mbarrier_try_wait(addr, 32);
    asmgen.barrier.fence_mbarrier_init_release_cluster();
    acc +%= asmgen.barrier.mbarrier_arrive_expect_tx(addr, 16);
    acc +%= asmgen.barrier.mbarrier_arrive_expect_tx_cluster(addr, 16);
    asmgen.barrier.fence_proxy_async_generic_release_shared_cta_cluster();
    acc +%= asmgen.barrier.mbarrier_try_wait_parity(addr, 0);
    asmgen.barrier.fence_proxy_async_generic_acquire_shared_cluster_cluster();
    asmgen.barrier.fence_proxy_async_shared_cta();
    asmgen.barrier.mbarrier_arrive_cluster(addr);
    acc +%= asmgen.barrier.mbarrier_try_wait_parity_cluster(addr, 0);

    // counted_barrier: named barrier 1, count sized to a 128-thread block.
    gen.barrier.barrier_cta_arrive(1, 128);
    gen.barrier.barrier_cta_sync(1, 128);
    gen.barrier.barrier_cta_arrive_aligned(1, 128);
    gen.barrier.barrier_cta_sync_aligned(1, 128);

    diag[0] = @truncate(acc);
}

comptime {
    cuda.abi.assertMatches(abi.signature, @TypeOf(cpAsyncSmoke));
    cuda.abi.assertMatches(abi.signature, @TypeOf(mbarGenSmoke));
    _ = cuda.Keep(.{ &cpAsyncSmoke, &mbarGenSmoke }).__zoxide_keep_kernels;
}
