//! Compile smoke for the generated warp-level bindings.
//!
//! Zig compiles lazily: a generated wrapper nobody calls is never type-checked,
//! let alone lowered. This file references every wrapper of the six warp
//! families in src/gen (redux 16, warp_match 4, vote 4, active_mask 1,
//! warp_barrier 1, warp_shuffle 12 — 38 decls total, 36 reachable; the two
//! match.all forms are not, see matchShuffleSmoke) so `zig build kernels`
//! proves they all compile, and the CI grep assertions prove they reach PTX.
//! The kernel is built for sm_100a because LLVM gates `redux.sync.*.f32` on it.
//!
//! The calls are formally correct, not a working algorithm: nothing here is a
//! sequence you could run. warp_reduce covers real shuffle semantics; this
//! covers the bindings.
const cuda = @import("cuda");
const gen = cuda.gen;
const asmgen = cuda.asm_gen;
const abi = @import("examples_abi").warpops_smoke;

const full_mask: i32 = -1; // 0xffff_ffff, every lane

/// redux (16), vote (4), active_mask (1), warp_barrier (1).
pub fn reduxVoteSmoke(src: [*]const i32, diag: [*]u32) callconv(.kernel) void {
    const v: i32 = src[0];
    const f: f32 = @floatFromInt(v);
    const pred = v > 0;
    var acc: u32 = 0;

    // redux, integer ops (8): add/min/max and the bitwise three, u32 min/max.
    acc +%= @bitCast(gen.warp.redux_sync_add(v, full_mask));
    acc +%= @bitCast(gen.warp.redux_sync_min_i32(v, full_mask));
    acc +%= @bitCast(gen.warp.redux_sync_max_i32(v, full_mask));
    acc +%= @bitCast(gen.warp.redux_sync_min_u32(v, full_mask));
    acc +%= @bitCast(gen.warp.redux_sync_max_u32(v, full_mask));
    acc +%= @bitCast(gen.warp.redux_sync_and(v, full_mask));
    acc +%= @bitCast(gen.warp.redux_sync_or(v, full_mask));
    acc +%= @bitCast(gen.warp.redux_sync_xor(v, full_mask));

    // redux, f32 ops (8): min/max × {plain, abs} × {NaN-propagating, not}.
    acc +%= @as(u32, @bitCast(gen.warp.redux_sync_min_f32(f, full_mask)));
    acc +%= @as(u32, @bitCast(gen.warp.redux_sync_max_f32(f, full_mask)));
    acc +%= @as(u32, @bitCast(gen.warp.redux_sync_min_abs_f32(f, full_mask)));
    acc +%= @as(u32, @bitCast(gen.warp.redux_sync_max_abs_f32(f, full_mask)));
    acc +%= @as(u32, @bitCast(gen.warp.redux_sync_min_nan_f32(f, full_mask)));
    acc +%= @as(u32, @bitCast(gen.warp.redux_sync_max_nan_f32(f, full_mask)));
    acc +%= @as(u32, @bitCast(gen.warp.redux_sync_min_abs_nan_f32(f, full_mask)));
    acc +%= @as(u32, @bitCast(gen.warp.redux_sync_max_abs_nan_f32(f, full_mask)));

    // vote (4).
    acc +%= @intFromBool(gen.warp.any_sync(full_mask, pred));
    acc +%= @intFromBool(gen.warp.all_sync(full_mask, pred));
    acc +%= @intFromBool(gen.warp.uni_sync(full_mask, pred));
    acc +%= @bitCast(gen.warp.ballot_sync(full_mask, pred));

    // active_mask (1), warp_barrier (1).
    acc +%= @bitCast(gen.warp.active_mask());
    gen.warp.sync_mask(full_mask);

    diag[0] = acc;
}

/// warp_match (2 of 4), warp_shuffle (12: 8 NVVM i32/f32 + 4 asm u64).
pub fn matchShuffleSmoke(src: [*]const i32, diag: [*]u32) callconv(.kernel) void {
    const v: i32 = src[0];
    const w: i64 = @as(i64, v) << 32 | v;
    const f: f32 = @floatFromInt(v);
    var acc: u32 = 0;

    // warp_match: any returns the peer mask. match_all_sync /
    // match_all_i64_sync are omitted: they return {i32, i1}, and Zig's NVPTX
    // backend cannot lower that aggregate ("Do not know how to promote this
    // operator!", any target).
    acc +%= @bitCast(gen.warp.match_any_sync(full_mask, v));
    acc +%= @bitCast(gen.warp.match_any_i64_sync(full_mask, w));

    // warp_shuffle, NVVM i32/f32 forms (8): (mask, value, delta/lane, clamp).
    acc +%= @bitCast(gen.warp.shuffle_down_sync(full_mask, v, 1, 31));
    acc +%= @bitCast(gen.warp.shuffle_up_sync(full_mask, v, 1, 0));
    acc +%= @bitCast(gen.warp.shuffle_xor_sync(full_mask, v, 1, 31));
    acc +%= @bitCast(gen.warp.shuffle_sync(full_mask, v, 0, 31));
    acc +%= @as(u32, @bitCast(gen.warp.shuffle_down_f32_sync(full_mask, f, 1, 31)));
    acc +%= @as(u32, @bitCast(gen.warp.shuffle_up_f32_sync(full_mask, f, 1, 0)));
    acc +%= @as(u32, @bitCast(gen.warp.shuffle_xor_f32_sync(full_mask, f, 1, 31)));
    acc +%= @as(u32, @bitCast(gen.warp.shuffle_f32_sync(full_mask, f, 0, 31)));

    // warp_shuffle, asm u64 forms (4): (value, delta/lane, mask) as u32 lanes.
    const u: u64 = @bitCast(w);
    acc +%= @truncate(asmgen.warp.shuffle_down_u64_sync(u, 1, 0xffff_ffff));
    acc +%= @truncate(asmgen.warp.shuffle_up_u64_sync(u, 1, 0xffff_ffff));
    acc +%= @truncate(asmgen.warp.shuffle_xor_u64_sync(u, 1, 0xffff_ffff));
    acc +%= @truncate(asmgen.warp.shuffle_u64_sync(u, 0, 0xffff_ffff));

    diag[0] = acc;
}

comptime {
    cuda.abi.assertMatches(abi.signature, @TypeOf(reduxVoteSmoke));
    cuda.abi.assertMatches(abi.signature, @TypeOf(matchShuffleSmoke));
    _ = cuda.Keep(.{ &reduxVoteSmoke, &matchShuffleSmoke }).__zoxide_keep_kernels;
}
