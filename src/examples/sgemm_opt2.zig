const cuda = @import("cuda");
const api = @import("examples_abi");

/// Register-blocked SGEMM, optimization round 2b: double-buffered shared
/// tiles (prefetch next K-slice into registers while computing the current
/// one). Same tiling as sgemm_opt (128x128 block tile, K-slice 16, 256
/// threads, 8x8 register accumulators, vectorized loads).
///
/// Pipeline per tile kt:
///   prefetch tile kt+1 global->registers (guarded, uniform `kt+1 < ktiles`)
///   compute tile kt from shared buffer `cur`
///   syncThreads  (all reads of cur done)
///   store registers -> shared buffer `1 - cur`
///   syncThreads  (buffer 1-cur ready)
///
/// Both barriers are on the universal path; load guards are branchless
/// selects, so no barrier can be duplicated onto divergent code.

pub const block_tile = 128;
pub const k_slice = 16;

var as_buf: [2][block_tile * k_slice]f32 addrspace(.shared) = undefined; // [buf][r][k]
var bs_buf: [2][k_slice * block_tile]f32 addrspace(.shared) = undefined; // [buf][k][c]

const TileRegs = struct {
    // Kept as vectors end-to-end so LLVM does not scalarize the 128-bit loads.
    a: [2]@Vector(4, f32),
    b: [2]@Vector(4, f32),
};

fn loadRegs(a: [*]const f32, b: [*]const f32, n: u32, block_row: u32, block_col: u32, k0: u32, tid: u32) TileRegs {
    var regs: TileRegs = .{ .a = @splat(@splat(0)), .b = @splat(@splat(0)) };
    _ = &regs;
    const vec_ok = n % 4 == 0;
    inline for (0..2) |i| {
        {
            const f4 = tid * 2 + i;
            const r = f4 / (k_slice / 4);
            const q = f4 % (k_slice / 4);
            const gr = block_row + r;
            const gk = k0 + q * 4;
            if (vec_ok and gr < n and gk + 4 <= n) {
                const src: *align(16) const @Vector(4, f32) = @ptrCast(@alignCast(a + gr * n + gk));
                regs.a[i] = src.*;
            } else {
                inline for (0..4) |j| {
                    regs.a[i][j] = if (gr < n and gk + j < n) a[gr * n + gk + j] else 0;
                }
            }
        }
        {
            const f4 = tid * 2 + i;
            const k = f4 / (block_tile / 4);
            const c4 = f4 % (block_tile / 4);
            const gk = k0 + k;
            const gc = block_col + c4 * 4;
            if (vec_ok and gk < n and gc + 4 <= n) {
                const src: *align(16) const @Vector(4, f32) = @ptrCast(@alignCast(b + gk * n + gc));
                regs.b[i] = src.*;
            } else {
                inline for (0..4) |j| {
                    regs.b[i][j] = if (gk < n and gc + j < n) b[gk * n + gc + j] else 0;
                }
            }
        }
    }
    return regs;
}

fn storeShared(comptime buf: usize, regs: TileRegs, tid: u32) void {
    inline for (0..2) |i| {
        {
            const f4 = tid * 2 + i;
            const r = f4 / (k_slice / 4);
            const q = f4 % (k_slice / 4);
            const dst: *align(16) addrspace(.shared) @Vector(4, f32) =
                @ptrCast(@alignCast(&as_buf[buf][r * k_slice + q * 4]));
            dst.* = regs.a[i];
        }
        {
            const f4 = tid * 2 + i;
            const k = f4 / (block_tile / 4);
            const c4 = f4 % (block_tile / 4);
            const dst: *align(16) addrspace(.shared) @Vector(4, f32) =
                @ptrCast(@alignCast(&bs_buf[buf][k * block_tile + c4 * 4]));
            dst.* = regs.b[i];
        }
    }
}

fn computeTile(comptime buf: usize, acc: *[8][8]f32, tx: u32, ty: u32) void {
    var k: u32 = 0;
    while (k < k_slice) : (k += 1) {
        var a_frag: [8]f32 = undefined;
        var b_frag: [8]f32 = undefined;
        inline for (0..8) |jj| {
            a_frag[jj] = as_buf[buf][(ty * 8 + jj) * k_slice + k];
        }
        inline for (0..2) |g| {
            const src: *align(16) addrspace(.shared) const @Vector(4, f32) =
                @ptrCast(@alignCast(&bs_buf[buf][k * block_tile + tx * 8 + g * 4]));
            const v = src.*;
            inline for (0..4) |j| b_frag[g * 4 + j] = v[j];
        }
        inline for (0..8) |jj| {
            inline for (0..8) |ii| {
                acc[jj][ii] = @mulAdd(f32, a_frag[jj], b_frag[ii], acc[jj][ii]);
            }
        }
    }
}

pub fn sgemmOpt2(a: [*]const f32, b: [*]const f32, c: [*]f32, n: u32) callconv(.kernel) void {
    const tx = cuda.threadIdx().x;
    const ty = cuda.threadIdx().y;
    const tid = ty * 16 + tx;

    const block_row = cuda.blockIdx().y * block_tile;
    const block_col = cuda.blockIdx().x * block_tile;

    var acc: [8][8]f32 = @splat(@splat(0));

    const ktiles = (n + k_slice - 1) / k_slice;

    // Prologue: tile 0 into buffer 0.
    storeShared(0, loadRegs(a, b, n, block_row, block_col, 0, tid), tid);
    cuda.syncThreads();

    var cur: u1 = 0;
    var kt: u32 = 0;
    while (kt < ktiles) : (kt += 1) {
        const has_next = kt + 1 < ktiles; // uniform across the block
        var next_regs: TileRegs = undefined;
        if (has_next) {
            next_regs = loadRegs(a, b, n, block_row, block_col, (kt + 1) * k_slice, tid);
        }
        if (cur == 0) computeTile(0, &acc, tx, ty) else computeTile(1, &acc, tx, ty);
        cuda.syncThreads();
        if (has_next) {
            if (cur == 0) storeShared(1, next_regs, tid) else storeShared(0, next_regs, tid);
        }
        cuda.syncThreads();
        cur = 1 - cur;
    }

    inline for (0..8) |jj| {
        const gr = block_row + ty * 8 + jj;
        inline for (0..8) |ii| {
            const gc = block_col + tx * 8 + ii;
            if (gr < n and gc < n) {
                c[gr * n + gc] = acc[jj][ii];
            }
        }
    }
}

comptime {
    // Drift from the signature `zoxide bench` launches through is a compile
    // error here rather than a silently mis-packed argument list.
    cuda.abi.assertMatches(api.sgemm, @TypeOf(sgemmOpt2));
    _ = cuda.Keep(.{&sgemmOpt2}).__zoxide_keep_kernels;
}
