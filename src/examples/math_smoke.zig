const cuda = @import("cuda");
const math = cuda.math;

/// Compile smoke for src/math.zig: every tier of the device math library in
/// one kernel. CI greps the PTX for the approx instructions the fast tier
/// must lower to; the software tier is exercised so it survives DCE.
pub fn mathSmoke(out: [*]f32, in: [*]const f32, n: u32) callconv(.kernel) void {
    const gid = cuda.globalThreadId();
    if (gid < n) {
        const x = in[gid];
        out[gid] = math.sin(x) + math.cos(x) + math.atan(x) + math.cbrt(x) +
            math.exp(x) + math.log(@abs(x) + 1) +
            math.sinFast(x) + math.cosFast(x) + math.expFast(x) + math.log2Fast(@abs(x) + 1) +
            math.tanhFast(x) + math.rsqrtFast(@abs(x) + 1) +
            math.fmax(x, 0.5) + math.sqrt(@abs(x));
    }
}

comptime {
    _ = cuda.Keep(.{&mathSmoke}).__zoxide_keep_kernels;
}
