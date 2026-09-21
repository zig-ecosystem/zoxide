//! Compile-time agreement between a kernel's device-side definition and the
//! host-side launch that calls it. Dependency-free comptime reflection, so both
//! the freestanding device module and the host module can import it.
//!
//! The problem it solves: `cuLaunchKernel` takes `void**`, one untyped pointer
//! per argument. Nothing checks that the host passes the number, order and width
//! of arguments the kernel actually declares, and a mismatch does not fault — the
//! kernel reads adjacent memory and returns plausible-looking wrong answers.
//! Worse, the two sides are compiled separately for different targets, so they
//! drift silently as a kernel's parameters change.
//!
//! The fix is a signature declared once in a file both sides import:
//!
//! ```zig
//! // kernels_abi.zig — imported by host and device alike
//! pub const scale = fn (x: [*]const f32, y: [*]f32, k: f32, n: u32) void;
//! ```
//!
//! The device asserts its definition matches:
//!
//! ```zig
//! pub fn scale(x: [*]const f32, y: [*]f32, k: f32, n: u32) callconv(.kernel) void { ... }
//! comptime { abi.assertMatches(api.scale, @TypeOf(scale)); }
//! ```
//!
//! and the host hands the same declaration to `Module.kernel`, which uses it to
//! check the argument tuple. Changing one side without the other is then a
//! compile error on that side rather than wrong numbers later.
//!
//! Note the declaration omits `callconv(.kernel)`. It has to: that calling
//! convention resolves per target, and on a host architecture
//! `std.builtin.CallingConvention.kernel` is `unreachable`, so the type cannot
//! even be spelled there. Only the parameter list matters for launching, so the
//! comparison deliberately ignores calling convention.

const std = @import("std");

/// Parameter types of a function type, in order.
///
/// Returns a comptime-only `[]const type`, so call sites have to be in comptime
/// context — container scope, a `comptime` block, or another comptime function.
pub fn paramTypes(comptime Fn: type) []const type {
    comptime {
        const info = switch (@typeInfo(Fn)) {
            .@"fn" => |f| f,
            else => @compileError("expected a function type, got " ++ @typeName(Fn)),
        };
        if (info.is_var_args) @compileError("kernels cannot be variadic");
        var types: [info.params.len]type = undefined;
        for (info.params, 0..) |p, i| {
            types[i] = p.type orelse
                @compileError("kernel parameter " ++ std.fmt.comptimePrint("{d}", .{i}) ++
                    " has no known type (generic parameters are not launchable)");
        }
        const frozen = types;
        return &frozen;
    }
}

/// Assert two function types have the same parameter list and return type,
/// ignoring calling convention.
///
/// `Declared` is the shared declaration, `Actual` is `@TypeOf(the_kernel)`.
pub fn assertMatches(comptime Declared: type, comptime Actual: type) void {
    comptime {
        const want = paramTypes(Declared);
        const got = paramTypes(Actual);
        if (want.len != got.len) {
            @compileError(std.fmt.comptimePrint(
                "kernel signature mismatch: declaration takes {d} parameter(s), definition takes {d}",
                .{ want.len, got.len },
            ));
        }
        for (want, got, 0..) |w, g, i| {
            if (w != g) {
                @compileError(std.fmt.comptimePrint(
                    "kernel signature mismatch at parameter {d}: declared {s}, defined {s}",
                    .{ i, @typeName(w), @typeName(g) },
                ));
            }
        }
        const wr = @typeInfo(Declared).@"fn".return_type.?;
        const gr = @typeInfo(Actual).@"fn".return_type.?;
        if (wr != gr) {
            @compileError("kernel signature mismatch in return type: declared " ++
                @typeName(wr) ++ ", defined " ++ @typeName(gr));
        }
    }
}

test "paramTypes extracts in order" {
    comptime {
        const got = paramTypes(fn ([*]const f32, u32, f64) void);
        std.debug.assert(got.len == 3);
        std.debug.assert(got[0] == [*]const f32);
        std.debug.assert(got[1] == u32);
        std.debug.assert(got[2] == f64);
    }
}

test "assertMatches accepts an identical parameter list" {
    const Declared = fn ([*]const f32, [*]f32, f32, u32) void;
    assertMatches(Declared, Declared);
}
