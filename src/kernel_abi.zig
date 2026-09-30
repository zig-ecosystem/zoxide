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

/// Launch bounds declared on a kernel signature — the comptime form of
/// CUDA's `__launch_bounds__` / cuda-oxide's `#[launch_bounds]`.
///
/// Three of the fields lower to PTX performance directives in the `.entry`
/// body (the device side emits them via `cuda.launchBounds`):
///
///   max_threads        -> `.maxntid T`
///   min_blocks_per_sm  -> `.minnctapersm N`
///   max_registers      -> `.maxnreg R` (a per-kernel, comptime-pinned form
///                         of `zoxide bench --maxrregcount`, which stays as
///                         the ptxas-time experiment knob)
///
/// `grid_multiple_of` has no PTX form; the host launch path validates it.
///
/// Declared once in the shared abi module: the device emits exactly these
/// numbers and the host validates exactly these numbers, so the two cannot
/// drift — same motivation as the signature itself.
pub const LaunchBounds = struct {
    max_threads: ?u32 = null,
    min_blocks_per_sm: ?u32 = null,
    max_registers: ?u32 = null,
    grid_multiple_of: ?u32 = null,
};

/// A declaration that carries bounds: `WithBounds(fn (...) void, .{ ... })`
/// is used anywhere a bare `fn (...) void` declaration was accepted.
/// Equivalently, a shared abi module can declare the same shape by hand —
///
/// ```zig
/// pub const my_kernel = struct {
///     pub const signature = fn ([*]f32, u32) void;
///     pub const launch_bounds = .{ .max_threads = 128 };
/// };
/// ```
///
/// Bounds are read structurally (field by field), so the abi module does not
/// need to import this file — which keeps it importable from both the host
/// and device modules without a module-graph clash.
pub fn WithBounds(comptime sig: type, comptime bounds: LaunchBounds) type {
    return struct {
        pub const signature = sig;
        pub const launch_bounds = bounds;
    };
}

/// The function type behind a declaration, bare or WithBounds.
pub fn signatureOf(comptime Decl: type) type {
    comptime {
        switch (@typeInfo(Decl)) {
            .@"fn" => return Decl,
            .@"struct" => {
                if (!@hasDecl(Decl, "signature"))
                    @compileError("expected a kernel signature (fn type) or WithBounds(fn, ...), got " ++ @typeName(Decl));
                return Decl.signature;
            },
            else => @compileError("expected a kernel signature (fn type) or WithBounds(fn, ...), got " ++ @typeName(Decl)),
        }
    }
}

/// The bounds a declaration carries, if any. Read structurally: any struct
/// whose `launch_bounds` decl has some subset of LaunchBounds' fields
/// converts; unknown field names are a compile error rather than silently
/// ignored (a typo'd bound would otherwise never fire).
pub fn boundsOf(comptime Decl: type) ?LaunchBounds {
    comptime {
        if (@typeInfo(Decl) != .@"struct" or !@hasDecl(Decl, "launch_bounds")) return null;
        const lb = Decl.launch_bounds;
        const lb_info = @typeInfo(@TypeOf(lb)).@"struct";
        var out = LaunchBounds{};
        for (@typeInfo(LaunchBounds).@"struct".fields) |f| {
            const has = @hasField(@TypeOf(lb), f.name);
            if (has) @field(out, f.name) = @field(lb, f.name);
        }
        for (lb_info.fields) |f| {
            if (!@hasField(LaunchBounds, f.name))
                @compileError("unknown launch bound '" ++ f.name ++ "'; valid: max_threads, min_blocks_per_sm, max_registers, grid_multiple_of");
        }
        return out;
    }
}

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
/// `Declared` is the shared declaration (bare fn type or WithBounds),
/// `Actual` is `@TypeOf(the_kernel)`.
pub fn assertMatches(comptime Declared: type, comptime Actual: type) void {
    comptime {
        const Decl = signatureOf(Declared);
        const want = paramTypes(Decl);
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
        const wr = @typeInfo(Decl).@"fn".return_type.?;
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

test "WithBounds carries the signature through assertMatches and boundsOf" {
    const Bare = fn ([*]const f32, u32) void;
    const Decl = WithBounds(Bare, .{ .max_threads = 128, .grid_multiple_of = 4 });
    comptime {
        assertMatches(Decl, Bare);
        std.debug.assert(signatureOf(Decl) == Bare);
        const b = boundsOf(Decl).?;
        std.debug.assert(b.max_threads.? == 128);
        std.debug.assert(b.grid_multiple_of.? == 4);
        std.debug.assert(boundsOf(Bare) == null);
    }
}
