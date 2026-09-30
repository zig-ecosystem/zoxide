//! Deferred device operations: a thin lazy layer over the eager stream API
//! in host.zig (`uploadAsync`/`downloadAsync`/`fillBytesAsync`/`launchOn`).
//!
//! What it is: operations as *values*. You describe the work first —
//! uploads, a fill, kernel launches, each bound to its stream — and execute
//! later with one `.sync()`. That buys two things over the eager calls:
//!
//!   1. the description can be built somewhere other than where it runs (a
//!      helper returns a pipeline; the caller decides when it fires);
//!   2. issue-all-then-sync-all is expressed once instead of interleaving
//!      the eager calls with `stream.sync()` by hand.
//!
//! What it is not: no futures, no dependency DAG, no scheduler. Ordering
//! within a stream is the hardware's; the layer adds nothing but values.
//! Cross-stream dependencies via `cuStreamWaitEvent` are *not* offered: the
//! binding does not exist in cuda_driver.zig yet, so two streams' operations
//! overlap with no way to express "launch waits for the other stream's
//! upload" beyond syncing both — add the binding first (GPU-verified) rather
//! than faking it with extra syncs.
//!
//! The eager API is unchanged and remains the right tool for single-stream
//! code; this layer pays for itself when the description and the execution
//! live in different places, or when several streams are involved.

const std = @import("std");
const host = @import("host.zig");

/// A deferred operation: a payload plus the function that executes it,
/// bound to a stream. Construct via `Builder`; run via `Builder.sync` or
/// directly (`op.run(ctx)`) — an Operation is just a value.
pub const Operation = struct {
    stream: host.Stream,
    payload: []align(16) const u8,
    runFn: *const fn (ctx: *host.Context, op: Operation) host.Error!void,

    pub fn run(self: Operation, ctx: *host.Context) host.Error!void {
        return self.runFn(ctx, self);
    }
};

fn shim(comptime Payload: type, comptime f: fn (ctx: *host.Context, p: *const Payload, stream: host.Stream) host.Error!void) *const fn (*host.Context, Operation) host.Error!void {
    return struct {
        fn call(ctx: *host.Context, op: Operation) host.Error!void {
            const p: *const Payload = @ptrCast(@alignCast(op.payload.ptr));
            return f(ctx, p, op.stream);
        }
    }.call;
}

/// Collects operations into an arena. The arena owns all payloads; free it
/// only after the last `.sync()` of its operations.
pub const Builder = struct {
    arena: std.mem.Allocator,
    ops: std.array_list.Managed(Operation),

    pub fn init(arena: std.mem.Allocator) Builder {
        return .{ .arena = arena, .ops = std.array_list.Managed(Operation).init(arena) };
    }

    fn add(self: *Builder, stream: host.Stream, comptime Payload: type, value: Payload, comptime f: anytype) !*Operation {
        const bytes = try self.arena.alignedAlloc(u8, .@"16", @sizeOf(Payload));
        const p: *Payload = @ptrCast(@alignCast(bytes.ptr));
        p.* = value;
        try self.ops.append(.{ .stream = stream, .payload = bytes, .runFn = shim(Payload, f) });
        return &self.ops.items[self.ops.items.len - 1];
    }

    /// Deferred `ctx.uploadAsync(dst, src, stream)`. Same element-type rule
    /// as the eager call: `dst` is a `Slice(T)`, `src` a `[]const T`.
    pub fn upload(self: *Builder, stream: host.Stream, dst: anytype, src: anytype) !*Operation {
        const P = struct { dst: @TypeOf(dst), src: @TypeOf(src) };
        return self.add(stream, P, .{ .dst = dst, .src = src }, struct {
            fn f(ctx: *host.Context, p: *const P, s: host.Stream) host.Error!void {
                try ctx.uploadAsync(p.dst, p.src, s);
            }
        }.f);
    }

    /// Deferred `ctx.downloadAsync(dst, src, stream)`.
    pub fn download(self: *Builder, stream: host.Stream, dst: anytype, src: anytype) !*Operation {
        const P = struct { dst: @TypeOf(dst), src: @TypeOf(src) };
        return self.add(stream, P, .{ .dst = dst, .src = src }, struct {
            fn f(ctx: *host.Context, p: *const P, s: host.Stream) host.Error!void {
                try ctx.downloadAsync(p.dst, p.src, s);
            }
        }.f);
    }

    /// Deferred `ctx.fillBytesAsync(dst, value, stream)`.
    pub fn fillBytes(self: *Builder, stream: host.Stream, dst: anytype, value: u8) !*Operation {
        const P = struct { dst: @TypeOf(dst), value: u8 };
        return self.add(stream, P, .{ .dst = dst, .value = value }, struct {
            fn f(ctx: *host.Context, p: *const P, s: host.Stream) host.Error!void {
                try ctx.fillBytesAsync(p.dst, p.value, s);
            }
        }.f);
    }

    /// Deferred `kern.launchOn(stream, grid, block, 0, args)`. The argument
    /// tuple is copied into the arena, so the caller's stack frame may be
    /// gone by the time the launch runs — that is the point of deferral.
    pub fn launch(
        self: *Builder,
        comptime Decl: type,
        kern: host.Kernel(Decl),
        stream: host.Stream,
        grid: host.Dims,
        block: host.Dims,
        args: anytype,
    ) !*Operation {
        const P = struct {
            kern: host.Kernel(Decl),
            grid: host.Dims,
            block: host.Dims,
            args: @TypeOf(args),
        };
        return self.add(stream, P, .{ .kern = kern, .grid = grid, .block = block, .args = args }, struct {
            fn f(ctx: *host.Context, p: *const P, s: host.Stream) host.Error!void {
                _ = ctx;
                try p.kern.launchOn(s, p.grid, p.block, 0, p.args);
            }
        }.f);
    }

    /// A caller-supplied operation — the extension point, and the test seam
    /// (a runFn that never touches ctx runs fine without a GPU).
    pub fn custom(self: *Builder, stream: host.Stream, runFn: *const fn (ctx: *host.Context, op: Operation) host.Error!void) !*Operation {
        const bytes = try self.arena.alignedAlloc(u8, .@"16", 16);
        @memset(bytes, 0);
        try self.ops.append(.{ .stream = stream, .payload = bytes, .runFn = runFn });
        return &self.ops.items[self.ops.items.len - 1];
    }

    /// Issue every operation in construction order, then synchronize each
    /// stream that was used, once. Two operations on the same stream are
    /// ordered by the hardware; two on different streams overlap — there is
    /// deliberately no cross-stream wait (see the module doc comment).
    ///
    /// On error the failing operation stops the issue, and streams already
    /// issued to are still synchronized before the error propagates, so a
    /// failed op never leaves in-flight work unaccounted for.
    pub fn sync(self: *Builder, ctx: *host.Context) host.Error!void {
        var first_err: ?host.Error = null;
        for (self.ops.items) |op| {
            op.run(ctx) catch |e| {
                first_err = e;
                break;
            };
        }
        try self.syncStreams();
        if (first_err) |e| return e;
    }

    /// Synchronize each distinct stream used by the recorded operations.
    /// Public because partial runs (issue some, sync, issue more) are a
    /// legitimate use.
    pub fn syncStreams(self: *Builder) host.Error!void {
        var seen = std.array_list.Managed(host.Stream).init(self.arena);
        for (self.ops.items) |op| {
            var dup = false;
            for (seen.items) |s| {
                if (s.s == op.stream.s) {
                    dup = true;
                    break;
                }
            }
            if (!dup) {
                try op.stream.sync();
                try seen.append(op.stream);
            }
        }
    }
};

// ---------------------------------------------------------------------------
// Tests — the graph bookkeeping is host-testable; the exec path is three
// driver calls away from anything interesting and needs a GPU. The custom()
// seam exists so run-order and error semantics are testable without one.
// ---------------------------------------------------------------------------

const testing = std.testing;

fn fakeStream(h: usize) host.Stream {
    return .{ .drv = undefined, .s = @ptrFromInt(h) };
}

test "Builder records operations in construction order, with their streams" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var b = Builder.init(arena.allocator());
    const noop = struct {
        fn f(ctx: *host.Context, op: Operation) host.Error!void {
            _ = ctx;
            _ = op;
        }
    }.f;

    const s1 = fakeStream(0x1000);
    const s2 = fakeStream(0x2000);
    _ = try b.custom(s1, noop);
    _ = try b.custom(s2, noop);
    _ = try b.custom(s1, noop);

    try testing.expectEqual(@as(usize, 3), b.ops.items.len);
    try testing.expectEqual(s1.s, b.ops.items[0].stream.s);
    try testing.expectEqual(s2.s, b.ops.items[1].stream.s);
    try testing.expectEqual(s1.s, b.ops.items[2].stream.s);
}

test "ops run in order; an error stops the issue" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();

    const G = struct {
        var log: [4]u8 = undefined;
        var len: usize = 0;
        fn reset() void {
            len = 0;
        }
        fn fA(ctx: *host.Context, op: Operation) host.Error!void {
            _ = ctx;
            _ = op;
            log[len] = 'a';
            len += 1;
        }
        fn fB(ctx: *host.Context, op: Operation) host.Error!void {
            _ = ctx;
            _ = op;
            log[len] = 'b';
            len += 1;
        }
        fn fBad(ctx: *host.Context, op: Operation) host.Error!void {
            _ = ctx;
            _ = op;
            log[len] = 'x';
            len += 1;
            return error.CudaCall;
        }
    };

    const s = fakeStream(0x3000);
    G.reset();
    var b = Builder.init(arena.allocator());
    _ = try b.custom(s, G.fA);
    _ = try b.custom(s, G.fB);
    for (b.ops.items) |op| try op.run(undefined);
    try testing.expectEqualStrings("ab", G.log[0..G.len]);

    // Error path: the failing op runs, the one after must not.
    G.reset();
    var b2 = Builder.init(arena.allocator());
    _ = try b2.custom(s, G.fBad);
    _ = try b2.custom(s, G.fA);
    for (b2.ops.items) |op| {
        op.run(undefined) catch break;
    }
    try testing.expectEqualStrings("x", G.log[0..G.len]);
}

test "launch payload survives the arena (args outlive the caller's frame)" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var b = Builder.init(arena.allocator());
    const Decl = fn ([*]f32, u32) void;
    const K = host.Kernel(Decl);
    const k: K = .{
        .inner = undefined,
        .limits = .{
            .max_threads_per_block = 1024,
            .max_block = .{ 1024, 1024, 64 },
            .max_grid = .{ 2147483647, 65535, 65535 },
            .max_shared_per_block = 49152,
        },
        .max_threads = 1024,
    };
    // Note the args are still checked against the signature at comptime —
    // deferral does not weaken the typed-launch contract.
    const buf: host.Slice(f32) = .{ .ptr = 0xdead, .len = 4 };
    const args = .{ buf, @as(u32, 99) };
    const op = try b.launch(Decl, k, fakeStream(0x4000), .{ .x = 4 }, .{ .x = 256 }, args);
    // The args tuple was copied into the arena: read it back through the payload.
    const P = struct { kern: K, grid: host.Dims, block: host.Dims, args: @TypeOf(args) };
    const p: *const P = @ptrCast(@alignCast(op.payload.ptr));
    try testing.expectEqual(@as(usize, 4), p.args.@"0".len);
    try testing.expectEqual(@as(u32, 99), p.args.@"1");
    try testing.expectEqual(@as(u32, 4), p.grid.x);
}

test "upload payload keeps the element-typed slice, not bytes" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var b = Builder.init(arena.allocator());
    const dst: host.Slice(f32) = .{ .ptr = 0x1000, .len = 4 };
    const src = [_]f32{ 1, 2, 3, 4 };
    const op = try b.upload(fakeStream(0x5000), dst, &src);
    const P = struct { dst: host.Slice(f32), src: *const [4]f32 };
    const p: *const P = @ptrCast(@alignCast(op.payload.ptr));
    try testing.expectEqual(@as(usize, 4), p.dst.len);
    try testing.expectEqual(@as(f32, 3), p.src[2]);
}
