//! PTX post-processing.
//!
//! Zig cannot emit a device global that the CUDA driver can resolve by name.
//! A module-scope `var` reaches the PTX as
//!
//!     .global .align 4 .b8 mod_$_dev_scale[16] = {...};
//!
//! with no `.visible`, so ptxas keeps it module-local and `cuModuleGetGlobal`
//! cannot find it. Every route to external linkage fails in the NVPTX backend
//! (measured on zig 0.16.0 / LLVM 21.1.8) because Zig lowers `export` on a
//! variable to an LLVM alias, and NVPTX only accepts aliases whose aliasee is a
//! non-kernel function:
//!
//!     export var x addrspace(.global)  -> Alias and aliasee types don't match
//!     @export(&x, .{ .linkage = .strong }) -> same
//!     export var x                     -> LLVM ERROR: NVPTX aliasee must be a
//!                                         non-kernel function definition (abort)
//!
//! So visibility is added here instead. This only inserts one token on the
//! declaration line; accesses (`mov.b64 %rd, sym` plus `ld.global`/`st.global`)
//! do not depend on visibility and are left untouched.
//!
//! Caveat worth stating plainly: this relies on the shape of LLVM's NVPTX
//! output, not on a documented guarantee. It is narrow and it fails loudly —
//! either ptxas rejects the file, or the symbol is still missing and
//! `Module.global` reports `GlobalNotFound` — but it is not a language-level
//! promise.
const std = @import("std");

pub const Promoted = struct {
    /// Rewritten PTX. Caller owns it.
    text: []u8,
    /// Symbols this run added `.visible` to, in the order found. Caller owns the
    /// slice; the names point into `text`.
    promoted: [][]const u8,
    /// Requested names whose declaration already carried `.visible`. Nothing was
    /// done for them and nothing is wrong — kept apart from `missing` so
    /// re-running the pass is not mistaken for a failure.
    already: [][]const u8,
    /// Requested names with no matching module-scope global at all. Non-empty
    /// means the caller asked for something that is not there, which would
    /// otherwise only surface much later as a `GlobalNotFound` at module load on
    /// a GPU host. Caller owns the slice; names point into the caller's `names`.
    missing: [][]const u8,

    pub fn deinit(self: Promoted, gpa: std.mem.Allocator) void {
        gpa.free(self.text);
        gpa.free(self.promoted);
        gpa.free(self.already);
        gpa.free(self.missing);
    }
};

const Decl = struct {
    name: []const u8,
    /// Already carries `.visible`, so there is nothing to do for it. Tracked
    /// separately from "absent" so a second run does not look like a failure.
    visible: bool,
};

/// Extract the symbol name from a module-scope `.global` declaration line.
/// Returns null if the line does not look like one.
///
/// `.global .align 4 .b8 mod_$_dev_scale[16] = {0, 0};` -> `mod_$_dev_scale`
fn declName(line: []const u8) ?Decl {
    var it = std.mem.tokenizeAny(u8, line, " \t");
    var first = it.next() orelse return null;
    var visible = false;
    if (std.mem.eql(u8, first, ".visible")) {
        visible = true;
        first = it.next() orelse return null;
    }
    if (!std.mem.eql(u8, first, ".global")) return null;
    while (it.next()) |tok| {
        // Skip directives (.align, .b8, ...) and their numeric operands.
        if (tok.len == 0) continue;
        if (tok[0] == '.') continue;
        if (std.ascii.isDigit(tok[0])) continue;
        // First identifier-like token is the symbol; trim any array suffix or
        // trailing punctuation.
        const end = std.mem.indexOfAny(u8, tok, "[=;") orelse tok.len;
        const name = tok[0..end];
        return if (name.len == 0) null else .{ .name = name, .visible = visible };
    }
    return null;
}

/// True if `symbol` is the PTX name for the Zig declaration `want`. Zig mangles
/// module-scope decls as `<root source file stem>_$_<decl>`, so accept either
/// the full mangled name or the bare decl.
fn symbolMatches(symbol: []const u8, want: []const u8) bool {
    if (std.mem.eql(u8, symbol, want)) return true;
    if (std.mem.endsWith(u8, symbol, want)) {
        const prefix = symbol[0 .. symbol.len - want.len];
        return std.mem.endsWith(u8, prefix, "_$_");
    }
    return false;
}

/// Add `.visible` to the module-scope `.global` declarations named in `names`.
///
/// Already-visible declarations are counted as promoted and left alone, so the
/// pass is idempotent.
pub fn promoteGlobals(
    gpa: std.mem.Allocator,
    ptx: []const u8,
    names: []const []const u8,
) !Promoted {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    // Offsets, not slices: `out` reallocates as it grows and `toOwnedSlice` may
    // move the buffer, so any slice into it taken during the loop would dangle.
    var spans: std.ArrayList(struct { off: usize, len: usize }) = .empty;
    defer spans.deinit(gpa);

    const Outcome = enum { absent, promoted, already };
    const seen = try gpa.alloc(Outcome, names.len);
    defer gpa.free(seen);
    @memset(seen, .absent);

    var lines = std.mem.splitScalar(u8, ptx, '\n');
    var first = true;
    while (lines.next()) |line| {
        if (!first) try out.append(gpa, '\n');
        first = false;

        var wrote = false;
        if (declName(line)) |decl| {
            for (names, 0..) |want, i| {
                if (!symbolMatches(decl.name, want)) continue;
                if (decl.visible) {
                    seen[i] = .already;
                    break; // leave the line as it is
                }
                seen[i] = .promoted;
                const start = out.items.len;
                try out.appendSlice(gpa, ".visible ");
                try out.appendSlice(gpa, line);
                // Record where the name landed in the rewritten text.
                const rel = std.mem.indexOf(u8, out.items[start..], decl.name).?;
                try spans.append(gpa, .{ .off = start + rel, .len = decl.name.len });
                wrote = true;
                break;
            }
        }
        if (!wrote) try out.appendSlice(gpa, line);
    }

    var already: std.ArrayList([]const u8) = .empty;
    errdefer already.deinit(gpa);
    var missing: std.ArrayList([]const u8) = .empty;
    errdefer missing.deinit(gpa);
    for (names, seen) |want, o| switch (o) {
        .absent => try missing.append(gpa, want),
        .already => try already.append(gpa, want),
        .promoted => {},
    };

    // Resolve the offsets only once the text has its final address.
    const text = try out.toOwnedSlice(gpa);
    errdefer gpa.free(text);
    const promoted = try gpa.alloc([]const u8, spans.items.len);
    errdefer gpa.free(promoted);
    for (spans.items, promoted) |s, *p| p.* = text[s.off..][0..s.len];

    return .{
        .text = text,
        .promoted = promoted,
        .already = try already.toOwnedSlice(gpa),
        .missing = try missing.toOwnedSlice(gpa),
    };
}

/// Rewrite `in_path` to `out_path` so the named device globals carry `.visible`.
/// Returns a non-null exit code on failure, having already reported why.
///
/// Lives here rather than in the CLI because two entry points need it: the
/// `zoxide ptx-export` subcommand, and the host-native helper the build graph
/// runs (`tools/ptx-promote.zig`). The build cannot use the zoxide binary
/// itself, since that one follows `-Dtarget` and is not runnable on the build
/// host when cross-compiling.
pub fn applyToFile(
    gpa: std.mem.Allocator,
    io: std.Io,
    in_path: []const u8,
    out_path: []const u8,
    csv: []const u8,
) !?u8 {
    var names: std.ArrayList([]const u8) = .empty;
    defer names.deinit(gpa);
    var it = std.mem.tokenizeScalar(u8, csv, ',');
    while (it.next()) |n| {
        const trimmed = std.mem.trim(u8, n, " \t");
        if (trimmed.len != 0) try names.append(gpa, trimmed);
    }
    if (names.items.len == 0) {
        std.debug.print("error: no device global names given\n", .{});
        return 1;
    }

    const src = std.Io.Dir.cwd().readFileAlloc(io, in_path, gpa, .unlimited) catch |err| {
        std.debug.print("error: cannot read PTX '{s}' ({s})\n", .{ in_path, @errorName(err) });
        return 1;
    };
    defer gpa.free(src);

    const r = try promoteGlobals(gpa, src, names.items);
    defer r.deinit(gpa);

    // A name that matched nothing is almost always a typo or a global the
    // optimiser removed. Failing here is the point: the alternative is a PTX
    // that loads fine and only fails at cuModuleGetGlobal, on a GPU host, much
    // later.
    if (r.missing.len != 0) {
        std.debug.print("error: {d} requested global(s) not present in '{s}':\n", .{ r.missing.len, in_path });
        for (r.missing) |m| std.debug.print("  {s}\n", .{m});
        std.debug.print(
            "  besides a typo, the usual cause is that the global was folded away: LLVM\n" ++
                "  assumes nothing outside the module writes it, so reads can be constant-\n" ++
                "  folded against the initialiser and the symbol dropped. Read it with\n" ++
                "  cuda.ldg() instead, which is opaque to the optimiser.\n" ++
                "  Declared globals in this file:\n",
            .{},
        );
        var lines = std.mem.splitScalar(u8, src, '\n');
        var any = false;
        while (lines.next()) |line| {
            if (declName(line) == null) continue;
            std.debug.print("    {s}\n", .{line[0..@min(line.len, 96)]});
            any = true;
        }
        if (!any) std.debug.print("    (none)\n", .{});
        return 1;
    }

    std.Io.Dir.cwd().writeFile(io, .{ .sub_path = out_path, .data = r.text }) catch |err| {
        std.debug.print("error: cannot write PTX '{s}' ({s})\n", .{ out_path, @errorName(err) });
        return 1;
    };

    for (r.promoted) |p| std.debug.print("exported device global: {s}\n", .{p});
    for (r.already) |a| std.debug.print("device global already visible: {s}\n", .{a});
    return null;
}

/// `<prog> <in.ptx> --globals a,b [-o out.ptx]`, shared by the CLI subcommand
/// and the build-time helper.
pub fn cliMain(gpa: std.mem.Allocator, io: std.Io, args: []const [:0]const u8) !u8 {
    var input: ?[]const u8 = null;
    var output: ?[]const u8 = null;
    var globals: ?[]const u8 = null;
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const a = args[i];
        const is_globals = std.mem.eql(u8, a, "--globals");
        const is_out = std.mem.eql(u8, a, "-o");
        if (is_globals or is_out) {
            i += 1;
            if (i >= args.len) {
                std.debug.print("error: {s} needs a value\n", .{a});
                return 1;
            }
            if (is_out) output = args[i] else globals = args[i];
        } else if (input == null) {
            input = a;
        } else {
            std.debug.print("error: unexpected argument '{s}'\n", .{a});
            return 1;
        }
    }
    if (input == null or globals == null) {
        std.debug.print("error: expected '<in.ptx> --globals a,b [-o out.ptx]'\n", .{});
        return 1;
    }
    // Default to rewriting in place, which is what a build post-step wants.
    const dst = output orelse input.?;
    if (try applyToFile(gpa, io, input.?, dst, globals.?)) |code| return code;
    std.debug.print("wrote PTX: {s}\n", .{dst});
    return 0;
}

test declName {
    const t = std.testing;
    try t.expectEqualStrings("g_$_dev_scale", declName(".global .align 4 .b8 g_$_dev_scale[16] = {0, 1};").?.name);
    try t.expectEqualStrings("x", declName(".global .align 4 .b8 x;").?.name);
    try t.expectEqualStrings("__anon_815", declName(".global .align 4 .b8 __anon_815[4] = {0};").?.name);
    try t.expect(!declName(".global .align 4 .b8 x;").?.visible);
    // Not module-scope global declarations.
    try t.expect(declName(".visible .entry g_$_k(") == null);
    try t.expect(declName("\tld.global.nc.b32 %r1, [%rd2];") == null);
    try t.expect(declName(".target sm_90") == null);
    try t.expect(declName("") == null);
    // Already visible: recognised, and flagged so it is not re-promoted.
    const v = declName(".visible .global .align 4 .b8 y[4];").?;
    try t.expectEqualStrings("y", v.name);
    try t.expect(v.visible);
}

test symbolMatches {
    const t = std.testing;
    try t.expect(symbolMatches("g_$_dev_scale", "dev_scale"));
    try t.expect(symbolMatches("g_$_dev_scale", "g_$_dev_scale"));
    try t.expect(!symbolMatches("g_$_dev_scale2", "dev_scale"));
    // Suffix match must be on a mangling boundary, not any substring.
    try t.expect(!symbolMatches("g_$_my_dev_scale", "dev_scale"));
}

test "promoteGlobals adds visible to the named decl only" {
    const gpa = std.testing.allocator;
    const src =
        ".version 7.8\n" ++
        ".global .align 4 .b8 g_$_dev_scale[16] = {0, 0, 128, 63};\n" ++
        ".global .align 1 .b8 __anon_815[4] = {1, 2, 3, 4};\n" ++
        ".visible .entry g_$_k(\n" ++
        "\tmov.b64 %rd6, g_$_dev_scale;\n";

    const r = try promoteGlobals(gpa, src, &.{"dev_scale"});
    defer r.deinit(gpa);

    try std.testing.expectEqual(@as(usize, 1), r.promoted.len);
    try std.testing.expectEqualStrings("g_$_dev_scale", r.promoted[0]);
    try std.testing.expectEqual(@as(usize, 0), r.missing.len);
    try std.testing.expectEqual(@as(usize, 0), r.already.len);
    // The requested decl gained .visible.
    try std.testing.expect(std.mem.indexOf(u8, r.text, ".visible .global .align 4 .b8 g_$_dev_scale[16]") != null);
    // The compiler-internal one did not.
    try std.testing.expect(std.mem.indexOf(u8, r.text, ".visible .global .align 1 .b8 __anon_815") == null);
    // Accesses are untouched.
    try std.testing.expect(std.mem.indexOf(u8, r.text, "\tmov.b64 %rd6, g_$_dev_scale;") != null);
    // Line count preserved.
    try std.testing.expectEqual(
        std.mem.count(u8, src, "\n"),
        std.mem.count(u8, r.text, "\n"),
    );
}

test "promoteGlobals reports names it could not find" {
    const gpa = std.testing.allocator;
    const src = ".global .align 4 .b8 g_$_dev_scale[16] = {0};\n";
    const r = try promoteGlobals(gpa, src, &.{ "dev_scale", "dev_bias" });
    defer r.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 1), r.promoted.len);
    try std.testing.expectEqual(@as(usize, 1), r.missing.len);
    try std.testing.expectEqualStrings("dev_bias", r.missing[0]);
}

test "promoteGlobals is idempotent" {
    const gpa = std.testing.allocator;
    const src = ".global .align 4 .b8 g_$_dev_scale[16] = {0};\n";
    const a = try promoteGlobals(gpa, src, &.{"dev_scale"});
    defer a.deinit(gpa);
    const b = try promoteGlobals(gpa, a.text, &.{"dev_scale"});
    defer b.deinit(gpa);
    try std.testing.expectEqualStrings(a.text, b.text);
    // Second pass has nothing to do, and reports that as "already visible"
    // rather than as a missing symbol — otherwise a re-run would look like a
    // failure to the caller.
    try std.testing.expectEqual(@as(usize, 0), b.promoted.len);
    try std.testing.expectEqual(@as(usize, 0), b.missing.len);
    try std.testing.expectEqual(@as(usize, 1), b.already.len);
    try std.testing.expectEqualStrings("dev_scale", b.already[0]);
}
