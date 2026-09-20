//! `zoxide gen` — generate Zig NVVM intrinsic bindings from cuda-oxide's
//! intrinsics data (catalog.json + probes/*.ll).
//!
//! Probes are the authoritative source for LLVM signatures (concrete declare
//! lines); the catalog contributes family/module/name metadata for grouping.
//! Output: a single Zig file of extern decls + thin wrappers, grouped by
//! cuda-oxide's rust.module.

const std = @import("std");

const Entry = struct {
    id: []const u8,
    family: []const u8 = "misc",
    module: []const u8 = "misc",
    name: []const u8, // wrapper name
    symbol: []const u8, // llvm.nvvm.*
    ret: []const u8, // zig type
    args: [][]const u8, // zig types
};

const Agg = struct { llvm: []const u8, zig: []const u8, name: []const u8 };

fn jsonStr(v: ?std.json.Value) ?[]const u8 {
    const x = v orelse return null;
    return switch (x) {
        .string => |s| s,
        else => null,
    };
}

/// Map one LLVM IR type spelling to a Zig type. Returns null if unmappable.
fn mapType(alloc: std.mem.Allocator, llvm_ty: []const u8, aggs: *std.ArrayList(Agg)) !?[]const u8 {
    const t = std.mem.trim(u8, llvm_ty, " ");
    if (std.mem.eql(u8, t, "void")) return "void";
    if (std.mem.eql(u8, t, "i1")) return "bool";
    if (std.mem.eql(u8, t, "i8")) return "i8";
    if (std.mem.eql(u8, t, "i16")) return "i16";
    if (std.mem.eql(u8, t, "i32")) return "i32";
    if (std.mem.eql(u8, t, "i64")) return "i64";
    if (std.mem.eql(u8, t, "i128")) return "i128";
    if (std.mem.eql(u8, t, "half")) return "f16";
    if (std.mem.eql(u8, t, "float")) return "f32";
    if (std.mem.eql(u8, t, "double")) return "f64";
    if (std.mem.eql(u8, t, "ptr")) return "?*anyopaque";
    if (std.mem.eql(u8, t, "ptr addrspace(1)")) return "[*]addrspace(.global) const u8";
    if (std.mem.eql(u8, t, "ptr addrspace(3)")) return "[*]addrspace(.shared) u8";
    if (std.mem.eql(u8, t, "ptr addrspace(5)")) return "[*]addrspace(.local) u8";
    if (t.len > 0 and t[0] == '{') {
        // aggregate return like "{ i32, i1 }": cache an extern struct
        for (aggs.items) |a| {
            if (std.mem.eql(u8, a.llvm, t)) return a.name;
        }
        const name = try std.fmt.allocPrint(alloc, "Agg{d}", .{aggs.items.len});
        var zig = std.ArrayList(u8).empty;
        var depth: usize = 0;
        var start: usize = 1;
        var idx: usize = 0;
        var i: usize = 1;
        try zig.appendSlice(alloc, "extern struct {");
        while (i < t.len) : (i += 1) {
            const c = t[i];
            if (c == '{' or c == '<' or c == '(') depth += 1;
            if (c == '}' or c == '>' or c == ')') {
                if (c == '}' and depth == 0) {
                    if (std.mem.trim(u8, t[start..i], " ").len > 0) {
                        const m = (try mapType(alloc, t[start..i], aggs)) orelse return null;
                        try zig.appendSlice(alloc, try std.fmt.allocPrint(alloc, " f{d}: {s},", .{ idx, m }));
                        idx += 1;
                    }
                    break;
                }
                depth -= 1;
            }
            if (c == ',' and depth == 0) {
                const m = (try mapType(alloc, t[start..i], aggs)) orelse return null;
                try zig.appendSlice(alloc, try std.fmt.allocPrint(alloc, " f{d}: {s},", .{ idx, m }));
                idx += 1;
                start = i + 1;
            }
        }
        try zig.appendSlice(alloc, " }");
        try aggs.append(alloc, .{ .llvm = try alloc.dupe(u8, t), .zig = zig.items, .name = name });
        return name;
    }
    return null;
}

/// Parse "declare <ret> @<sym>(<args>)" from probe text. Returns null if absent.
fn parseDeclare(text: []const u8) ?struct { ret: []const u8, sym: []const u8, args: []const u8 } {
    var it = std.mem.splitScalar(u8, text, '\n');
    while (it.next()) |line| {
        const l = std.mem.trim(u8, line, " \r\t");
        if (!std.mem.startsWith(u8, l, "declare ")) continue;
        if (std.mem.indexOf(u8, l, "@llvm.") == null) continue;
        const after = l["declare ".len..];
        const at = std.mem.indexOfScalar(u8, after, '@') orelse return null;
        const ret = std.mem.trim(u8, after[0..at], " ");
        const rest = after[at + 1 ..];
        const open = std.mem.indexOfScalar(u8, rest, '(') orelse return null;
        // find matching close paren
        var depth: usize = 1;
        var close = open + 1;
        while (close < rest.len and depth > 0) : (close += 1) {
            if (rest[close] == '(') depth += 1;
            if (rest[close] == ')') depth -= 1;
        }
        return .{
            .ret = ret,
            .sym = rest[0..open],
            .args = rest[open + 1 .. close - 1],
        };
    }
    return null;
}

fn splitTopLevel(alloc: std.mem.Allocator, s: []const u8) ![][]const u8 {
    var out = std.ArrayList([]const u8).empty;
    var depth: usize = 0;
    var start: usize = 0;
    for (s, 0..) |c, i| {
        if (c == '{' or c == '<' or c == '(') depth += 1;
        if (c == '}' or c == '>' or c == ')') depth -= 1;
        if (c == ',' and depth == 0) {
            try out.append(alloc, std.mem.trim(u8, s[start..i], " "));
            start = i + 1;
        }
    }
    const last = std.mem.trim(u8, s[start..], " ");
    if (last.len > 0) try out.append(alloc, last);
    return out.items;
}

// --- asm-probe support -----------------------------------------------------
//
// Probes that lower via inline PTX look like:
//   %r = call i32 asm "abs.bf16x2 $0, $1;", "=r,r"(i32 %arg0)
// Zig supports NAMED operand substitution (%[name]) but not positional $N,
// so we rewrite the template: $N -> %[outN] for N < n_outputs else
// %[in{N-nout}], and emit a matching named-constraint asm block.

const AsmOperand = struct {
    constraint: u8, // r l f d h n
    ty: []const u8, // zig type
    is_comptime: bool = false, // "n" immediates
};

const AsmEntry = struct {
    id: []const u8,
    family: []const u8,
    module: []const u8,
    name: []const u8,
    template: []const u8, // converted to %[name] form, zig-escaped
    outputs: []AsmOperand,
    inputs: []AsmOperand,
    ret: []const u8, // zig type: "void", single type, or Agg* name
    is_volatile: bool,
};

fn constraintType(a: std.mem.Allocator, c: u8) !?AsmOperand {
    _ = a;
    return switch (c) {
        'r' => .{ .constraint = 'r', .ty = "u32" },
        'l' => .{ .constraint = 'l', .ty = "u64" },
        'f' => .{ .constraint = 'f', .ty = "f32" },
        'd' => .{ .constraint = 'd', .ty = "f64" },
        'h' => .{ .constraint = 'h', .ty = "u16" },
        'n' => .{ .constraint = 'n', .ty = "u32", .is_comptime = true },
        else => null,
    };
}

const ParsedAsm = struct { ret: []const u8, template: []const u8, constraints: []const u8 };

/// Extract (retty, template, constraints) from the first `call ... asm` line.
fn parseAsmProbe(text: []const u8) ?ParsedAsm {
    var it = std.mem.splitScalar(u8, text, '\n');
    while (it.next()) |line| {
        const l = std.mem.trim(u8, line, " \r\t");
        const asm_pos = std.mem.indexOf(u8, l, " asm") orelse continue;
        // return type: between "call " and " asm"
        const call_pos = std.mem.indexOf(u8, l, "call ") orelse continue;
        const ret = std.mem.trim(u8, l[call_pos + 5 .. asm_pos], " ");
        // template: first quoted string after `asm`
        const after_asm = l[asm_pos + 4 ..];
        const q1 = std.mem.indexOfScalar(u8, after_asm, '"') orelse continue;
        const q2 = std.mem.indexOfScalarPos(u8, after_asm, q1 + 1, '"') orelse continue;
        const template = after_asm[q1 + 1 .. q2];
        const q3 = std.mem.indexOfScalarPos(u8, after_asm, q2 + 1, '"') orelse continue;
        const q4 = std.mem.indexOfScalarPos(u8, after_asm, q3 + 1, '"') orelse continue;
        const constraints = after_asm[q3 + 1 .. q4];
        return .{ .ret = ret, .template = template, .constraints = constraints };
    }
    return null;
}

/// Rewrite LLVM positional operands ($N) into zig named operands and split
/// the constraint string. Returns null on unsupported template features.
fn convertAsm(
    a: std.mem.Allocator,
    pa: ParsedAsm,
    aggs: *std.ArrayList(Agg),
) !?struct { template: []const u8, outputs: []AsmOperand, inputs: []AsmOperand, ret: []const u8, is_volatile: bool } {
    // constraints: "=f,=f,f,r,n,~{memory}"
    var outputs = std.ArrayList(AsmOperand).empty;
    var inputs = std.ArrayList(AsmOperand).empty;
    var has_mem_clobber = false;
    var cit = std.mem.splitScalar(u8, pa.constraints, ',');
    while (cit.next()) |raw| {
        const c = std.mem.trim(u8, raw, " ");
        if (c.len == 0) continue;
        if (c[0] == '~') {
            has_mem_clobber = true;
            continue;
        }
        if (c.len > 2) return null; // exotic multi-char constraint
        var letter = c[0];
        var is_out = false;
        if (letter == '=') {
            is_out = true;
            letter = c[1];
        }
        const op = (try constraintType(a, letter)) orelse return null;
        if (is_out) try outputs.append(a, op) else try inputs.append(a, op);
    }

    // zig caps inline asm at 16 outputs and 32 inputs (AstGen).
    if (outputs.items.len > 15 or inputs.items.len > 31) return null;

    // template conversion
    var tmpl = std.ArrayList(u8).empty;    var i: usize = 0;
    while (i < pa.template.len) : (i += 1) {
        const ch = pa.template[i];
        if (ch == '$') {
            if (i + 1 < pa.template.len and pa.template[i + 1] == '$') {
                try tmpl.append(a, '$');
                i += 1;
                continue;
            }
            if (i + 1 < pa.template.len and pa.template[i + 1] == '{') return null; // ${N:mod} unsupported
            var j = i + 1;
            while (j < pa.template.len and std.ascii.isDigit(pa.template[j])) j += 1;
            if (j == i + 1) return null;
            const n = std.fmt.parseInt(usize, pa.template[i + 1 .. j], 10) catch return null;
            const nout = outputs.items.len;
            if (n < nout) {
                try tmpl.print(a, "%[out{d}]", .{n});
            } else {
                try tmpl.print(a, "%[in{d}]", .{n - nout});
            }
            i = j - 1;
            continue;
        }
        if (ch == '"' or ch == '\\') try tmpl.append(a, '\\');
        try tmpl.append(a, ch);
    }

    // result type
    const ret = if (outputs.items.len == 0)
        "void"
    else if (outputs.items.len == 1)
        outputs.items[0].ty
    else blk: {
        // aggregate: cache per shape
        var shape = std.ArrayList(u8).empty;
        for (outputs.items) |o| try shape.print(a, "{s},", .{o.ty});
        for (aggs.items) |agg| {
            if (std.mem.eql(u8, agg.llvm, shape.items)) break :blk agg.name;
        }
        const name = try std.fmt.allocPrint(a, "Agg{d}", .{aggs.items.len});
        var zig = std.ArrayList(u8).empty;
        try zig.appendSlice(a, "struct {");
        for (outputs.items, 0..) |o, k| {
            try zig.print(a, " f{d}: {s},", .{ k, o.ty });
        }
        try zig.appendSlice(a, " }");
        try aggs.append(a, .{ .llvm = shape.items, .zig = zig.items, .name = name });
        break :blk name;
    };

    return .{
        .template = tmpl.items,
        .outputs = outputs.items,
        .inputs = inputs.items,
        .ret = ret,
        .is_volatile = has_mem_clobber or outputs.items.len == 0,
    };
}

fn emitAsmEntry(src: *std.ArrayList(u8), a: std.mem.Allocator, e: AsmEntry) !void {
    try src.print(a, "    /// {s} ({s}) [asm]\n    pub fn @\"{s}\"(", .{ e.id, e.family, e.name });
    for (e.inputs, 0..) |in, j| {
        if (in.is_comptime) {
            try src.print(a, "comptime in{d}: {s}, ", .{ j, in.ty });
        } else {
            try src.print(a, "in{d}: {s}, ", .{ j, in.ty });
        }
    }
    try src.print(a, ") {s} {{\n", .{e.ret});
    const multi = e.outputs.len > 1;
    if (multi) {
        for (e.outputs, 0..) |o, k| {
            try src.print(a, "        var out{d}: {s} = undefined;\n", .{ k, o.ty });
        }
    }
    const single = e.outputs.len == 1;
    try src.print(a, "        {s}{s} (\"{s}\"\n", .{
        if (single) "return " else "",
        if (e.is_volatile) "asm volatile" else "asm",
        e.template,
    });
    try src.print(a, "            : ", .{});
    if (multi) {
        for (e.outputs, 0..) |o, k| {
            try src.print(a, "[out{d}] \"={c}\" (out{d}), ", .{ k, o.constraint, k });
        }
    } else if (e.outputs.len == 1) {
        try src.print(a, "[out0] \"={c}\" (-> {s}), ", .{ e.outputs[0].constraint, e.outputs[0].ty });
    }
    try src.print(a, "\n            : ", .{});
    for (e.inputs, 0..) |in, j| {
        try src.print(a, "[in{d}] \"{c}\" (in{d}), ", .{ j, in.constraint, j });
    }
    try src.print(a, "\n        );\n", .{});
    if (multi) {
        try src.print(a, "        return .{{", .{});
        for (e.outputs, 0..) |_, k| {
            try src.print(a, " .f{d} = out{d},", .{ k, k });
        }
        try src.print(a, " }};\n", .{});
    }
    try src.print(a, "    }}\n", .{});
}


pub fn genMain(gpa: std.mem.Allocator, io: std.Io, args: []const []const u8, out: *std.Io.Writer) !u8 {
    var dir_path: ?[]const u8 = null;
    var out_path: []const u8 = "src/cuda/gen/intrinsics.zig";
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "-o")) {
            i += 1;
            if (i >= args.len) {
                try out.print("error: -o requires a value\n", .{});
                return 1;
            }
            out_path = args[i];
        } else if (dir_path == null) {
            dir_path = args[i];
        } else {
            try out.print("error: unexpected argument '{s}'\n", .{args[i]});
            return 1;
        }
    }
    const dir = dir_path orelse {
        try out.print("error: expected 'zoxide gen <cuda-oxide/intrinsics dir> [-o out.zig]'\n", .{});
        return 1;
    };

    // --- catalog.json: id -> family/module/name ---
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();

    const cat_path = try std.fs.path.join(a, &.{ dir, "catalog.json" });
    const cat_text = std.Io.Dir.cwd().readFileAlloc(io, cat_path, a, .unlimited) catch |e| {
        try out.print("error: cannot read {s}: {s}\n", .{ cat_path, @errorName(e) });
        return 1;
    };
    const parsed = std.json.parseFromSlice(std.json.Value, a, cat_text, .{}) catch |e| {
        try out.print("error: cannot parse catalog.json: {s}\n", .{@errorName(e)});
        return 1;
    };
    var meta = std.StringHashMap(struct { family: []const u8, module: []const u8, name: []const u8 }).init(a);
    const intr = parsed.value.object.get("intrinsics").?.array;
    for (intr.items) |e| {
        const o = e.object;
        const id = jsonStr(o.get("id")) orelse continue;
        const family = jsonStr(o.get("family")) orelse "misc";
        var module: []const u8 = "misc";
        var name: []const u8 = id;
        if (o.get("rust")) |r| {
            module = jsonStr(r.object.get("module")) orelse module;
            name = jsonStr(r.object.get("name")) orelse name;
        }
        try meta.put(id, .{ .family = family, .module = module, .name = name });
    }

    // --- probes/*.ll: authoritative LLVM signatures ---
    const probes_path = try std.fs.path.join(a, &.{ dir, "probes" });
    var probes_dir = std.Io.Dir.cwd().openDir(io, probes_path, .{ .iterate = true }) catch |e| {
        try out.print("error: cannot open {s}: {s}\n", .{ probes_path, @errorName(e) });
        return 1;
    };
    defer probes_dir.close(io);

    var entries = std.ArrayList(Entry).empty;
    var asm_entries = std.ArrayList(AsmEntry).empty;
    var unmapped = std.StringHashMap(usize).init(a); // reason -> count
    var unmapped_fams = std.StringHashMap(usize).init(a);
    var seen_syms = std.StringHashMap(void).init(a);
    var probed_ids = std.StringHashMap(void).init(a);
    var aggs = std.ArrayList(Agg).empty;

    var it = probes_dir.iterate();
    while (try it.next(io)) |ent| {
        if (ent.kind != .file or !std.mem.endsWith(u8, ent.name, ".ll")) continue;
        const id = ent.name[0 .. ent.name.len - 3];
        const text = probes_dir.readFileAlloc(io, ent.name, a, .unlimited) catch continue;
        const decl = parseDeclare(text) orelse {
            // asm-based probe: convert positional template to named operands.
            const pa = parseAsmProbe(text) orelse {
                try bump(&unmapped, "no llvm declare and no asm call in probe");
                continue;
            };
            const conv = (try convertAsm(a, pa, &aggs)) orelse {
                try bump(&unmapped, "asm probe with unsupported constraint/template");
                continue;
            };
            try probed_ids.put(try a.dupe(u8, id), {});
            const m = meta.get(id);
            const id_owned = try a.dupe(u8, id);
            try asm_entries.append(a, .{
                .id = id_owned,
                .family = if (m) |mm| mm.family else "misc",
                .module = if (m) |mm| mm.module else "misc",
                .name = if (m) |mm| mm.name else id_owned,
                .template = conv.template,
                .outputs = conv.outputs,
                .inputs = conv.inputs,
                .ret = conv.ret,
                .is_volatile = conv.is_volatile,
            });
            continue;
        };
        try probed_ids.put(try a.dupe(u8, id), {});
        if (seen_syms.contains(decl.sym)) continue; // duplicate symbol, first wins
        const ret = (try mapType(a, decl.ret, &aggs)) orelse {
            try bump(&unmapped, try std.fmt.allocPrint(a, "unmappable return type '{s}'", .{decl.ret}));
            continue;
        };
        const arg_strs = try splitTopLevel(a, decl.args);
        var arg_types = try a.alloc([]const u8, arg_strs.len);
        var bad: ?[]const u8 = null;
        for (arg_strs, 0..) |as_, j| {
            if (try mapType(a, as_, &aggs)) |t| {
                arg_types[j] = t;
            } else {
                bad = try a.dupe(u8, as_);
                break;
            }
        }
        if (bad) |bt| {
            try bump(&unmapped, try std.fmt.allocPrint(a, "unmappable arg type '{s}'", .{bt}));
            continue;
        }
        try seen_syms.put(decl.sym, {});
        const m = meta.get(id);
        const id_owned = try a.dupe(u8, id);
        try entries.append(a, .{
            .id = id_owned,
            .family = if (m) |mm| mm.family else "misc",
            .module = if (m) |mm| mm.module else "misc",
            .name = if (m) |mm| mm.name else id_owned,
            .symbol = try a.dupe(u8, decl.sym),
            .ret = ret,
            .args = arg_types,
        });
    }

    // catalog entries without probes
    var mit = meta.iterator();
    while (mit.next()) |kv| {
        if (!probed_ids.contains(kv.key_ptr.*)) {
            try bump(&unmapped, "no probe (not lowered via plain NVVM intrinsic)");
            try bump(&unmapped_fams, kv.value_ptr.family);
        }
    }

    // --- emit Zig ---
    var src = std.ArrayList(u8).empty;
    try src.print(a, 
        \\//! @generated by `zoxide gen` — DO NOT EDIT.
        \\//! Source: cuda-oxide intrinsics catalog (Apache-2.0), schema 46.
        \\//! Wrappers call LLVM NVVM intrinsics; freestanding nvptx64-cuda only.
        \\
        \\
    , .{});
    for (aggs.items) |agg| {
        try src.print(a, "pub const {s} = {s}; // {s}\n", .{ agg.name, agg.zig, agg.llvm });
    }
    if (aggs.items.len > 0) try src.print(a, "\n", .{});

    // group by module, preserving first-seen order
    var modules = std.ArrayList([]const u8).empty;
    for (entries.items) |e| {
        var found = false;
        for (modules.items) |m| {
            if (std.mem.eql(u8, m, e.module)) found = true;
        }
        if (!found) try modules.append(a, e.module);
    }
    for (modules.items) |m| {
        try src.print(a, "/// family group: {s}\npub const @\"{s}\" = struct {{\n", .{ m, m });
        for (entries.items) |e| {
            if (!std.mem.eql(u8, e.module, m)) continue;
            try src.print(a, "    extern fn @\"{s}\"(", .{e.symbol});
            for (e.args, 0..) |t, j| {
                try src.print(a, "a{d}: {s}, ", .{ j, t });
            }
            try src.print(a, ") {s};\n", .{e.ret});
            try src.print(a, "    /// {s} ({s})\n    pub fn @\"{s}\"(", .{ e.id, e.family, e.name });
            for (e.args, 0..) |t, j| {
                try src.print(a, "a{d}: {s}, ", .{ j, t });
            }
            try src.print(a, ") {s} {{\n        return @\"{s}\"(", .{ e.ret, e.symbol });
            for (e.args, 0..) |_, j| {
                try src.print(a, "a{d}, ", .{j});
            }
            try src.print(a, ");\n    }}\n", .{});
        }
        try src.print(a, "}};\n\n", .{});
    }

    std.Io.Dir.cwd().writeFile(io, .{ .sub_path = out_path, .data = src.items }) catch |e| {
        try out.print("error: cannot write {s}: {s}\n", .{ out_path, @errorName(e) });
        return 1;
    };

    // --- asm-derived bindings: intrinsics_asm.zig next to out_path ---
    const asm_path = try std.fs.path.join(a, &.{ std.fs.path.dirname(out_path) orelse ".", "instrinsics_asm.zig" });
    var asrc = std.ArrayList(u8).empty;
    try asrc.print(a,
        \\//! @generated by `zoxide gen` — DO NOT EDIT.
        \\//! Source: cuda-oxide intrinsics catalog (Apache-2.0), schema 46.
        \\//! Wrappers use zig named-operand inline asm (%[name]); LLVM positional
        \\//! $N templates from the probes are rewritten by the generator.
        \\//! Freestanding nvptx64-cuda only.
        \\
        \\
    , .{});
    var agg_seen_asm = std.StringHashMap(void).init(a);
    var modules_asm = std.ArrayList([]const u8).empty;
    for (asm_entries.items) |e| {
        var found = false;
        for (modules_asm.items) |m| {
            if (std.mem.eql(u8, m, e.module)) found = true;
        }
        if (!found) try modules_asm.append(a, e.module);
    }
    for (modules_asm.items) |m| {
        try asrc.print(a, "/// family group: {s}\npub const @\"{s}\" = struct {{\n", .{ m, m });
        for (asm_entries.items) |e| {
            if (!std.mem.eql(u8, e.module, m)) continue;
            try emitAsmEntry(&asrc, a, e);
        }
        try asrc.print(a, "}};\n\n", .{});
    }
    // aggregate return types referenced by asm wrappers (emit after structs use)
    // Note: emitted at top would be nicer, but order doesn't matter in zig.
    for (aggs.items) |agg| {
        // only emit aggregates created during asm conversion (shape = "ty,ty,")
        if (std.mem.endsWith(u8, agg.llvm, ",")) {
            if (agg_seen_asm.contains(agg.name)) continue;
            try agg_seen_asm.put(agg.name, {});
            try asrc.print(a, "pub const {s} = {s}; // asm multi-output {s}\n", .{ agg.name, agg.zig, agg.llvm });
        }
    }
    std.Io.Dir.cwd().writeFile(io, .{ .sub_path = asm_path, .data = asrc.items }) catch |e| {
        try out.print("error: cannot write {s}: {s}\n", .{ asm_path, @errorName(e) });
        return 1;
    };

    // --- report ---
    try out.print("generated {s}: {d} wrappers in {d} groups\n", .{ out_path, entries.items.len, modules.items.len });
    try out.print("generated {s}: {d} asm wrappers in {d} groups\n", .{ asm_path, asm_entries.items.len, modules_asm.items.len });
    try out.print("unmapped total: ", .{});
    var total_unmapped: usize = 0;
    var uit = unmapped.iterator();
    while (uit.next()) |kv| total_unmapped += kv.value_ptr.*;
    try out.print("{d}\n", .{total_unmapped});
    uit = unmapped.iterator();
    while (uit.next()) |kv| {
        try out.print("  {d:4} {s}\n", .{ kv.value_ptr.*, kv.key_ptr.* });
    }
    try out.print("unmapped-by-family (no-probe entries):\n", .{});
    var fit = unmapped_fams.iterator();
    while (fit.next()) |kv| {
        try out.print("  {d:4} {s}\n", .{ kv.value_ptr.*, kv.key_ptr.* });
    }
    return 0;
}

fn bump(map: *std.StringHashMap(usize), key: []const u8) !void {
    const g = try map.getOrPut(key);
    if (!g.found_existing) g.value_ptr.* = 0;
    g.value_ptr.* += 1;
}
