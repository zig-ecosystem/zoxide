//! Lossless PTX text view: a tokenizer plus a statement splitter, intended
//! for lint checks and instruction census queries — not a compiler.
//!
//! "Lossless" is the load-bearing property: every byte of the input belongs
//! to exactly one statement, statements are stored in order, and
//! concatenating their spans reproduces the input byte-for-byte (tested).
//! Deliberately *not* done: no expression grammar (operands are kept as
//! source spans), no type checking, no verification against the ISA. The
//! model is a viewer — anything it cannot structure, it still passes
//! through, because a lint tool that rejects valid-but-unfamiliar PTX is
//! worse than one that says nothing about it.
//!
//! Statement model, matching what NVPTX actually emits for this repo:
//!
//!   directive    — `.target sm_90`, `.visible .entry ...` headers
//!   declaration  — `.param`/`.reg`/`.shared`/`.global`/`.const` statements
//!   label        — `$L__BB0_1:` alone on a line
//!   instruction  — `mnemonic.modifiers operands;` optionally `@%p1`-guarded
//!   brace        — a statement whose last content byte is `{` or `}`
//!   trivia       — whitespace/comment-only stretches
//!
//! Spans may carry leading trivia (a statement owns the bytes since the end
//! of the previous one); classification always looks at the first real word.
//! Inline-asm blocks (module-scope asm emits raw multi-line text, e.g.
//! hgemm_wgmma4's %zacc declarations and tma.zig's $zwait loops) split into
//! ordinary statements — no special casing is needed, because the contract
//! is byte-exactness, not nesting.

const std = @import("std");

pub const StmtKind = enum {
    directive,
    declaration,
    label,
    instruction,
    open_brace,
    close_brace,
    trivia,
};

pub const Stmt = struct {
    kind: StmtKind,
    /// [start, end) into the source. Concatenating all statements' spans in
    /// order reproduces the input — the round-trip property.
    start: u32,
    end: u32,
    /// 1-based line of the span start (may be leading trivia).
    line: u32,
    /// Leading mnemonic without modifiers (`ld`, `mbarrier.try_wait`,
    /// `target`); null for trivia/braces.
    mnemonic: ?[]const u8 = null,
    /// Modifier suffixes without the dot, in order ({"rn", "f32"}).
    modifiers: []const []const u8 = &.{},
    /// True if the instruction carries a leading @%pN guard.
    predicated: bool = false,
    /// Operand text as a source span (empty for non-instructions).
    operands: []const u8 = "",

    pub fn text(self: Stmt, src: []const u8) []const u8 {
        return src[self.start..self.end];
    }
};

pub const Document = struct {
    src: []const u8,
    stmts: []Stmt,

    /// The lossless contract, checked rather than assumed.
    pub fn roundTrips(self: Document) bool {
        var pos: u32 = 0;
        for (self.stmts) |s| {
            if (s.start != pos or s.end < s.start) return false;
            pos = s.end;
        }
        return pos == self.src.len;
    }

    /// Instruction census: count by mnemonic + modifiers
    /// (`ld.global.v4.f32`). Keys are freshly allocated; the caller frees
    /// them with the map.
    pub fn census(self: Document, gpa: std.mem.Allocator) !std.StringHashMap(u32) {
        var map = std.StringHashMap(u32).init(gpa);
        errdefer map.deinit();
        for (self.stmts) |s| {
            if (s.kind != .instruction or s.mnemonic == null) continue;
            const key = try joinMods(gpa, s.mnemonic.?, s.modifiers);
            const gop = try map.getOrPut(key);
            if (gop.found_existing) {
                gop.value_ptr.* += 1;
                gpa.free(key);
            } else {
                gop.value_ptr.* = 1;
            }
        }
        return map;
    }
};

/// Census key helper: `mnemonic` + `.a.b.c`.
pub fn joinMods(gpa: std.mem.Allocator, mnemonic: []const u8, mods: []const []const u8) ![]u8 {
    var buf = std.array_list.Managed(u8).init(gpa);
    defer buf.deinit();
    try buf.appendSlice(mnemonic);
    for (mods) |m| {
        try buf.append('.');
        try buf.appendSlice(m);
    }
    return buf.toOwnedSlice();
}

fn isIdentStart(c: u8) bool {
    return std.ascii.isAlphabetic(c) or c == '_' or c == '%' or c == '$';
}
fn isIdent(c: u8) bool {
    return isIdentStart(c) or std.ascii.isDigit(c) or c == '.';
}

/// Split `src` into lossless statements. Linear in input size.
///
/// Termination rules, by what follows:
///   `;`                    — instruction or declaration (any brace depth;
///                            inline-asm interiors are ordinary statements)
///   newline                — directive (statement began with `.`), label
///                            candidate, or pure trivia
///   `{` / `}`              — brace statement absorbs the pending text
///   `:` + newline          — label, only if the statement so far is one
///                            bare word (so `shared::cta` never splits)
pub fn parse(gpa: std.mem.Allocator, src: []const u8) !Document {
    var stmts = std.array_list.Managed(Stmt).init(gpa);
    errdefer {
        for (stmts.items) |s| gpa.free(s.modifiers);
        stmts.deinit();
    }

    var i: usize = 0;
    var line: u32 = 1;
    var stmt_start: usize = 0;
    var stmt_line: u32 = 1;
    var saw_content = false; // any non-trivia byte in the pending span
    var first_char: u8 = 0; // first content byte ('.', '@', ident start, ...)
    var single_word = true; // pending span so far is one bare identifier word
    var operand_depth: u32 = 0; // braces of vector operands `{a, b}` inside an instruction
    var saw_eq = false; // '=' in the pending span: an initializer brace follows

    while (i < src.len) {
        const c = src[i];
        if (c == '\n') {
            if (saw_content and (first_char == '.') and operand_depth == 0) {
                // Directive ends at the newline; the newline itself becomes
                // leading trivia of the next statement, keeping spans
                // contiguous.
                try emitStmt(gpa, &stmts, src, stmt_start, i, stmt_line);
                stmt_start = i;
                saw_content = false;
                single_word = true;
                first_char = 0;
                saw_eq = false;
            }
            i += 1;
            line += 1;
            if (!saw_content) {
                stmt_line = line;
            }
            continue;
        }
        if (c == ' ' or c == '\t' or c == '\r') {
            if (saw_content) single_word = false;
            i += 1;
            continue;
        }
        if (c == '/' and i + 1 < src.len and src[i + 1] == '/') {
            while (i < src.len and src[i] != '\n') i += 1;
            continue;
        }
        if (c == '/' and i + 1 < src.len and src[i + 1] == '*') {
            i += 2;
            while (i + 1 < src.len and !(src[i] == '*' and src[i + 1] == '/')) {
                if (src[i] == '\n') line += 1;
                i += 1;
            }
            i = @min(i + 2, src.len);
            continue;
        }
        if (c == '"') {
            i += 1;
            while (i < src.len and src[i] != '"') {
                if (src[i] == '\\') i += 1;
                if (i < src.len and src[i] == '\n') line += 1;
                i += 1;
            }
            i = @min(i + 1, src.len);
            saw_content = true;
            single_word = false;
            continue;
        }
        if (c == ';' and operand_depth == 0) {
            i += 1;
            try emitStmt(gpa, &stmts, src, stmt_start, i, stmt_line);
            stmt_start = i;
            stmt_line = line;
            saw_content = false;
            single_word = true;
            first_char = 0;
            saw_eq = false;
            continue;
        }
        if (c == '{' or c == '}') {
            // A brace after an instruction mnemonic is a vector-operand
            // group `{a, b, c}`; so is one after '=' (a declaration
            // initializer like `.global .b8 x[4] = {1, 2};`). After other
            // directive text, or alone, it is a body/asm block.
            if (saw_content and (isIdentStart(first_char) or saw_eq)) {
                if (c == '{') operand_depth += 1 else operand_depth -|= 1;
                i += 1;
                single_word = false;
                continue;
            }
            const kind: StmtKind = if (c == '{') .open_brace else .close_brace;
            i += 1;
            try emitBrace(&stmts, stmt_start, i, stmt_line, kind);
            stmt_start = i;
            stmt_line = line;
            saw_content = false;
            single_word = true;
            first_char = 0;
            saw_eq = false;
            continue;
        }
        if (c == ':' and single_word and first_char != '.' and
            (i + 1 >= src.len or src[i + 1] == '\n'))
        {
            i += 1;
            try emitBrace(&stmts, stmt_start, i, stmt_line, .label);
            stmt_start = i;
            stmt_line = line;
            saw_content = false;
            single_word = true;
            first_char = 0;
            saw_eq = false;
            continue;
        }
        if (c == '=') saw_eq = true;
        if (!saw_content) {
            saw_content = true;
            first_char = c;
            if (!isIdentStart(c)) single_word = false;
        } else if (single_word and !isIdent(c)) {
            single_word = false;
        }
        i += 1;
    }
    if (stmt_start < src.len) {
        try emitStmt(gpa, &stmts, src, stmt_start, src.len, stmt_line);
    }
    return .{ .src = src, .stmts = try stmts.toOwnedSlice() };
}

fn emitBrace(stmts: *std.array_list.Managed(Stmt), s: usize, e: usize, line: u32, kind: StmtKind) !void {
    try stmts.append(.{ .kind = kind, .start = @intCast(s), .end = @intCast(e), .line = line });
}

/// Classify and emit a content statement (terminated by `;` or newline).
fn emitStmt(gpa: std.mem.Allocator, stmts: *std.array_list.Managed(Stmt), src: []const u8, s: usize, e: usize, line: u32) !void {
    const text = src[s..e];
    var st = Stmt{ .kind = .instruction, .start = @intCast(s), .end = @intCast(e), .line = line };

    var it = TokenIter{ .text = text };
    var first = it.nextWord() orelse {
        st.kind = .trivia;
        try stmts.append(st);
        return;
    };
    if (first.word.len > 0 and first.word[0] == '@') {
        st.predicated = true;
        first = it.nextWord() orelse {
            st.kind = .trivia;
            try stmts.append(st);
            return;
        };
    }
    // Strip a leading dot (directives) then split into base + modifiers.
    var word = first.word;
    if (word.len > 0 and word[0] == '.') {
        st.kind = .directive;
        word = word[1..];
        // `.visible .entry` / `.weak .func`: linkage prefixes are not the
        // mnemonic — take the next directive word instead.
        if (std.mem.eql(u8, word, "visible") or std.mem.eql(u8, word, "weak") or std.mem.eql(u8, word, "extern")) {
            if (it.nextWord()) |w| {
                word = w.word;
                if (word.len > 0 and word[0] == '.') word = word[1..];
            }
        }
    }
    var parts = std.mem.splitScalar(u8, word, '.');
    const base = parts.next() orelse word;
    var mods = std.array_list.Managed([]const u8).init(gpa);
    errdefer mods.deinit();
    while (parts.next()) |m| {
        if (m.len != 0) try mods.append(m);
    }
    st.mnemonic = if (base.len == 0) null else base;
    st.modifiers = try mods.toOwnedSlice();
    if (st.kind == .directive and st.mnemonic != null) {
        const d = st.mnemonic.?;
        if (std.mem.eql(u8, d, "param") or std.mem.eql(u8, d, "reg") or
            std.mem.eql(u8, d, "shared") or std.mem.eql(u8, d, "global") or
            std.mem.eql(u8, d, "const") or std.mem.eql(u8, d, "local"))
            st.kind = .declaration;
    }
    st.operands = std.mem.trim(u8, it.rest(), " \t\r\n;");
    try stmts.append(st);
}

/// Minimal word iterator over a statement's text: words are runs of
/// identifier characters (including dots); everything else is skipped for
/// classification purposes — the full text always stays in the span.
pub const TokenIter = struct {
    text: []const u8,
    pos: usize = 0,

    pub fn nextWord(self: *TokenIter) ?struct { word: []const u8 } {
        while (self.pos < self.text.len) {
            const c = self.text[self.pos];
            if (c == ' ' or c == '\t' or c == '\r' or c == '\n' or
                c == '(' or c == ')' or c == ',' or c == '[' or c == ']' or c == '+' or c == '-')
            {
                self.pos += 1;
                continue;
            }
            if (c == '/' and self.pos + 1 < self.text.len and self.text[self.pos + 1] == '/') {
                // Comment: skip, do not stop — a comment may lead a statement.
                while (self.pos < self.text.len and self.text[self.pos] != '\n') self.pos += 1;
                continue;
            }
            const start = self.pos;
            if (c == '@') {
                self.pos += 1;
                if (self.pos < self.text.len and self.text[self.pos] == '!') self.pos += 1;
                while (self.pos < self.text.len and isIdent(self.text[self.pos])) self.pos += 1;
                return .{ .word = self.text[start..self.pos] };
            }
            while (self.pos < self.text.len and (isIdent(self.text[self.pos]) or self.text[self.pos] == '.')) self.pos += 1;
            if (self.pos == start) self.pos += 1; // other punctuation: skip
            if (self.pos > start) return .{ .word = self.text[start..self.pos] };
        }
        return null;
    }

    pub fn rest(self: *TokenIter) []const u8 {
        return self.text[self.pos..];
    }
};

// ---------------------------------------------------------------------------
// Lint
// ---------------------------------------------------------------------------

pub const Finding = struct {
    line: u32,
    msg: []const u8,
};

/// The checks `zoxide lint` runs. Each is something this repository has been
/// burned by, or a structural property a broken emitter violates.
pub fn lint(gpa: std.mem.Allocator, doc: Document) ![]Finding {
    var findings = std.array_list.Managed(Finding).init(gpa);
    errdefer findings.deinit();

    var depth: i32 = 0;
    var in_body = false; // inside an .entry/.func { ... }
    var body_pending = false; // saw an .entry/.func header, waiting for '{'
    var saw_target = false;

    for (doc.stmts) |s| {
        switch (s.kind) {
            .open_brace => {
                if (body_pending) {
                    in_body = true;
                    body_pending = false;
                }
                depth += 1;
            },
            .close_brace => {
                depth -= 1;
                if (depth < 0) {
                    try findings.append(.{ .line = s.line, .msg = "unbalanced braces: '}' with no matching '{'" });
                    depth = 0;
                }
                if (depth == 0) in_body = false;
            },
            .directive, .declaration => {
                if (s.mnemonic) |m| {
                    if (std.mem.eql(u8, m, "entry") or std.mem.eql(u8, m, "func")) body_pending = true;
                    if (std.mem.eql(u8, m, "target")) {
                        saw_target = true;
                        // `.target sm_90, debug` disables ptxas optimisation;
                        // it once reached a committed kernel via a debug-flag
                        // leak.
                        if (std.mem.indexOf(u8, s.text(doc.src), "debug") != null)
                            try findings.append(.{ .line = s.line, .msg = ".target carries the 'debug' flag (disables ptxas optimisation)" });
                    }
                }
            },
            .instruction => {
                if (!in_body)
                    try findings.append(.{ .line = s.line, .msg = "instruction outside any .entry/.func body" });
                // Positional-asm residue: LLVM does not substitute $N or
                // %[name] on NVPTX; they reach PTX verbatim and ptxas rejects
                // them (see README's asm notes). Lint catches it cheaper.
                const t = s.text(doc.src);
                if (std.mem.indexOf(u8, t, "%[") != null)
                    try findings.append(.{ .line = s.line, .msg = "unsubstituted %[name] asm operand" });
                if (hasPositionalResidue(t))
                    try findings.append(.{ .line = s.line, .msg = "unsubstituted $N asm operand" });
            },
            else => {},
        }
    }
    if (depth != 0) try findings.append(.{ .line = 0, .msg = "unbalanced braces at end of file" });
    if (!saw_target) try findings.append(.{ .line = 0, .msg = "no .target directive" });
    return findings.toOwnedSlice();
}

fn hasPositionalResidue(t: []const u8) bool {
    var i: usize = 0;
    while (i + 1 < t.len) : (i += 1) {
        if (t[i] == '$' and std.ascii.isDigit(t[i + 1])) return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const fixture =
    \\// a comment
    \\.version 8.2
    \\.target sm_90
    \\.address_size 64
    \\
    \\.visible .entry demo(
    \\    .param .u64 demo_param_0,
    \\    .param .u32 demo_param_1
    \\)
    \\{
    \\    .reg .f32 %f<4>;
    \\    .reg .pred %p<2>;
    \\    // body comment
    \\    ld.param.u64 %rd1, [demo_param_0];
    \\$L__BB0_1:
    \\    @%p1 bra $L__BB0_2;
    \\    ld.global.v4.f32 {%f1, %f2, %f3, %f4}, [%rd1];
    \\    add.rn.f32 %f1, %f1, %f2;
    \\    mbarrier.init.shared::cta.b64 [%r1], 1;
    \\    ret;
    \\$L__BB0_2:
    \\}
    \\
;

const FixtureDoc = struct {
    doc: Document,
    fn deinit(self: FixtureDoc) void {
        for (self.doc.stmts) |s| std.testing.allocator.free(s.modifiers);
        std.testing.allocator.free(self.doc.stmts);
    }
};

fn parseFixture(src: []const u8) !FixtureDoc {
    return .{ .doc = try parse(std.testing.allocator, src) };
}

test "parse round-trips the fixture byte-for-byte" {
    const d = try parseFixture(fixture);
    defer d.deinit();
    try std.testing.expect(d.doc.roundTrips());
}

test "classification: directives, labels, instructions, modifiers" {
    const d = try parseFixture(fixture);
    defer d.deinit();
    var saw_target = false;
    var saw_label = false;
    var saw_vec = false;
    var saw_pred = false;
    var saw_param_decl = false;
    var saw_shared_coloncolon = false;
    for (d.doc.stmts) |s| {
        switch (s.kind) {
            .directive => if (s.mnemonic != null and std.mem.eql(u8, s.mnemonic.?, "target")) {
                saw_target = true;
                try std.testing.expectEqual(@as(usize, 0), s.modifiers.len);
                try std.testing.expectEqualStrings("sm_90", s.operands);
            },
            .declaration => if (s.mnemonic != null and std.mem.eql(u8, s.mnemonic.?, "param")) {
                saw_param_decl = true;
            },
            .label => saw_label = true,
            .instruction => {
                if (s.predicated) saw_pred = true;
                if (s.mnemonic) |m| {
                    if (std.mem.eql(u8, m, "ld") and std.mem.eql(u8, s.operands, "{%f1, %f2, %f3, %f4}, [%rd1]")) saw_vec = true;
                    if (std.mem.eql(u8, m, "mbarrier")) saw_shared_coloncolon = true;
                }
            },
            else => {},
        }
    }
    try std.testing.expect(saw_target and saw_label and saw_vec and saw_pred and saw_param_decl);
    // `shared::cta` must not have split the statement at the colon.
    try std.testing.expect(saw_shared_coloncolon);
}

test "census counts mnemonic+modifier classes" {
    const d = try parseFixture(fixture);
    defer d.deinit();
    var map = try d.doc.census(std.testing.allocator);
    defer {
        var it = map.iterator();
        while (it.next()) |e| std.testing.allocator.free(e.key_ptr.*);
        map.deinit();
    }
    try std.testing.expectEqual(@as(u32, 1), map.get("add.rn.f32").?);
    try std.testing.expectEqual(@as(u32, 1), map.get("ld.global.v4.f32").?);
    try std.testing.expectEqual(@as(u32, 1), map.get("ret").?);
    try std.testing.expectEqual(@as(u32, 1), map.get("bra").?);
}

test "lint accepts a clean file" {
    const d = try parseFixture(fixture);
    defer d.deinit();
    const findings = try lint(std.testing.allocator, d.doc);
    defer std.testing.allocator.free(findings);
    try std.testing.expectEqual(@as(usize, 0), findings.len);
}

test "lint catches the failure classes it exists for" {
    const bad =
        \\.version 8.2
        \\    mov.u32 %r1, $0;
        \\    add.f32 %f1, %[x], %f2;
        \\}
    ;
    const d = try parseFixture(bad);
    defer d.deinit();
    const findings = try lint(std.testing.allocator, d.doc);
    defer std.testing.allocator.free(findings);
    // missing .target, 2x instruction-outside-body, $0 residue, %[ residue,
    // unbalanced close brace.
    try std.testing.expect(findings.len >= 5);
}

test "lint flags a debug target" {
    const src = ".version 8.2\n.target sm_90, debug\n";
    const d = try parseFixture(src);
    defer d.deinit();
    const findings = try lint(std.testing.allocator, d.doc);
    defer std.testing.allocator.free(findings);
    try std.testing.expectEqual(@as(usize, 1), findings.len);
}

test "inline-asm blocks split losslessly and lint clean" {
    const src =
        \\.target sm_90a
        \\.visible .entry k()
        \\{
        \\    .reg .f32 %zacc<2>;
        \\$Ld:
        \\    mov.f32 %zacc0, 0f00000000;
        \\    {
        \\    .reg .pred %pw;
        \\$zwait:
        \\    mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 %pw, [%r1], 0;
        \\    @!%pw bra $zwait;
        \\    }
        \\    ret;
        \\}
    ;
    const d = try parseFixture(src);
    defer d.deinit();
    try std.testing.expect(d.doc.roundTrips());
    const findings = try lint(std.testing.allocator, d.doc);
    defer std.testing.allocator.free(findings);
    try std.testing.expectEqual(@as(usize, 0), findings.len);
}

test "round-trip every kernel PTX if the build output exists" {
    // zig-out/kernels is a build product; absent on a fresh checkout. Skip,
    // don't fail — the byte-exactness property is what is being tested, not
    // the presence of the build.
    const io = std.testing.io;
    var dir = std.Io.Dir.cwd().openDir(io, "zig-out/kernels", .{ .iterate = true }) catch return;
    defer dir.close(io);
    var it = dir.iterate();
    var checked: usize = 0;
    while (try it.next(io)) |entry| {
        if (!std.mem.endsWith(u8, entry.name, ".ptx")) continue;
        const src = try dir.readFileAlloc(io, entry.name, std.testing.allocator, .unlimited);
        defer std.testing.allocator.free(src);
        const d = try parseFixture(src);
        defer d.deinit();
        try std.testing.expect(d.doc.roundTrips());
        checked += 1;
    }
    std.debug.print("round-tripped {d} kernel PTX files\n", .{checked});
}
