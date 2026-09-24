//! Build-time helper: add `.visible` to named device globals in a PTX file.
//!
//! Exists separately from the `zoxide ptx-export` subcommand for one reason: the
//! build graph has to run this on the *build host*, while the zoxide binary
//! follows `-Dtarget`. Cross-compiling the bundle (`-Dtarget=x86_64-linux-gnu`
//! on an aarch64 macOS host) otherwise fails with "the host system is unable to
//! execute binaries from the target".
//!
//! Usage: ptx-promote <in.ptx> --globals a,b [-o out.ptx]
const std = @import("std");
const ptx = @import("ptx");

pub fn main(init: std.process.Init) !u8 {
    const args = try std.process.Args.toSlice(init.minimal.args, init.arena.allocator());
    if (args.len < 2) {
        std.debug.print("usage: ptx-promote <in.ptx> --globals a,b [-o out.ptx]\n", .{});
        return 1;
    }
    return ptx.cliMain(init.gpa, init.io, args[1..]);
}
