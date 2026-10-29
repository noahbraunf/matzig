const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const translate = b.addTranslateC(.{
        .root_source_file = b.path("src/c.h"),
        .target = target,
        .optimize = optimize,
    });
    if (target.result.os.tag == .macos) {
        const sdk = std.mem.trimEnd(u8, b.run(&.{ "xcrun", "--show-sdk-path" }), "\n\r ");
        translate.addSystemFrameworkPath(.{ .cwd_relative = b.fmt("{s}/System/Library/Frameworks/Accelerate.framework/Frameworks", .{sdk}) });
    }
    const library = b.addModule("math551", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "c", .module = translate.createModule() }},
    });
    library.linkFramework("Accelerate", .{});

    const examples = b.step("examples", "Build and install the example programs");
    addExample(b, target, optimize, library, examples, "least-squares", "src/main.zig");
    addExample(b, target, optimize, library, examples, "deblur", "src/deblurring.zig");
    addExample(b, target, optimize, library, examples, "invpower", "src/invpower_demo.zig");
    addExample(b, target, optimize, library, examples, "qralg", "src/qralg_demo.zig");

    const test_module = b.createModule(.{
        .root_source_file = b.path("tests/library.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "math551", .module = library }},
    });
    const tests = b.addTest(.{ .root_module = test_module });
    const test_step = b.step("test", "Run library tests");
    test_step.dependOn(&b.addRunArtifact(tests).step);
    b.getInstallStep().dependOn(&tests.step);
}

fn addExample(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, library: *std.Build.Module, examples: *std.Build.Step, name: []const u8, source: []const u8) void {
    const module = b.createModule(.{
        .root_source_file = b.path(source),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "MATH551HW3", .module = library }},
    });
    const exe = b.addExecutable(.{ .name = name, .root_module = module });
    examples.dependOn(&b.addInstallArtifact(exe, .{}).step);
    const run_step = b.step(b.fmt("run-{s}", .{name}), b.fmt("Run the {s} example", .{name}));
    run_step.dependOn(&b.addRunArtifact(exe).step);
}
