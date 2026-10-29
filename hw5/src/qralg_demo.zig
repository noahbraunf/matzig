const std = @import("std");
const LinAlg = @import("MATH551HW3");

fn printMat(comptime M: usize, comptime N: usize, mat: LinAlg.Matrix(f64, M, N), w: anytype) !void {
    for (0..M) |i| {
        try w.print("  ", .{});
        for (0..N) |j| {
            try w.print("{d:>12.6} ", .{mat.at(i, j).?});
        }
        try w.print("\n", .{});
    }
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    var buf: [8192]u8 = undefined;
    var file_writer = std.Io.File.stdout().writer(io, &buf);
    var w = &file_writer.interface;

    const M = 4;
    const Mat = LinAlg.Matrix(f64, M, M);

    var A = Mat.init();
    A.populateFrom2DRowMajor(.{
        .{ 2.3346, 1.1384, 2.5606, 1.4507 },
        .{ 1.1384, 0.7860, 1.2743, 0.9531 },
        .{ 2.5606, 1.2743, 2.8147, 1.6487 },
        .{ 1.4507, 0.9531, 1.6487, 1.8123 },
    });

    try w.print("Input A (Q5):\n", .{});
    try printMat(M, M, A, w);

    // tridiagonalization
    const H = LinAlg.tridiagonalize(M, A);

    try w.print("\ntridiagonal form H = U^* A U\n", .{});
    try printMat(M, M, H, w);

    // QR algorithm
    const iters: usize = 500;
    const res = LinAlg.qrAlgorithm(M, H, iters);

    try w.print("\nEigenvalues after QR iteration ({d} iters)\n", .{
        iters,
    });

    try w.print("\nEigenvalues of A (diagonal of final matrix):\n", .{});
    for (&res.eigenvalues.data, 0..) |*eigenvalue, i| try w.print("\tlambda_{d} = {d:.6}\n", .{ i + 1, eigenvalue.* });

    try file_writer.flush();
}
