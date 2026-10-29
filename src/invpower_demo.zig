const std = @import("std");
const LinAlg = @import("MATH551HW3");

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    var buf: [8192]u8 = undefined;
    var file_writer = std.Io.File.stdout().writer(io, &buf);
    var w = &file_writer.interface;

    const M = 3;
    const Mat = LinAlg.Matrix(f64, M, M);
    const Vec = LinAlg.Vector(f64, M);

    // HW5 Q3 matrix
    var A = Mat.init();
    A.populateFrom2DRowMajor(.{
        .{ 1.5236, -0.1695, -0.3194 },
        .{ -0.1695, 3.0042, -0.2328 },
        .{ -0.3194, -0.2328, 2.7463 },
    });

    // starting vector choice
    const x0 = Vec{ .data = .{ 1.0, 1.0, 1.0 } };

    // Approximate eigenvalues given in the problem
    const eigenvalues = [_]f64{ 1.4, 2.7, 3.1 };

    const iters: usize = 200;
    try w.print("Inverse power method on A with {d} iterations\n", .{iters});

    for (eigenvalues) |mu| {
        const r = LinAlg.inversePower(M, A, mu, x0, iters);
        try w.print("offset mu = {d:.4}:\t\tlambda = {d:.6}\n", .{
            mu,
            r.eigenvalue,
        });
        try w.print("\teigenvector: ", .{});
        for (0..M) |i| try w.print("{d:.6} ", .{r.eigenvector.data[i]});
        try w.print("\n", .{});
    }

    try file_writer.flush();
}
