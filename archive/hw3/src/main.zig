const std = @import("std");
const LinAlg = @import("MATH551HW3");

pub fn main() !void {
    var buf: [8192]u8 = undefined;
    var file_writer = std.fs.File.stdout().writer(&buf);
    var w = &file_writer.interface;

    const M = 5;
    const N = 4;

    const Atype = LinAlg.Matrix(f64, M, N);
    const VecM = LinAlg.Vector(f64, M);

    const Adata = [_]f64{
        -0.897, 1.066,  1.33,   1.368,  -0.964,
        -1.004, -0.484, 1.37,   0.669,  0.84,
        -0.284, -1.327, -0.878, -0.392, 1.497,
        0.822,  -0.679, 0.678,  0.221,  -0.427,
    };

    const A: Atype = .{ .data = Adata };

    const b1 = VecM{ .data = .{ -0.847, -0.829, -0.579, 0.831, -1.088 } };
    const b2 = VecM{ .data = .{ 1.008, 0.594, 0.633, 2.692, -0.765 } };
    const b3 = VecM{ .data = .{ 0.623, 0.826, 0.385, -0.856, 0.215 } };

    const bs = [_]VecM{ b1, b2, b3 };

    const xs = LinAlg.solveLSQR(M, N, A, bs);

    for (xs, 0..) |x, i| {
        try w.print("x_{d} = ", .{i + 1});
        for (0..N) |j| {
            if (x.at(j)) |xj| {
                try w.print("{d:.6} ", .{xj});
            }
        }
        try w.print("\n", .{});
    }

    try w.flush();
}
