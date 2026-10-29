const std = @import("std");
const C2 = @import("C2");

fn printMatrix(comptime M: usize, comptime N: usize, mat: C2.Matrix(f64, M, N), w: *std.Io.Writer, name: []const u8) !void {
    try w.print("{s} =\n", .{name});
    for (0..M) |i| {
        for (0..N) |j| {
            try w.print("{d:>12.6}", .{mat.at(i, j) orelse 0.0});
        }
        try w.print("\n", .{});
    }
    try w.print("\n", .{});
}

fn demoQR(comptime M: usize, comptime N: usize, A: C2.Matrix(f64, M, N), w: *std.Io.Writer, label: []const u8) !void {
    try w.print("\t{s}\n", .{label});

    try printMatrix(M, N, A, w, "A");

    // Householder QR
    const qr = C2.houseHolderQR(M, N, A);
    const Q = C2.buildExplicitQ(M, N, qr.vs);
    const R_top = qr.R.getSubMatrix(N, N, 0, 0) orelse unreachable;

    try printMatrix(M, N, Q, w, "Q (Householder)");
    try printMatrix(N, N, R_top, w, "R (Householder)");

    // LAPACK QR (built-in)
    const lap = C2.lapackQR(M, N, A);

    try printMatrix(M, N, lap.Q, w, "Q (LAPACK)");
    try printMatrix(N, N, lap.R, w, "R (LAPACK)");
}

pub fn main() !void {
    var buf: [8192]u8 = undefined;
    var file_writer = std.fs.File.stdout().writer(&buf);
    var w = &file_writer.interface;

    var prng = std.Random.DefaultPrng.init(12345);
    const rand = prng.random();

    // A1 in R^{4x3}
    var A1 = C2.Matrix(f64, 4, 3).init();
    C2.fillRandom(4, 3, &A1, rand);
    try demoQR(4, 3, A1, w, "A1 in R^{4x3}");

    // A2 in R^{5x5}
    var A2 = C2.Matrix(f64, 5, 5).init();
    C2.fillRandom(5, 5, &A2, rand);
    try demoQR(5, 5, A2, w, "A2 in R^{5x5}");

    try w.flush();
}
