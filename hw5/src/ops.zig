const std = @import("std");
const c = @import("c");
const Matrix = @import("types.zig").Matrix;
const Vector = @import("types.zig").Vector;

pub fn mmulv(comptime M: usize, comptime N: usize, mat: Matrix(f64, M, N), vec: Vector(f64, N)) Vector(f64, M) {
    var output = std.mem.zeroes([M]f64);
    for (0..M) |row| {
        for (0..N) |col| {
            const m = mat.at(row, col) orelse continue;
            const v = vec.at(col) orelse continue;
            output[row] += m * v;
        }
    }
    return .{ .data = output };
}

pub fn blas_mmulv(comptime M: usize, comptime N: usize, mat: Matrix(f64, M, N), vec: Vector(f64, N)) Vector(f64, M) {
    var output = std.mem.zeroes([M]f64);
    _ = c.cblas_dgemv(c.CblasColMajor, c.CblasNoTrans, M, N, 1.0, &mat.data, M, &vec.data, @as(c_int, 1), 1.0, &output, @as(c_int, 1));

    return .{ .data = output };
}

/// Fill every entry of a Matrix with uniform random values in [-1, 1).
pub fn fillRandom(comptime M: usize, comptime N: usize, mat: *Matrix(f64, M, N), rand: std.Random) void {
    for (&mat.data) |*elem| {
        elem.* = rand.float(f64) * 2.0 - 1.0;
    }
}

/// Transpose an M×N matrix to N×M.
pub fn matTranspose(comptime M: usize, comptime N: usize, mat: Matrix(f64, M, N)) Matrix(f64, N, M) {
    var result = Matrix(f64, N, M).init();
    for (0..M) |i| {
        for (0..N) |j| {
            result.set(j, i, mat.at(i, j) orelse unreachable) catch unreachable;
        }
    }
    return result;
}

/// General matrix multiply: (M×K) * (K×N) → (M×N).
pub fn mmul(comptime M: usize, comptime K: usize, comptime N: usize, a: Matrix(f64, M, K), b: Matrix(f64, K, N)) Matrix(f64, M, N) {
    var result = Matrix(f64, M, N).init();
    for (0..M) |i| {
        for (0..N) |j| {
            var sum: f64 = 0;
            for (0..K) |kk| {
                sum += (a.at(i, kk) orelse unreachable) * (b.at(kk, j) orelse unreachable);
            }
            result.set(i, j, sum) catch unreachable;
        }
    }
    return result;
}

/// Element-wise matrix subtraction.
pub fn matSub(comptime M: usize, comptime N: usize, a: Matrix(f64, M, N), b: Matrix(f64, M, N)) Matrix(f64, M, N) {
    var result: Matrix(f64, M, N) = undefined;
    for (&result.data, a.data, b.data) |*r, ai, bi| {
        r.* = ai - bi;
    }
    return result;
}

/// Frobenius norm of a matrix.
pub fn frobNorm(comptime M: usize, comptime N: usize, mat: Matrix(f64, M, N)) f64 {
    var sum: f64 = 0;
    for (mat.data) |elem| {
        sum += elem * elem;
    }
    return @sqrt(sum);
}

/// N×N identity matrix.
pub fn eye(comptime N: usize) Matrix(f64, N, N) {
    var result = Matrix(f64, N, N).init();
    for (0..N) |i| {
        result.set(i, i, 1.0) catch unreachable;
    }
    return result;
}
