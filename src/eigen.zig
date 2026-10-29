const std = @import("std");
const c = @import("c");
const Matrix = @import("types.zig").Matrix;
const Vector = @import("types.zig").Vector;
const mmulv = @import("ops.zig").mmulv;
const mmul = @import("ops.zig").mmul;
const lapackQR = @import("qr.zig").lapackQR;

pub fn InversePowerResult(comptime M: usize) type {
    return struct {
        eigenvalue: f64,
        eigenvector: Vector(f64, M),
    };
}

pub fn inversePower(
    comptime M: usize,
    A: Matrix(f64, M, M),
    mu: f64,
    x0: Vector(f64, M),
    iterations: usize,
) InversePowerResult(M) {
    // normalize starting vector
    var x = x0;
    const n0 = x.norm2();
    if (n0 == 0) @panic("zero initial vector");
    for (&x.data) |*e| e.* /= n0;

    var lambda_prev: f64 = 0.0;
    var iter: usize = 0;

    while (iter < iterations) : (iter += 1) {
        // Build (A - mu I)
        var B: [M * M]f64 = A.data;
        inline for (0..M) |i| B[i * M + i] -= mu;

        // lapack linear system solver
        var y: [M]f64 = x.data;
        var ipiv: [M]c_int = undefined;
        var n_val: c_int = @intCast(M);
        var nrhs: c_int = 1;
        var lda: c_int = @intCast(M);
        var ldb: c_int = @intCast(M);
        var info: c_int = 0;
        _ = c.dgesv_(&n_val, &nrhs, &B, &lda, &ipiv, &y, &ldb, &info);
        if (info != 0) @panic("inversePower: dgesv failed (singular shift?)");

        // x(k+1) = y(k) / ||y(k)||_2
        var w = Vector(f64, M){ .data = y };
        const nw = w.norm2();
        for (&w.data) |*e| e.* /= nw;
        x = w;

        // Rayleigh quotient
        const Ax = mmulv(M, M, A, x);
        const lambda: f64 = x.dot(Ax);

        lambda_prev = lambda;
    }

    const Ax_f = mmulv(M, M, A, x);
    return .{
        .eigenvalue = x.dot(Ax_f),
        .eigenvector = x,
    };
}

/// Reduce a symmetric matrix A to tridiagonal form H = Q^T A Q via a
/// sequence of Householder reflectors.
pub fn tridiagonalize(comptime M: usize, A_in: Matrix(f64, M, M)) Matrix(f64, M, M) {
    var A = A_in;
    if (M < 3) return A; // already tridiagonal

    inline for (0..(M - 2)) |k| {
        const subColLength = M - k - 1;
        var x: Vector(f64, subColLength) = .{ .data = undefined };

        for (0..subColLength) |i| {
            x.set(i, A.at(k + 1 + i, k).?).?;
        }

        const xNorm = x.norm2();
        const sgn: f64 = if (xNorm >= 0.0) 1.0 else -1.0;
        var v = x;
        v.set(0, xNorm * sgn + x.at(0).?).?;
        const vNorm = v.norm2();
        for (&v.data) |*vi| {
            vi.* /= vNorm;
        }

        var w: Vector(f64, M - k) = .{ .data = undefined };
        for (k..M) |j| {
            var dot: f64 = 0.0;
            for (0..subColLength) |i| {
                dot += v.at(i).? * A.at(k + 1 + i, j).?;
            }
            w.set(j - k, dot).?;
        }

        for (0..subColLength) |i| {
            for (k..M) |j| {
                const val = A.at(k + 1 + i, j).? - 2.0 * v.at(i).? * w.at(j - k).?;
                A.set(k + 1 + i, j, val) catch unreachable;
            }
        }

        // red line in slides
        var u: Vector(f64, M) = .{ .data = undefined };
        for (0..M) |i| {
            var dot: f64 = 0.0;
            for (0..subColLength) |j| {
                dot += A.at(i, k + 1 + j).? * v.at(j).?;
            }
            u.set(i, dot).?;
        }

        for (0..M) |i| {
            for (0..subColLength) |j| {
                const val = A.at(i, k + 1 + j).? - 2.0 * u.at(i).? * v.at(j).?;
                A.set(i, k + 1 + j, val) catch unreachable;
            }
        }

        // zero out rest to ensure hessenberg form
        for (1..subColLength) |i| {
            A.set(k + 1 + i, k, 0.0) catch unreachable;
            A.set(k, k + 1 + i, 0.0) catch unreachable;
        }
    }

    return A;
}

pub fn qrAlgorithm(
    comptime M: usize,
    H_in: Matrix(f64, M, M),
    iterations: usize,
) struct { eigenvalues: Vector(f64, M) } {
    var H = H_in;
    var iter: usize = 0;
    while (iter < iterations) : (iter += 1) {
        const qr = lapackQR(M, M, H);
        H = mmul(M, M, M, qr.R, qr.Q);
    }
    var eigenvalues: [M]f64 = undefined;
    for (0..M) |i| {
        eigenvalues[i] = H.at(i, i).?;
    }
    return .{ .eigenvalues = .{ .data = eigenvalues } };
}
