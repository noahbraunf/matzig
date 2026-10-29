const std = @import("std");
const c = @import("c");
const LinAlg = @import("ops.zig");
const Matrix = @import("types.zig").Matrix;
const Vector = @import("types.zig").Vector;

pub fn SVDResult(comptime M: usize, comptime N: usize) type {
    const K = @min(M, N);
    return struct {
        U: Matrix(f64, M, N),
        S: Vector(f64, K),
        Vt: Matrix(f64, N, N),
    };
}

pub fn lapackSVD(allocator: std.mem.Allocator, comptime M: usize, comptime N: usize, A: Matrix(f64, M, N)) ?SVDResult(M, N) {
    var work_a = A.data;
    var result: SVDResult(M, N) = undefined;

    const m_val: c_int = @intCast(M);
    const n_val: c_int = @intCast(N);
    const lda: c_int = @intCast(M); // leading dimension A
    var ldu: c_int = @intCast(M); // leading dimension U
    var ldvt: c_int = @intCast(N); // leading dimension V^T
    var info: c_int = 0;
    var jobu: u8 = 'S'; // compute reduced U
    var jobvt: u8 = 'S'; // compute reduced V^T

    // optimize workspace query
    var lwork: c_int = -1;
    var work_opt: f64 = 0; // LAPACK will write correct work size into this
    _ = c.dgesvd_(
        &jobu,
        &jobvt,
        &m_val,
        &n_val,
        &work_a,
        &lda,
        &result.S.data,
        &result.U.data,
        &ldu,
        &result.Vt.data,
        &ldvt,
        &work_opt,
        &lwork,
        &info,
    );

    if (info != 0) return null;

    lwork = @intFromFloat(work_opt);
    const lwork_size: usize = @intCast(lwork);

    const work = allocator.alloc(f64, lwork_size) catch return null;
    defer allocator.free(work);

    _ = c.dgesvd_(
        &jobu,
        &jobvt,
        &m_val,
        &n_val,
        &work_a,
        &lda,
        &result.S.data, // S ouput
        &result.U.data, // U output
        &ldu,
        &result.Vt.data,
        &ldvt,
        work.ptr,
        &lwork,
        &info,
    );

    if (info != 0) return null;

    return result;
}

/// Tikhonov regularized solution via SVD.
///
///   x_lambda = V diag(sigma_i / (sigma_i^2 + lambda)) * U^T * b
pub fn tikhonov(
    comptime M: usize,
    comptime N: usize,
    svd: SVDResult(M, N),
    b: Vector(f64, M),
    lambda: f64,
) Vector(f64, N) {
    const K = @min(M, N);

    var Ut_b = LinAlg.mmulv(N, M, svd.U.transpose(), b);

    // calculate (Sigma^2 + lambda I)^{-1} Sigma Ut_b
    for (0..K) |i| {
        const si = svd.S.data[i];
        Ut_b.set(i, Ut_b.at(i).? * si / (si * si + lambda)).?;
    }

    // x = V Ut_b
    const x = LinAlg.mmulv(M, N, svd.Vt.transpose(), Ut_b);
    return x;
}
