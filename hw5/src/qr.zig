const std = @import("std");
const c = @import("c");
const Matrix = @import("types.zig").Matrix;
const Vector = @import("types.zig").Vector;
const fillRandom = @import("ops.zig").fillRandom;

pub const GramSchmidtType = enum {
    Naive,
    Modified,
};

pub fn gramSchmidt(comptime T: type, comptime AlgorithmType: GramSchmidtType, comptime M: usize, comptime N: usize, input: Matrix(T, M, N)) ?Matrix(T, M, N) {
    var q = Matrix(T, M, N).init();
    const VecT = Vector(T, M);

    switch (AlgorithmType) {
        // Classical Gram-Schmidt: projections are computed against the
        // original input vector v_j, so all projections for column j use
        // the same (unmodified) right-hand side in the inner product.
        .Naive => {
            for (0..N) |j| {
                const vj: VecT = input.getColVec(j) orelse return null;
                var wj: VecT = vj;
                for (0..j) |i| {
                    const qi: VecT = q.getColVec(i) orelse return null;
                    const rij: T = qi.dot(vj);
                    wj = wj.sub(qi.scaledBy(rij));
                }
                const rjj: T = wj.norm2();
                if (rjj == 0) return null;
                q.setColVec(j, wj.scaledBy(1.0 / rjj));
            }
        },
        // Modified Gram-Schmidt: each projection is computed against the
        // progressively updated w_j, re-orthogonalizing after every
        // subtraction.
        .Modified => {
            for (0..N) |j| {
                var wj: VecT = input.getColVec(j) orelse return null;
                for (0..j) |i| {
                    const qi: VecT = q.getColVec(i) orelse return null;
                    const rij: T = qi.dot(wj);
                    wj = wj.sub(qi.scaledBy(rij));
                }
                const rjj: T = wj.norm2();
                if (rjj == 0) return null;
                q.setColVec(j, wj.scaledBy(1.0 / rjj));
            }
        },
    }
    return q;
}

pub fn HouseholderVecs(comptime M: usize, comptime N: usize) type {
    // Zig 0.16 replaced the generic @Type builtin with targeted constructors;
    // @Tuple takes a slice of field types and returns the tuple type.
    var field_types: [N]type = undefined;
    inline for (0..N) |k| {
        field_types[k] = Vector(f64, M - k);
    }
    return @Tuple(&field_types);
}

/// QR factorization with Householder reflectors
pub fn houseHolderQR(comptime M: usize, comptime N: usize, A: Matrix(f64, M, N)) struct {
    R: Matrix(f64, M, N),
    vs: HouseholderVecs(M, N),
} {
    var R = A;
    var vs: HouseholderVecs(M, N) = undefined;

    inline for (0..N) |k| {
        const L = M - k;

        // Extract x = R[k:M, k] into a Vector(f64, L)
        var x = Vector(f64, L).init();
        for (0..L) |i| {
            x.data[i] = R.at(k + i, k) orelse unreachable;
        }

        // v = x + sign(x[0]) * ||x|| * e_1
        var v = x;
        const norm_x = x.norm2();
        const sign: f64 = if (x.data[0] >= 0.0) 1.0 else -1.0;
        v.data[0] += sign * norm_x;

        // Normalize v
        const norm_v = v.norm2();
        if (norm_v > 1e-14) { // near zero means will get innacurate reults
            for (&v.data) |*elem| {
                elem.* /= norm_v;
            }

            // Apply H_k = I - 2vv^T to R[k:M, k:N]
            for (k..N) |j| {
                var dot: f64 = 0.0;
                for (0..L) |i| {
                    dot += v.data[i] * (R.at(k + i, j) orelse unreachable);
                }
                for (0..L) |i| {
                    const old = R.at(k + i, j) orelse unreachable;
                    R.set(k + i, j, old - 2.0 * dot * v.data[i]) catch unreachable;
                }
            }
        }

        vs[k] = v;
    }

    return .{ .R = R, .vs = vs };
}

/// Apply Q = Q_0 * Q_1 * ... * Q_{N-1} to a vector (reflectors applied in reverse order).
pub fn applyQ(comptime M: usize, comptime N: usize, vs: HouseholderVecs(M, N), vec: Vector(f64, M)) Vector(f64, M) {
    var result = vec;
    inline for (0..N) |kk| {
        const k = N - 1 - kk;
        const L = M - k;
        var dot: f64 = 0.0;
        for (0..L) |i| {
            dot += vs[k].data[i] * result.data[k + i];
        }
        for (0..L) |i| {
            result.data[k + i] -= 2.0 * dot * vs[k].data[i];
        }
    }
    return result;
}

/// Apply Q^T = Q_{N-1} * ... * Q_0 to a vector (reflectors applied in forward order).
pub fn applyQT(comptime M: usize, comptime N: usize, vs: HouseholderVecs(M, N), vec: Vector(f64, M)) Vector(f64, M) {
    var result = vec;
    inline for (0..N) |k| {
        const L = M - k;
        var dot: f64 = 0.0;
        for (0..L) |i| {
            dot += vs[k].data[i] * result.data[k + i];
        }
        for (0..L) |i| {
            result.data[k + i] -= 2.0 * dot * vs[k].data[i];
        }
    }
    return result;
}

/// Build the explicit Q matrix (M x N) from Householder reflectors.
pub fn buildExplicitQ(comptime M: usize, comptime N: usize, vs: HouseholderVecs(M, N)) Matrix(f64, M, N) {
    var Q = Matrix(f64, M, N).init();
    for (0..N) |j| {
        var ej = Vector(f64, M).init();
        ej.data[j] = 1.0;
        Q.setColVec(j, applyQ(M, N, vs, ej));
    }
    return Q;
}

/// Compute thin QR factorization using LAPACK
pub fn lapackQR(comptime M: usize, comptime N: usize, A: Matrix(f64, M, N)) struct { Q: Matrix(f64, M, N), R: Matrix(f64, N, N) } {
    var work_a = A.data; // column-major copy
    var tau: [N]f64 = undefined;
    var info: c_int = 0;
    var m_val: c_int = @intCast(M);
    var n_val: c_int = @intCast(N);
    var lda_val: c_int = @intCast(M);
    var work: [256]f64 = undefined;
    var lwork_val: c_int = 256;

    // QR factorization (overwrites work_a with packed reflectors)
    _ = c.dgeqrf_(&m_val, &n_val, &work_a, &lda_val, &tau, &work, &lwork_val, &info);

    // Extract R from the upper triangle of work_a
    var R = Matrix(f64, N, N).init();
    for (0..N) |i| {
        for (i..N) |j| {
            R.set(i, j, work_a[j * M + i]) catch unreachable;
        }
    }

    // Build Q from the packed reflectors
    _ = c.dorgqr_(&m_val, &n_val, &n_val, &work_a, &lda_val, &tau, &work, &lwork_val, &info);

    var Q = Matrix(f64, M, N).init();
    Q.data = work_a;

    return .{ .Q = Q, .R = R };
}
