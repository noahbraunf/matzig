const std = @import("std");
const c = @cImport(@cInclude("Accelerate/Accelerate.h"));

pub fn Matrix(comptime T: type, comptime M: usize, comptime N: usize) type {
    switch (@typeInfo(T)) {
        .int, .comptime_int, .float, .comptime_float => {},
        else => @compileError("Invalid type T for a matrix"),
    }
    return struct {
        const Self = @This();

        data: [N * M]T, // column major format for more contiguous memory
        // TODO: Consider adding support for row major?

        pub fn init() Self {
            return .{ .data = std.mem.zeroes([N * M]T) };
        }

        pub fn at(self: Self, row: usize, col: usize) ?T {
            if (row >= M or col >= N) {
                return null;
            }

            return self.data[col * M + row];
            // c0    : r0 r1 ... r(M-1)
            // c1    : r0 r1 ... r(M-1)
            // ...     ^index M
            // c(N-1): r0 r1 ... r(M-1)
            //            ^ index M*(N-1)+1
            // at(1, 0) => c1r1 = index (1 * M + 0) = M <- checks out!
            //    ^  ^
            //    r1 c0
        }

        pub fn set(self: *Self, row: usize, col: usize, value: T) error{OutOfBounds}!void {
            if (row >= M or col >= N) {
                return error.OutOfBounds;
            }

            self.data[col * M + row] = value;
        }

        pub fn populateFrom2DRowMajor(self: *Self, data: [M][N]T) void {
            for (0..M) |row| {
                for (0..N) |col| {
                    self.set(row, col, data[row][col]);
                }
            }
        }
        pub fn populateFrom2DColMajor(self: *Self, data: [N][M]T) void {
            for (0..N) |col| {
                for (0..M) |row| {
                    self.set(row, col, data[col][row]);
                }
            }
        }
        pub fn getColVec(self: *const Self, index: usize) ?Vector(T, M) {
            if (index >= N) {
                return null;
            }
            var newData: [M]T = undefined;
            const start = index * M;
            @memcpy(&newData, self.data[start .. start + M]);
            return .{ .data = newData };
        }

        pub fn setColVec(self: *Self, index: usize, vec: Vector(T, M)) void {
            const start = index * M;
            @memcpy(self.data[start .. start + M], &vec.data);
        }

        pub inline fn ndim() [2]usize {
            return [_]usize{ M, N };
        }

        pub inline fn nrows() usize {
            return M;
        }

        pub inline fn ncols() usize {
            return N;
        }

        pub fn getSubMatrix(self: Self, comptime m: usize, comptime n: usize, rowPos: usize, colPos: usize) ?Matrix(T, m, n) {
            if (rowPos + m > M or colPos + n > N) return null;
            var result = Matrix(T, m, n).init();
            for (0..m) |row| {
                for (0..n) |col| {
                    const val = self.at(rowPos + row, colPos + col) orelse unreachable;
                    result.set(row, col, val) catch unreachable;
                }
            }
            return result;
        }

        pub fn print(self: Self, w: *std.Io.Writer) !void {
            for (0..M) |m| {
                try w.printAsciiChar('(', .{});
                for (0..N) |n| {
                    if (self.at(m, n)) |value| {
                        try w.print("{d:.4}", .{value});
                    }
                    if (n != N - 1) {
                        try w.print(", ", .{});
                    }
                }
                try w.print(")\n", .{});
            }
        }
    };
}

pub fn Vector(comptime T: type, comptime M: usize) type {
    switch (@typeInfo(T)) {
        .int, .comptime_int, .float, .comptime_float => {},
        else => @compileError("Invalid type T for a matrix"),
    }
    return struct {
        const Self = @This();

        data: [M]T,

        pub fn init() Self {
            return .{ .data = std.mem.zeroes([M]T) };
        }

        pub fn at(self: Self, idx: usize) ?T {
            if (idx >= M) {
                return null;
            }

            return self.data[idx];
        }

        pub fn set(self: *Self, idx: usize, value: T) ?void {
            if (idx >= M) {
                return null;
            }

            self.data[idx] = value;
        }

        pub inline fn dim() usize {
            return M;
        }

        pub fn scalarDiv(self: *Self, value: f64) ?void {
            if (value == 0) {
                return null;
            }

            for (&self.data) |elem| {
                elem.* /= value;
            }
        }

        pub fn norm(self: *const Self, l: f64) f64 {
            var sum: f64 = 0.0;
            for (self.data) |elem| {
                const val: f64 = @floatCast(@abs(elem));
                sum += std.math.pow(f64, val, l);
            }
            return std.math.pow(f64, sum, 1.0 / l);
        }

        pub fn norm2(self: Self) T {
            var sum: T = 0;
            for (self.data) |elem| {
                sum += elem * elem;
            }
            return @sqrt(sum);
        }

        pub fn dot(self: Self, other: Self) T {
            var sum: T = 0;
            for (self.data, other.data) |a, b| {
                sum += a * b;
            }
            return sum;
        }

        pub fn sub(self: Self, other: Self) Self {
            var result: [M]T = undefined;
            for (self.data, other.data, 0..) |a, b, i| {
                result[i] = a - b;
            }
            return .{ .data = result };
        }

        pub fn scaledBy(self: Self, scalar: T) Self {
            var result: [M]T = undefined;
            for (self.data, 0..) |elem, i| {
                result[i] = elem * scalar;
            }
            return .{ .data = result };
        }

        pub fn print(self: Self, w: *std.Io.Writer) !void {
            for (0..M) |n| {
                try w.print("({d:.4})\n", .{self.data[n]});
            }
        }
    };
}

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
        // progressively updated w_j, re-orthogonalising after every
        // subtraction.  This gives much better numerical accuracy than the
        // classical variant because rounding errors are not amplified by
        // operating on stale inner products.
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
    var fields: [N]std.builtin.Type.StructField = undefined;
    inline for (0..N) |k| {
        const VecType = Vector(f64, M - k);
        fields[k] = .{
            .name = std.fmt.comptimePrint("{d}", .{k}),
            .type = VecType,
            .default_value_ptr = null,
            .is_comptime = false,
            .alignment = @alignOf(VecType),
        };
    }
    return @Type(.{ .@"struct" = .{
        .layout = .auto,
        .fields = &fields,
        .decls = &.{},
        .is_tuple = true,
    } });
}

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
        if (norm_v > 1e-14) {
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

/// Apply Q = H_0 * H_1 * ... * H_{N-1} to a vector (reflectors applied in reverse order).
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

/// Apply Q^T = H_{N-1} * ... * H_0 to a vector (reflectors applied in forward order).
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

pub fn mmulv(comptime M: usize, comptime N: usize, mat: Matrix(f64, M, N), vec: Vector(f64, N)) Vector(f64, Matrix(f64, M, N)) {
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

/// Build the explicit Q matrix (M×N) from Householder reflectors.
pub fn buildExplicitQ(comptime M: usize, comptime N: usize, vs: HouseholderVecs(M, N)) Matrix(f64, M, N) {
    var Q = Matrix(f64, M, N).init();
    for (0..N) |j| {
        var ej = Vector(f64, M).init();
        ej.data[j] = 1.0;
        Q.setColVec(j, applyQ(M, N, vs, ej));
    }
    return Q;
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

/// Compute thin QR factorization using LAPACK (dgeqrf + dorgqr).
/// Returns Q (M×N) and R (N×N upper triangular).
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

/// Verify that the columns of `q` are orthonormal (Q^T Q ≈ I) and that
/// they span the same column space as `input` (Q Q^T v_j ≈ v_j for every
/// input column v_j).
fn verifyGramSchmidt(comptime M: usize, comptime N: usize, q: Matrix(f64, M, N), input: Matrix(f64, M, N), tol: f64) !void {
    const Vec = Vector(f64, M);

    // 1. Orthonormality: <q_i, q_j> ≈ δ_ij
    for (0..N) |i| {
        const qi: Vec = q.getColVec(i) orelse return error.TestUnexpectedResult;
        for (0..N) |j| {
            const qj: Vec = q.getColVec(j) orelse return error.TestUnexpectedResult;
            const expected: f64 = if (i == j) 1.0 else 0.0;
            try std.testing.expectApproxEqAbs(expected, qi.dot(qj), tol);
        }
    }

    // 2. Column-space preservation: QQ^T v_j = v_j  (each input column
    //    lies in the column space of Q, so its projection should be itself).
    for (0..N) |j| {
        const vj: Vec = input.getColVec(j) orelse return error.TestUnexpectedResult;
        var proj_data: [M]f64 = std.mem.zeroes([M]f64);
        for (0..N) |i| {
            const qi: Vec = q.getColVec(i) orelse return error.TestUnexpectedResult;
            const coeff = qi.dot(vj);
            for (0..M) |k| {
                proj_data[k] += coeff * qi.data[k];
            }
        }
        const proj: Vec = .{ .data = proj_data };
        const err = proj.sub(vj).norm2();
        try std.testing.expectApproxEqAbs(@as(f64, 0.0), err, tol);
    }
}

/// Run a single Gram-Schmidt round-trip test for the given algorithm,
/// dimensions, and PRNG seed.
fn runGramSchmidtTest(comptime AlgType: GramSchmidtType, comptime M: usize, comptime N: usize, seed: u64) !void {
    var prng = std.Random.DefaultPrng.init(seed);
    const rand = prng.random();

    var input = Matrix(f64, M, N).init();
    fillRandom(M, N, &input, rand);

    const q = gramSchmidt(f64, AlgType, M, N, input) orelse
        return error.TestUnexpectedResult;

    try verifyGramSchmidt(M, N, q, input, 1e-12);
}

// ---- Classical (Naive) Gram-Schmidt tests --------------------------------

test "classical GS: tall 4×3" {
    try runGramSchmidtTest(.Naive, 4, 3, 42);
}

test "classical GS: square 5×5" {
    try runGramSchmidtTest(.Naive, 5, 5, 123);
}

test "classical GS: tall 10×4" {
    try runGramSchmidtTest(.Naive, 10, 4, 999);
}

test "classical GS: tall 20×8" {
    try runGramSchmidtTest(.Naive, 20, 8, 7777);
}

// ---- Modified Gram-Schmidt tests -----------------------------------------

test "modified GS: tall 4×3" {
    try runGramSchmidtTest(.Modified, 4, 3, 42);
}

test "modified GS: square 5×5" {
    try runGramSchmidtTest(.Modified, 5, 5, 123);
}

test "modified GS: tall 10×4" {
    try runGramSchmidtTest(.Modified, 10, 4, 999);
}

test "modified GS: tall 20×8" {
    try runGramSchmidtTest(.Modified, 20, 8, 7777);
}

// Determinism: both algorithms produce the same Q for well-conditioned random input (up to sign/rounding). We check that column norms and mutual dot-products match.
test "classical and modified GS agree on random input" {
    const M = 6;
    const N = 4;
    var prng = std.Random.DefaultPrng.init(314159);
    const rand = prng.random();

    var input = Matrix(f64, M, N).init();
    fillRandom(M, N, &input, rand);

    const q_cgs = gramSchmidt(f64, .Naive, M, N, input) orelse
        return error.TestUnexpectedResult;
    const q_mgs = gramSchmidt(f64, .Modified, M, N, input) orelse
        return error.TestUnexpectedResult;

    const Vec = Vector(f64, M);
    const tol: f64 = 1e-12;
    for (0..N) |j| {
        const col_c: Vec = q_cgs.getColVec(j) orelse return error.TestUnexpectedResult;
        const col_m: Vec = q_mgs.getColVec(j) orelse return error.TestUnexpectedResult;
        // Columns may differ by sign; check |<c, m>| ≈ 1.
        try std.testing.expectApproxEqAbs(@as(f64, 1.0), @abs(col_c.dot(col_m)), tol);
    }
}

// Multiple random seeds
test "classical GS: 16 random seeds 6×4" {
    for (0..16) |seed| {
        try runGramSchmidtTest(.Naive, 6, 4, seed);
    }
}

test "modified GS: 16 random seeds 6×4" {
    for (0..16) |seed| {
        try runGramSchmidtTest(.Modified, 6, 4, seed);
    }
}

fn runHouseholderTest(comptime M: usize, comptime N: usize, seed: u64) !void {
    var prng = std.Random.DefaultPrng.init(seed);
    const rand = prng.random();

    var A = Matrix(f64, M, N).init();
    fillRandom(M, N, &A, rand);

    const qr = houseHolderQR(M, N, A);
    const tol = 1e-10;

    // 1. R is upper triangular: sub-diagonal entries are zero
    for (0..M) |i| {
        for (0..N) |j| {
            if (i > j) {
                try std.testing.expectApproxEqAbs(0.0, qr.R.at(i, j) orelse 0.0, tol);
            }
        }
    }

    // 2. Q * R = A: applying Q to each column of R recovers A
    for (0..N) |j| {
        const r_col = qr.R.getColVec(j) orelse return error.TestUnexpectedResult;
        const reconstructed = applyQ(M, N, qr.vs, r_col);
        const a_col = A.getColVec(j) orelse return error.TestUnexpectedResult;
        try std.testing.expectApproxEqAbs(0.0, reconstructed.sub(a_col).norm2(), tol);
    }

    // 3. Thin Q is orthonormal: build Q's columns via Q * e_j, check Q^T Q = I_N
    var q_cols: [N]Vector(f64, M) = undefined;
    for (0..N) |j| {
        var ej = Vector(f64, M).init();
        ej.data[j] = 1.0;
        q_cols[j] = applyQ(M, N, qr.vs, ej);
    }
    for (0..N) |i| {
        for (0..N) |j| {
            const expected: f64 = if (i == j) 1.0 else 0.0;
            try std.testing.expectApproxEqAbs(expected, q_cols[i].dot(q_cols[j]), tol);
        }
    }
}

test "householder QR: tall 4×3" {
    try runHouseholderTest(4, 3, 42);
}

test "householder QR: square 5×5" {
    try runHouseholderTest(5, 5, 123);
}

test "householder QR: tall 10×4" {
    try runHouseholderTest(10, 4, 999);
}

test "householder QR: tall 20×8" {
    try runHouseholderTest(20, 8, 7777);
}

test "householder QR: 16 random seeds 6×4" {
    for (0..16) |seed| {
        try runHouseholderTest(6, 4, seed);
    }
}
