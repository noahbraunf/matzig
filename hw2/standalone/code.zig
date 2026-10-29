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

/// Fill every entry of a Matrix with uniform random values in [-1, 1).
pub fn fillRandom(comptime M: usize, comptime N: usize, mat: *Matrix(f64, M, N), rand: std.Random) void {
    for (&mat.data) |*elem| {
        elem.* = rand.float(f64) * 2.0 - 1.0;
    }
}

pub fn fillRandom(comptime M: usize, comptime N: usize, mat: *Matrix(f64, M, N), rand: std.Random) void {
    for (&mat.data) |*elem| {
        elem.* = rand.float(f64) * 2.0 - 1.0;
    }
}

/// Build the Q matrix (M x N) from Householder reflectors.
pub fn buildExplicitQ(comptime M: usize, comptime N: usize, vs: HouseholderVecs(M, N)) Matrix(f64, M, N) {
    var Q = Matrix(f64, M, N).init();
    for (0..N) |j| {
        var ej = Vector(f64, M).init();
        ej.data[j] = 1.0;
        Q.setColVec(j, applyQ(M, N, vs, ej));
    }
    return Q;
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

fn printMatrix(comptime M: usize, comptime N: usize, mat: Matrix(f64, M, N), w: *std.Io.Writer, name: []const u8) !void {
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
    const qr = houseHolderQR(M, N, A);
    const Q = buildExplicitQ(M, N, qr.vs);
    const R_top = qr.R.getSubMatrix(N, N, 0, 0) orelse unreachable;

    try printMatrix(M, N, Q, w, "Q (Householder)");
    try printMatrix(N, N, R_top, w, "R (Householder)");

    // LAPACK QR (built-in)
    const lap = lapackQR(M, N, A);

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
    fillRandom(4, 3, &A1, rand);
    try demoQR(4, 3, A1, w, "A1 in R^{4x3}");

    // A2 in R^{5x5}
    var A2 = C2.Matrix(f64, 5, 5).init();
    fillRandom(5, 5, &A2, rand);
    try demoQR(5, 5, A2, w, "A2 in R^{5x5}");

    try w.flush();
}
