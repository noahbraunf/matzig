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

        pub fn set(self: *Self, row: usize, col: usize, value: T) ?void {
            if (row >= M or col >= N) {
                return null;
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

        pub inline fn ndim() [2]usize {
            return [_]usize{ M, N };
        }

        pub inline fn nrows() usize {
            return M;
        }

        pub inline fn ncols() usize {
            return N;
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

        pub fn print(self: Self, w: *std.Io.Writer) !void {
            for (0..M) |n| {
                try w.print("({d:.4})\n", .{self.data[n]});
            }
        }
    };
}

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

pub fn main() !void {
    var stdout_buffer: [1024]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;
    var prng: std.Random.DefaultPrng = .init(42);
    const rand = prng.random();

    const Matrix4x3d = Matrix(f64, 4, 3);
    const Vector3d = Vector(f64, 3);
    var m4x3data: [Matrix4x3d.nrows() * Matrix4x3d.ncols()]f64 = undefined;
    var v3data: [Vector3d.dim()]f64 = undefined;

    for (0..m4x3data.len) |i| {
        m4x3data[i] = rand.float(f64);
    }

    for (0..v3data.len) |i| {
        v3data[i] = rand.float(f64);
    }

    const mat4x3: Matrix4x3d = .{ .data = m4x3data };
    const vec3: Vector3d = .{ .data = v3data };

    const Matrix5x5d = Matrix(f64, 5, 5);
    const Vector5d = Vector(f64, 5);

    var m5x5data: [Matrix5x5d.nrows() * Matrix5x5d.ncols()]f64 = undefined;
    var v5data: [Vector5d.dim()]f64 = undefined;

    for (0..m5x5data.len) |i| {
        m5x5data[i] = rand.float(f64);
    }

    for (0..v5data.len) |i| {
        v5data[i] = rand.float(f64);
    }

    const mat5x5: Matrix5x5d = .{ .data = m5x5data };
    const vec5: Vector5d = .{ .data = v5data };

    try stdout.print("A1, x1\n\n", .{});

    try stdout.print("My Implementation:\n", .{});
    try mmulv(4, 3, mat4x3, vec3).print(stdout);

    try stdout.print("\nBLAS implementation (via Apple accelerate framework):\n", .{});
    try blas_mmulv(4, 3, mat4x3, vec3).print(stdout);

    try stdout.print("\n\nA2, x2\n\n", .{});

    try stdout.print("My Implementation:\n", .{});
    try mmulv(5, 5, mat5x5, vec5).print(stdout);

    try stdout.print("\nBLAS implementation (via Apple accelerate framework):\n", .{});
    try blas_mmulv(5, 5, mat5x5, vec5).print(stdout);

    try stdout.flush();
}
