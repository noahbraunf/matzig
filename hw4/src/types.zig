const std = @import("std");

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

        pub fn transpose(self: Self) Matrix(T, N, M) {
            var result = Matrix(T, N, M).init();

            for (0..M) |row| {
                for (0..N) |col| {
                    const val = self.at(row, col) orelse unreachable;
                    result.set(col, row, val) catch unreachable;
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
