const Matrix = @import("types.zig").Matrix;
const Vector = @import("types.zig").Vector;
const houseHolderQR = @import("qr.zig").houseHolderQR;
const applyQT = @import("qr.zig").applyQT;

pub fn solveLSQR(comptime M: usize, comptime N: usize, A: Matrix(f64, M, N), bs: anytype) [@typeInfo(@TypeOf(bs)).array.len]Vector(f64, N) {
    const VecContainer = @TypeOf(bs);
    const info = @typeInfo(VecContainer);

    const array = switch (info) {
        .array => |a| a,
        else => @compileError("Expected an array of vectors"),
    };

    if (array.child != Vector(f64, M)) {
        @compileError("Expected an array of vectors");
    }

    const K = array.len;

    if (M > N) {
        // compute Q, R using householder reflector
        const qrResult = houseHolderQR(M, N, A);
        var results: [K]Vector(f64, N) = undefined;
        for (bs, 0..) |b, k| {
            // compute Q^T * b
            const qTb = applyQT(M, N, qrResult.vs, b);
            const R = qrResult.R;

            // solve R*x = Q^T*b
            var x: Vector(f64, N) = undefined;
            for (0..N) |i| {
                x.set(i, qTb.at(i).?).?; // start x at qTb (truncated)
            }
            for (0..N) |rev_i| {
                const i = N - 1 - rev_i;

                for ((i + 1)..N) |j| {
                    // x_i already equals y_i
                    x.set(i, x.at(i).? - R.at(i, j).? * x.at(j).?).?;
                }

                x.set(i, x.at(i).? / R.at(i, i).?).?;
            }

            results[k] = x;
        }

        return results;
    } else {
        @compileError("requires M>N");
    }
}
