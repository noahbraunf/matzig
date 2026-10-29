/// Constructs a 1-D Gaussian blurring matrix A (100×100), applies it to a
/// step-function signal x with additive Gaussian noise, then computes the
/// spectral condition number k(A) = sigma_max / sigma_min via LAPACK SVD.
const std = @import("std");
const LinAlg = @import("MATH551HW3");

const n = 100;
const spread: f64 = 10.0;
const sigma: f64 = 0.1;

/// Get normal random number from uniform random generator
fn normalRandom(rand: std.Random) f64 {
    const a = 1.0 - rand.float(f64);
    const b = rand.float(f64);
    return @sqrt(-2.0 * @log(a)) * @cos(2.0 * std.math.pi * b);
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer {
        const deinit_status = gpa.deinit();
        if (deinit_status == .leak) {
            @panic("Memory Leak");
        }
    }

    const allocator = gpa.allocator();

    var buf: [16384]u8 = undefined;
    var file_writer = std.fs.File.stdout().writer(&buf);
    var w = &file_writer.interface;

    var prng = std.Random.DefaultPrng.init(42);
    const rand = prng.random();

    var x = LinAlg.Vector(f64, n).init();
    for (30..60) |i| {
        x.data[i] = 1.0;
    }

    // Blurring matrix A
    var A = LinAlg.Matrix(f64, n, n).init();
    for (0..n) |i| {
        for (0..n) |j| {
            const fi: f64 = @floatFromInt(i);
            const fj: f64 = @floatFromInt(j);
            const d = (fi - fj) / spread;
            A.set(i, j, @exp(-0.5 * d * d)) catch unreachable;
        }
    }

    // Blurred signal b
    var b = LinAlg.mmulv(n, n, A, x);
    for (0..n) |i| {
        b.data[i] += sigma * normalRandom(rand);
    }

    // SVD of A allows us to calculate condition number
    const svd = LinAlg.lapackSVD(allocator, n, n, A).?;
    const sigma_max = svd.S.data[0];
    const sigma_min = svd.S.data[n - 1];
    const kappa = sigma_max / sigma_min;

    // Sweep different regularization values
    std.debug.print("SVD: sigma_max={d:.4} sigma_min={e:.4} kappa={e:.4}\n\n", .{ sigma_max, sigma_min, kappa });
    std.debug.print("Tikhonov sweep:\n", .{});
    std.debug.print("\t{s:<12}\t{s:<14}\n\n", .{ "lambda", "||x_lambda - x||_2" });

    const lambdas = [_]f64{ 1e-6, 1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.5, 1.0 }; // parameters to sweep

    var best_err: f64 = std.math.inf(f64);
    var best_lambda: f64 = 0;
    var best_x = LinAlg.Vector(f64, n).init();

    for (lambdas) |lam| {
        const x_lam = LinAlg.tikhonov(n, n, svd, b, lam);
        const e = x_lam.sub(x).norm2();
        std.debug.print("\t{e:<12.3}\t{d:.6}\n", .{ lam, e });
        if (e < best_err) {
            best_err = e;
            best_lambda = lam;
            best_x = x_lam;
        }
    }

    std.debug.print("\nBest lambda = {e:.3}\t(||x_lambda - x||_2 = {d:.6})\n", .{ best_lambda, best_err });

    // CSV of best output for plotting
    try w.print("index,x_true,b_noisy,x_lambda\n", .{});
    for (0..n) |i| {
        try w.print("{d},{d:.8},{d:.8},{d:.8}\n", .{ i, x.data[i], b.data[i], best_x.data[i] });
    }

    try w.flush();
}
