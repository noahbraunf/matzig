# Matzig

A Zig module for fixed-size matrices and vectors, QR and least-squares
solvers, SVD and Tikhonov regularization, and eigenvalue routines. It collects
the latest implementation from the original coursework without changing its
source. Earlier versions and the standalone HW2 program are preserved in
`archive/` for reference.

The build uses Zig 0.17.0 on macOS, the Apple SDK, and the Accelerate
framework. The C bindings are generated during the build.

## Use as a dependency

In your project's `build.zig.zon`, add a local dependency (adjust the path):

```zig
.dependencies = .{
    .matzig = .{ .path = "../matzig" },
},
```

In `build.zig`, pass the target and optimization options to the dependency and
add its module to your executable or library module:

```zig
const dep = b.dependency("matzig", .{ .target = target, .optimize = optimize });
// After creating your module:
my_module.addImport("matzig", dep.module("matzig"));
```

Then import and use it in Zig:

```zig
const linalg = @import("matzig");
const v = linalg.Vector(f64, 2){ .data = .{ 3, -4 } };
const result = linalg.mmulv(2, 2, linalg.eye(2), v);
```

The public API is defined in `src/root.zig`. It includes `Matrix`, `Vector`,
`gramSchmidt`, `houseHolderQR`, `lapackQR`, `solveLSQR`, `lapackSVD`,
`tikhonov`, `inversePower`, `tridiagonalize`, `qrAlgorithm`, and matrix
operations. The function signatures are in their respective source files.

## Development

```sh
zig build test                 # test the public module
zig build examples             # install example programs to zig-out/bin
zig build run-least-squares    # run an example
zig build --help               # list the other example run steps
```

The archived HW1–HW4 sources are historical snapshots, not part of the
library build. They use removed Zig APIs, and the standalone HW2 file has a
duplicate declaration. Building them on Zig 0.17 would require source edits.
