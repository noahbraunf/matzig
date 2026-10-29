const std = @import("std");
const linalg = @import("matzig");

test "identity preserves a vector" {
    const matrix = linalg.eye(2);
    const vector = linalg.Vector(f64, 2){ .data = .{ 3, -4 } };
    const result = linalg.mmulv(2, 2, matrix, vector);
    try std.testing.expectEqual(@as(f64, 3), result.data[0]);
    try std.testing.expectEqual(@as(f64, -4), result.data[1]);
}
