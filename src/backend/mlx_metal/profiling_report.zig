const std = @import("std");

const max_schedule_report_rows = 10;

const FieldSamples = struct {
    values: std.ArrayList(u64) = .empty,

    fn deinit(self: *FieldSamples, allocator: std.mem.Allocator) void {
        self.values.deinit(allocator);
    }

    fn record(self: *FieldSamples, allocator: std.mem.Allocator, value: ?u64) !void {
        if (value) |actual| {
            try self.values.append(allocator, actual);
        }
    }

    fn median(self: *FieldSamples) ?u64 {
        if (self.values.items.len == 0) return null;
        sortU64(self.values.items);
        return self.values.items[self.values.items.len / 2];
    }
};

const BackendExecuteSamples = struct {
    count: usize = 0,
    compiled_program_us: FieldSamples = .{},
    compiled_program_host_enqueue_us: FieldSamples = .{},
    compiled_program_device_sync_wait_us: FieldSamples = .{},
    schedule_items: FieldSamples = .{},
    nodes: FieldSamples = .{},
    fusion_groups: FieldSamples = .{},
    materialization_boundaries: FieldSamples = .{},

    fn deinit(self: *BackendExecuteSamples, allocator: std.mem.Allocator) void {
        self.compiled_program_us.deinit(allocator);
        self.compiled_program_host_enqueue_us.deinit(allocator);
        self.compiled_program_device_sync_wait_us.deinit(allocator);
        self.schedule_items.deinit(allocator);
        self.nodes.deinit(allocator);
        self.fusion_groups.deinit(allocator);
        self.materialization_boundaries.deinit(allocator);
    }

    fn record(self: *BackendExecuteSamples, allocator: std.mem.Allocator, line: []const u8) !void {
        self.count += 1;
        try self.compiled_program_us.record(allocator, parseU64Field(line, "compiled_program_us"));
        try self.compiled_program_host_enqueue_us.record(allocator, parseU64Field(line, "compiled_program_host_enqueue_us"));
        try self.compiled_program_device_sync_wait_us.record(allocator, parseU64Field(line, "compiled_program_device_sync_wait_us"));
        try self.schedule_items.record(allocator, parseU64Field(line, "schedule_items"));
        try self.nodes.record(allocator, parseU64Field(line, "nodes"));
        try self.fusion_groups.record(allocator, parseU64Field(line, "fusion_groups"));
        try self.materialization_boundaries.record(allocator, parseU64Field(line, "materialization_boundaries"));
    }
};

const ScheduleNodeOpSample = struct {
    op: []const u8,
    count: u64 = 0,
};

const ScheduleFusionGroupSample = struct {
    ops: []const u8,
    groups: u64 = 0,
    scheduled_nodes: u64 = 0,
};

const ScheduleReport = struct {
    schedule_item_count: usize = 0,
    node_item_count: usize = 0,
    fusion_group_item_count: usize = 0,
    node_ops: std.ArrayList(ScheduleNodeOpSample) = .empty,
    fusion_groups: std.ArrayList(ScheduleFusionGroupSample) = .empty,

    fn deinit(self: *ScheduleReport, allocator: std.mem.Allocator) void {
        self.node_ops.deinit(allocator);
        self.fusion_groups.deinit(allocator);
    }

    fn record(self: *ScheduleReport, allocator: std.mem.Allocator, line: []const u8) !void {
        self.schedule_item_count += 1;

        const kind = parseTextField(line, "kind") orelse return;
        if (std.mem.eql(u8, kind, "node")) {
            self.node_item_count += 1;
            const op = parseTextField(line, "op") orelse return;
            try self.recordNodeOp(allocator, op);
        } else if (std.mem.eql(u8, kind, "fusion_group")) {
            self.fusion_group_item_count += 1;
            const ops = parseTextField(line, "ops") orelse return;
            const scheduled_nodes = parseU64Field(line, "count") orelse 0;
            try self.recordFusionGroup(allocator, ops, scheduled_nodes);
        }
    }

    fn recordNodeOp(self: *ScheduleReport, allocator: std.mem.Allocator, op: []const u8) !void {
        for (self.node_ops.items) |*sample| {
            if (std.mem.eql(u8, sample.op, op)) {
                sample.count += 1;
                return;
            }
        }
        try self.node_ops.append(allocator, .{ .op = op, .count = 1 });
    }

    fn recordFusionGroup(self: *ScheduleReport, allocator: std.mem.Allocator, ops: []const u8, scheduled_nodes: u64) !void {
        for (self.fusion_groups.items) |*sample| {
            if (std.mem.eql(u8, sample.ops, ops)) {
                sample.groups += 1;
                sample.scheduled_nodes += scheduled_nodes;
                return;
            }
        }
        try self.fusion_groups.append(allocator, .{ .ops = ops, .groups = 1, .scheduled_nodes = scheduled_nodes });
    }

    fn writeTo(self: *ScheduleReport, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        if (self.schedule_item_count == 0) return;

        try writer.print(
            "pjrtx_profile_schedule_report schedule_item_count={d} node_item_count={d} fusion_group_item_count={d}\n",
            .{ self.schedule_item_count, self.node_item_count, self.fusion_group_item_count },
        );

        sortScheduleNodeOps(self.node_ops.items);
        const node_limit = @min(self.node_ops.items.len, max_schedule_report_rows);
        for (self.node_ops.items[0..node_limit], 1..) |sample, rank| {
            try writer.print("pjrtx_profile_schedule_node_op rank={d} op={s} count={d}\n", .{ rank, sample.op, sample.count });
        }

        sortScheduleFusionGroups(self.fusion_groups.items);
        const group_limit = @min(self.fusion_groups.items.len, max_schedule_report_rows);
        for (self.fusion_groups.items[0..group_limit], 1..) |sample, rank| {
            try writer.print(
                "pjrtx_profile_schedule_fusion_group rank={d} groups={d} scheduled_nodes={d} ops=\"{s}\"\n",
                .{ rank, sample.groups, sample.scheduled_nodes, sample.ops },
            );
        }
    }
};

const ProfileReport = struct {
    backend_execute: BackendExecuteSamples = .{},
    schedule: ScheduleReport = .{},
    tok_per_second: ?f64 = null,

    fn deinit(self: *ProfileReport, allocator: std.mem.Allocator) void {
        self.backend_execute.deinit(allocator);
        self.schedule.deinit(allocator);
    }

    fn ingest(self: *ProfileReport, allocator: std.mem.Allocator, text: []const u8) !void {
        var lines = std.mem.splitScalar(u8, text, '\n');
        while (lines.next()) |raw_line| {
            const line = stripCarriageReturn(raw_line);
            if (std.mem.indexOf(u8, line, "pjrtx_profile")) |profile_start| {
                const profile_line = line[profile_start..];
                if (isProfileEvent(profile_line, "backend_execute")) {
                    try self.backend_execute.record(allocator, profile_line);
                } else if (isProfileEvent(profile_line, "backend_schedule_item")) {
                    try self.schedule.record(allocator, profile_line);
                }
            }
            if (tokPerSecondFromLine(line)) |rate| {
                self.tok_per_second = rate;
            }
        }
    }

    fn writeTo(self: *ProfileReport, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        try writer.print("pjrtx_profile_report backend_execute_count={d}", .{self.backend_execute.count});
        try writeOptionalU64(writer, "median_compiled_program_us", self.backend_execute.compiled_program_us.median());
        try writeOptionalU64(writer, "median_compiled_program_host_enqueue_us", self.backend_execute.compiled_program_host_enqueue_us.median());
        try writeOptionalU64(writer, "median_compiled_program_device_sync_wait_us", self.backend_execute.compiled_program_device_sync_wait_us.median());
        try writeOptionalU64(writer, "median_schedule_items", self.backend_execute.schedule_items.median());
        try writeOptionalU64(writer, "median_nodes", self.backend_execute.nodes.median());
        try writeOptionalU64(writer, "median_fusion_groups", self.backend_execute.fusion_groups.median());
        try writeOptionalU64(writer, "median_materialization_boundaries", self.backend_execute.materialization_boundaries.median());
        if (self.tok_per_second) |rate| {
            try writer.print(" tok_per_second={d:.3}", .{rate});
        } else {
            try writer.writeAll(" tok_per_second=na");
        }
        try writer.writeByte('\n');
        try self.schedule.writeTo(writer);
    }
};

/// Reads a PjRTx/ZML profile log and prints one stable summary line.
pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    var args = std.process.Args.Iterator.init(init.minimal.args);
    const argv0 = args.next() orelse "profiling_report";
    const path = args.next() orelse {
        std.debug.print("usage: {s} <profile-log>\n", .{argv0});
        return error.InvalidArgument;
    };
    if (args.next() != null) {
        std.debug.print("usage: {s} <profile-log>\n", .{argv0});
        return error.InvalidArgument;
    }

    const text = try std.Io.Dir.cwd().readFileAlloc(init.io, path, allocator, .limited(512 * 1024 * 1024));
    defer allocator.free(text);

    var report: ProfileReport = .{};
    defer report.deinit(allocator);
    try report.ingest(allocator, text);

    var rendered = std.Io.Writer.Allocating.init(allocator);
    defer rendered.deinit();
    try report.writeTo(&rendered.writer);
    try std.Io.File.stdout().writeStreamingAll(init.io, rendered.written());
}

fn isProfileEvent(line: []const u8, comptime event_name: []const u8) bool {
    const value = parseTextField(line, "event") orelse return false;
    return std.mem.eql(u8, value, event_name);
}

fn parseTextField(line: []const u8, comptime field_name: []const u8) ?[]const u8 {
    const raw = parseField(line, field_name) orelse return null;
    return std.mem.trim(u8, raw, "\"");
}

fn parseU64Field(line: []const u8, comptime field_name: []const u8) ?u64 {
    const raw = parseField(line, field_name) orelse return null;
    return std.fmt.parseInt(u64, raw, 10) catch null;
}

fn parseField(line: []const u8, comptime field_name: []const u8) ?[]const u8 {
    var fields = std.mem.splitScalar(u8, line, ' ');
    while (fields.next()) |field| {
        const eq = std.mem.indexOfScalar(u8, field, '=') orelse continue;
        if (std.mem.eql(u8, field[0..eq], field_name)) {
            return std.mem.trim(u8, field[eq + 1 ..], "\"");
        }
    }
    return null;
}

fn tokPerSecondFromLine(line: []const u8) ?f64 {
    const marker = std.mem.indexOf(u8, line, "tok/s") orelse return null;
    var end = marker;
    while (end > 0 and !isDecimalByte(line[end - 1])) {
        end -= 1;
    }
    var start = end;
    while (start > 0 and isDecimalByte(line[start - 1])) {
        start -= 1;
    }
    if (start == end) return null;
    return std.fmt.parseFloat(f64, line[start..end]) catch null;
}

fn stripCarriageReturn(line: []const u8) []const u8 {
    return if (line.len != 0 and line[line.len - 1] == '\r') line[0 .. line.len - 1] else line;
}

fn isDecimalByte(byte: u8) bool {
    return (byte >= '0' and byte <= '9') or byte == '.';
}

fn writeOptionalU64(writer: *std.Io.Writer, comptime name: []const u8, value: ?u64) std.Io.Writer.Error!void {
    if (value) |actual| {
        try writer.print(" {s}={d}", .{ name, actual });
    } else {
        try writer.print(" {s}=na", .{name});
    }
}

fn sortU64(values: []u64) void {
    if (values.len < 2) return;
    var index: usize = 1;
    while (index < values.len) : (index += 1) {
        const value = values[index];
        var slot = index;
        while (slot > 0 and values[slot - 1] > value) : (slot -= 1) {
            values[slot] = values[slot - 1];
        }
        values[slot] = value;
    }
}

fn sortScheduleNodeOps(values: []ScheduleNodeOpSample) void {
    if (values.len < 2) return;
    var index: usize = 1;
    while (index < values.len) : (index += 1) {
        const value = values[index];
        var slot = index;
        while (slot > 0 and scheduleNodeOpBefore(value, values[slot - 1])) : (slot -= 1) {
            values[slot] = values[slot - 1];
        }
        values[slot] = value;
    }
}

fn scheduleNodeOpBefore(lhs: ScheduleNodeOpSample, rhs: ScheduleNodeOpSample) bool {
    if (lhs.count != rhs.count) return lhs.count > rhs.count;
    return std.mem.order(u8, lhs.op, rhs.op) == .lt;
}

fn sortScheduleFusionGroups(values: []ScheduleFusionGroupSample) void {
    if (values.len < 2) return;
    var index: usize = 1;
    while (index < values.len) : (index += 1) {
        const value = values[index];
        var slot = index;
        while (slot > 0 and scheduleFusionGroupBefore(value, values[slot - 1])) : (slot -= 1) {
            values[slot] = values[slot - 1];
        }
        values[slot] = value;
    }
}

fn scheduleFusionGroupBefore(lhs: ScheduleFusionGroupSample, rhs: ScheduleFusionGroupSample) bool {
    if (lhs.groups != rhs.groups) return lhs.groups > rhs.groups;
    if (lhs.scheduled_nodes != rhs.scheduled_nodes) return lhs.scheduled_nodes > rhs.scheduled_nodes;
    return std.mem.order(u8, lhs.ops, rhs.ops) == .lt;
}

test "profile report parses backend medians and token rate" {
    const log =
        \\noise
        \\pjrtx_profile event=backend_execute executable=0x1 device=0 args=1 outputs=1 schedule_items=10 nodes=20 fusion_groups=3 materialization_boundaries=1 compiled_program_us=100 compiled_program_host_enqueue_us=70 compiled_program_device_sync_wait_us=9
        \\pjrtx_profile event=backend_execute executable=0x1 device=0 args=1 outputs=1 schedule_items=12 nodes=22 fusion_groups=5 materialization_boundaries=2 compiled_program_us=120 compiled_program_host_enqueue_us=90 compiled_program_device_sync_wait_us=11
        \\decode 5.459s · 20.1tok/s
    ;

    var report: ProfileReport = .{};
    defer report.deinit(std.testing.allocator);
    try report.ingest(std.testing.allocator, log);

    var rendered = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer rendered.deinit();
    try report.writeTo(&rendered.writer);

    try std.testing.expectEqualStrings(
        "pjrtx_profile_report backend_execute_count=2 median_compiled_program_us=120 median_compiled_program_host_enqueue_us=90 median_compiled_program_device_sync_wait_us=11 median_schedule_items=12 median_nodes=22 median_fusion_groups=5 median_materialization_boundaries=2 tok_per_second=20.100\n",
        rendered.written(),
    );
}

test "profile report includes verbose schedule section when schedule lines exist" {
    const log =
        \\pjrtx_profile event=backend_schedule_item executable=0x1 schedule_index=0 kind=node index=0 count=1 elapsed_us=5 node_kind=matmul instruction=0 op=dot_general
        \\pjrtx_profile event=backend_schedule_item executable=0x1 schedule_index=1 kind=fusion_group index=0 count=3 elapsed_us=4 first_node=1 last_node=3 node_count=3 ops="add,convert,power"
        \\pjrtx_profile event=backend_schedule_item executable=0x1 schedule_index=2 kind=node index=4 count=1 elapsed_us=1 node_kind=reduction instruction=4 op=reduce_sum
        \\pjrtx_profile event=backend_schedule_item executable=0x1 schedule_index=3 kind=fusion_group index=1 count=3 elapsed_us=3 first_node=5 last_node=7 node_count=3 ops="add,convert,power"
        \\pjrtx_profile event=backend_schedule_item executable=0x1 schedule_index=4 kind=node index=8 count=1 elapsed_us=5 node_kind=matmul instruction=8 op=dot_general
        \\pjrtx_profile event=backend_execute executable=0x1 device=0 args=1 outputs=1 schedule_items=5 nodes=9 fusion_groups=2 materialization_boundaries=0 compiled_program_us=100
    ;

    var report: ProfileReport = .{};
    defer report.deinit(std.testing.allocator);
    try report.ingest(std.testing.allocator, log);

    var rendered = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer rendered.deinit();
    try report.writeTo(&rendered.writer);

    try std.testing.expectEqualStrings(
        "pjrtx_profile_report backend_execute_count=1 median_compiled_program_us=100 median_compiled_program_host_enqueue_us=na median_compiled_program_device_sync_wait_us=na median_schedule_items=5 median_nodes=9 median_fusion_groups=2 median_materialization_boundaries=0 tok_per_second=na\n" ++
            "pjrtx_profile_schedule_report schedule_item_count=5 node_item_count=3 fusion_group_item_count=2\n" ++
            "pjrtx_profile_schedule_node_op rank=1 op=dot_general count=2\n" ++
            "pjrtx_profile_schedule_node_op rank=2 op=reduce_sum count=1\n" ++
            "pjrtx_profile_schedule_fusion_group rank=1 groups=2 scheduled_nodes=6 ops=\"add,convert,power\"\n",
        rendered.written(),
    );
}

test "profile report tolerates old backend execute lines" {
    const log =
        \\pjrtx_profile event=backend_execute executable=0x1 device=0 args=1 outputs=1 schedule_items=1181 nodes=2352 fusion_groups=555 materialization_boundaries=4 compiled_program_us=48704
        \\decode 5.459s · 20.1tok/s
    ;

    var report: ProfileReport = .{};
    defer report.deinit(std.testing.allocator);
    try report.ingest(std.testing.allocator, log);

    var rendered = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer rendered.deinit();
    try report.writeTo(&rendered.writer);

    try std.testing.expectEqualStrings(
        "pjrtx_profile_report backend_execute_count=1 median_compiled_program_us=48704 median_compiled_program_host_enqueue_us=na median_compiled_program_device_sync_wait_us=na median_schedule_items=1181 median_nodes=2352 median_fusion_groups=555 median_materialization_boundaries=4 tok_per_second=20.100\n",
        rendered.written(),
    );
}
