const device_memory = @import("device_memory.zig");

const Device = device_memory.Device;
const Memory = device_memory.Memory;

/// Owns device and memory placement metadata for a runtime buffer.
pub const Placement = struct {
    device_id: i32,
    memory_id: i32,
    device: *const Device,
    memory: *Memory,

    /// Validates that the requested memory can be addressed by the device.
    pub fn init(device: *const Device, memory: *Memory) error{InvalidArgument}!Placement {
        if (!memory.isAddressableBy(device)) return error.InvalidArgument;
        return .{
            .device_id = device.id,
            .memory_id = memory.id,
            .device = device,
            .memory = memory,
        };
    }

    /// Returns true when this placement belongs to an execution device slot.
    pub fn matchesExecutionSlot(self: Placement, device_id: i32) bool {
        return self.device_id == device_id;
    }

    /// Accounts bytes transferred from host into this memory placement.
    pub fn recordHostToDeviceTransfer(self: Placement, byte_size: usize) void {
        self.memory.stats.host_to_device_bytes += @intCast(byte_size);
    }

    /// Accounts bytes transferred from this memory placement back to host.
    pub fn recordDeviceToHostTransfer(self: Placement, byte_size: usize) void {
        self.memory.stats.device_to_host_bytes += @intCast(byte_size);
    }

    /// Test-only hook for focused placement validation.
    pub fn setDeviceIdForTest(self: *Placement, device_id: i32) void {
        self.device_id = device_id;
    }
};
