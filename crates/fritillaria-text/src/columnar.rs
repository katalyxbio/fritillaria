//! Line and field discovery, on the host and on the device.
//!
//! [`batch`] is the CPU reference and the oracle; [`device`] is what a GPU
//! consumer receives. Both describe the same thing — a flat tab table with
//! per-record boundaries — so they can be compared directly.

pub mod batch;
pub mod device;

pub use batch::{Dialect, FieldTable, LineKind, RecordBatch, scan_lines};
pub use device::{DeviceColumns, DeviceRecordBatch};
