/// NEON leaf-kernel operations contract.
///
/// This is intentionally empty for now and will be expanded as NEON-specific
/// leaf kernels are wired into query paths.
pub trait NeonLeafOps {}

/// Placeholder implementation for metrics without NEON specializations.
pub struct UnsupportedNeonLeafOps;

impl NeonLeafOps for UnsupportedNeonLeafOps {}
