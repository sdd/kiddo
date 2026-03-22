/// AVX2 leaf-kernel operations contract.
///
/// This is intentionally empty for now and will be expanded as AVX2-specific
/// leaf kernels are wired into query paths.
pub trait Avx2LeafOps {}

/// Placeholder implementation for metrics without AVX2 specializations.
pub struct UnsupportedAvx2LeafOps;

impl Avx2LeafOps for UnsupportedAvx2LeafOps {}
