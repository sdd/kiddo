//! Advanced distance-metric extension traits used by kiddo's query engine.

#[doc(inline)]
pub use crate::dist::{
    DistanceMetric, DistanceMetricAvx2, DistanceMetricAvx512, DistanceMetricNeon,
    DistanceMetricScalar,
};

#[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
#[doc(inline)]
pub use crate::dist::distance_metric_avx2::{UnsupportedAvx2F32LeafOps, UnsupportedAvx2F64LeafOps};

#[cfg(all(feature = "simd", target_feature = "avx512f"))]
#[doc(inline)]
pub use crate::dist::distance_metric_avx512::{
    UnsupportedAvx512F32LeafOps, UnsupportedAvx512F64LeafOps,
};

#[cfg(all(feature = "simd", target_arch = "aarch64", target_feature = "neon"))]
#[doc(inline)]
pub use crate::dist::distance_metric_neon::{UnsupportedNeonF32LeafOps, UnsupportedNeonF64LeafOps};
