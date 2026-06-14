//! Advanced distance-metric extension traits used by kiddo's query engine.

#[doc(inline)]
pub use crate::dist::{
    DistanceMetric, DistanceMetricAvx2, DistanceMetricAvx512, DistanceMetricCore,
    DistanceMetricNeon, DistanceMetricSimdBlock,
};
