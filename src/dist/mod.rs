#[cfg(all(feature = "simd", target_feature = "avx2"))]
pub mod distance_metric_avx2;
#[cfg(all(feature = "simd", target_feature = "avx512f"))]
pub mod distance_metric_avx512;
pub mod distance_metric_core;
#[cfg(all(feature = "simd", target_feature = "neon"))]
pub mod distance_metric_neon;

pub mod dot_product;
pub mod manhattan;
pub mod squared_euclidean;
mod traits_unified_2_adapters;

pub use distance_metric_core::DistanceMetricCore;
use std::cmp::Ordering;

use crate::stem_strategies::donnelly_2_blockmarker_simd::{
    DistanceMetricSimdBlock3, DistanceMetricSimdBlock4,
};
use crate::traits_unified_2::AxisUnified;

pub use dot_product::DotProduct;
pub use manhattan::Manhattan;
pub use squared_euclidean::SquaredEuclidean;

/// AVX512 extension hooks.
///
/// Default behavior is "not specialized". Concrete metrics can override hook
/// methods in arch-specific code without changing public query bounds.
pub trait DistanceMetricAvx512<A: Copy>: DistanceMetricCore<A> {
    /// Whether a specialized AVX512 path is provided by this metric impl.
    const HAS_AVX512_SPECIALIZATION: bool = false;

    #[cfg(all(feature = "simd", target_feature = "avx512f"))]
    type Avx512F64Ops: distance_metric_avx512::Avx512F64LeafOps;
}

/// AVX2 extension hooks.
pub trait DistanceMetricAvx2<A: Copy>: DistanceMetricCore<A> {
    /// Whether a specialized AVX2 path is provided by this metric impl.
    const HAS_AVX2_SPECIALIZATION: bool = false;

    #[cfg(all(feature = "simd", target_feature = "avx2"))]
    type Avx2LeafOps: distance_metric_avx2::Avx2LeafOps;
}

/// NEON extension hooks.
pub trait DistanceMetricNeon<A: Copy>: DistanceMetricCore<A> {
    /// Whether a specialized NEON path is provided by this metric impl.
    const HAS_NEON_SPECIALIZATION: bool = false;

    #[cfg(all(feature = "simd", target_feature = "neon"))]
    type NeonLeafOps: distance_metric_neon::NeonLeafOps;
}

/// Unified distance metric trait (V3 umbrella).
///
/// Public query APIs can bind to this single trait while architecture-specific
/// hooks remain implementation details selected via monomorphization + cfg.
pub trait DistanceMetricUnified<A: Copy>:
    DistanceMetricCore<A> + DistanceMetricAvx512<A> + DistanceMetricAvx2<A> + DistanceMetricNeon<A>
{
}

impl<T, A: Copy> DistanceMetricUnified<A> for T where
    T: DistanceMetricCore<A>
        + DistanceMetricAvx512<A>
        + DistanceMetricAvx2<A>
        + DistanceMetricNeon<A>
{
}

/// V3-facing metric contract used by kd-tree query paths.
///
/// This bridges the new `crate::dist` traits to legacy internal query plumbing
/// while keeping query modules free of direct legacy trait references.
pub trait KdTreeDistanceMetric<A: Copy, const K: usize>:
    crate::traits_unified_2::DistanceMetricUnified<A, K, Output = Self::DistOutput>
    + DistanceMetricSimdBlock3<A, K, Self::DistOutput>
    + DistanceMetricSimdBlock4<A, K, Self::DistOutput>
{
    /// Distance accumulator / output type used by query paths.
    type DistOutput: AxisUnified<Coord = Self::DistOutput>;

    /// Widen a single coordinate from axis type `A`.
    fn widen_coord(a: A) -> Self::DistOutput;

    /// One-axis distance contribution.
    fn dist1(a: Self::DistOutput, b: Self::DistOutput) -> Self::DistOutput;

    /// Ordering-compatible comparison helper.
    fn cmp(a: Self::DistOutput, b: Self::DistOutput) -> Ordering;
}

impl<T, A: Copy, const K: usize> KdTreeDistanceMetric<A, K> for T
where
    T: crate::traits_unified_2::DistanceMetricUnified<A, K>
        + DistanceMetricSimdBlock3<
            A,
            K,
            <T as crate::traits_unified_2::DistanceMetricUnified<A, K>>::Output,
        > + DistanceMetricSimdBlock4<
            A,
            K,
            <T as crate::traits_unified_2::DistanceMetricUnified<A, K>>::Output,
        >,
    <T as crate::traits_unified_2::DistanceMetricUnified<A, K>>::Output:
        AxisUnified<Coord = <T as crate::traits_unified_2::DistanceMetricUnified<A, K>>::Output>,
{
    type DistOutput = <T as crate::traits_unified_2::DistanceMetricUnified<A, K>>::Output;

    #[inline(always)]
    fn widen_coord(a: A) -> Self::DistOutput {
        <T as crate::traits_unified_2::DistanceMetricUnified<A, K>>::widen_coord(a)
    }

    #[inline(always)]
    fn dist1(a: Self::DistOutput, b: Self::DistOutput) -> Self::DistOutput {
        <T as crate::traits_unified_2::DistanceMetricUnified<A, K>>::dist1(a, b)
    }

    #[inline(always)]
    fn cmp(a: Self::DistOutput, b: Self::DistOutput) -> Ordering {
        <T as crate::traits_unified_2::DistanceMetricUnified<A, K>>::cmp(a, b)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DistanceMetricCore, DistanceMetricUnified, DotProduct, Manhattan, SquaredEuclidean,
    };

    #[test]
    fn v3_squared_euclidean_f64_works() {
        type M = SquaredEuclidean<f64>;
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 2.0, -1.0];

        let aw = a.map(M::widen_coord);
        let bw = b.map(M::widen_coord);
        let d = <M as DistanceMetricCore<f64>>::dist::<3>(&aw, &bw);
        assert_eq!(d, 25.0);
    }

    #[test]
    fn v3_manhattan_f64_works() {
        type M = Manhattan<f64>;
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 2.0, -1.0];

        let aw = a.map(M::widen_coord);
        let bw = b.map(M::widen_coord);
        let d = <M as DistanceMetricCore<f64>>::dist::<3>(&aw, &bw);
        assert_eq!(d, 7.0);
    }

    #[test]
    fn v3_unified_bound_is_satisfied() {
        fn assert_unified<M: DistanceMetricUnified<f64>>() {}
        assert_unified::<SquaredEuclidean<f64>>();
    }

    #[test]
    fn v3_dot_product_f64_works() {
        type M = DotProduct<f64>;
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 2.0, -1.0];

        let aw = a.map(M::widen_coord);
        let bw = b.map(M::widen_coord);
        let d = <M as DistanceMetricCore<f64>>::dist::<3>(&aw, &bw);
        assert_eq!(d, 5.0);
    }
}
