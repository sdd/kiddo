use core::cmp::Ordering;
use core::ops::{Add, Mul};

use fixed::traits::LossyFrom;

use crate::dist::distance_metric_core::DistanceMetricCore;
use crate::dist::{DistanceMetricAvx2, DistanceMetricAvx512, DistanceMetricNeon};
use crate::traits_unified_2::AxisUnified;

#[cfg(all(feature = "simd", target_feature = "avx2"))]
mod avx2;

#[cfg(all(feature = "simd", target_feature = "avx512f"))]
mod avx512;

#[cfg(all(feature = "simd", target_feature = "neon"))]
mod neon;

/// Squared Euclidean distance metric, parameterized by output type `R`.
pub struct SquaredEuclidean<R>(core::marker::PhantomData<R>);

impl<A, R> DistanceMetricCore<A> for SquaredEuclidean<R>
where
    A: Copy,
    R: AxisUnified<Coord = R> + LossyFrom<A> + Mul<Output = R> + Add<Output = R>,
{
    type Output = R;
    const ORDERING: Ordering = Ordering::Less;

    #[inline(always)]
    fn widen_coord(a: A) -> R {
        R::lossy_from(a)
    }

    #[inline(always)]
    fn dist1(a: R, b: R) -> R {
        let d = if a >= b { a - b } else { b - a };
        d * d
    }
}

impl<A, R> DistanceMetricAvx512<A> for SquaredEuclidean<R>
where
    A: Copy,
    R: AxisUnified<Coord = R> + LossyFrom<A> + Mul<Output = R> + Add<Output = R>,
{
    #[cfg(all(feature = "simd", target_feature = "avx512f"))]
    const HAS_AVX512_SPECIALIZATION: bool = true;

    #[cfg(all(feature = "simd", target_feature = "avx512f"))]
    type Avx512F64Ops = avx512::SquaredEuclideanAvx512F64LeafOps;
}

impl<A, R> DistanceMetricAvx2<A> for SquaredEuclidean<R>
where
    A: Copy,
    R: AxisUnified<Coord = R> + LossyFrom<A> + Mul<Output = R> + Add<Output = R>,
{
    #[cfg(all(feature = "simd", target_feature = "avx2"))]
    type Avx2LeafOps = avx2::SquaredEuclideanAvx2LeafOps;
}

impl<A, R> DistanceMetricNeon<A> for SquaredEuclidean<R>
where
    A: Copy,
    R: AxisUnified<Coord = R> + LossyFrom<A> + Mul<Output = R> + Add<Output = R>,
{
    #[cfg(all(feature = "simd", target_feature = "neon"))]
    type NeonLeafOps = neon::SquaredEuclideanNeonLeafOps;
}
