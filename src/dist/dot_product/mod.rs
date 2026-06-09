use std::{
    cmp::Ordering,
    ops::{Add, Mul},
};

use fixed::traits::LossyFrom;

use crate::Axis;

use crate::dist::distance_metric_core::DistanceMetricCore;
use crate::dist::{DistanceMetricAvx2, DistanceMetricAvx512, DistanceMetricNeon};

/// Dot product similarity metric, parameterized by output type `R`.
///
/// Only really useful when the points in the tree are all on the surface of a K-dimensional unit
/// hypersphere. When this is the case though, this metric can be faster to compute than others.
/// Returns the cosine of the angle between the two points.
///
/// Since the cosine increases towards 1 with decreased angle between the points being compared,
/// unlike the other metrics, the sorted results from a query using it are sorted greatest-first
/// rather than least-first.
///
/// Ref: <https://en.wikipedia.org/wiki/Cosine_similarity>
pub struct DotProduct<R>(core::marker::PhantomData<R>);

impl<A, R> DistanceMetricCore<A> for DotProduct<R>
where
    A: Copy,
    R: Axis<Coord = R> + LossyFrom<A> + Mul<Output = R> + Add<Output = R>,
{
    type Output = R;
    const ORDERING: Ordering = Ordering::Greater;

    #[inline(always)]
    fn widen_coord(a: A) -> R {
        R::lossy_from(a)
    }

    #[inline(always)]
    fn dist1(a: R, b: R) -> R {
        a * b
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
mod avx2;

#[cfg(all(feature = "simd", target_feature = "avx512f"))]
mod avx512;

#[cfg(all(feature = "simd", target_arch = "aarch64", target_feature = "neon"))]
mod neon;

impl<A, R> DistanceMetricAvx512<A> for DotProduct<R>
where
    A: Copy,
    R: Axis<Coord = R> + LossyFrom<A> + Mul<Output = R> + Add<Output = R>,
{
    #[cfg(all(feature = "simd", target_feature = "avx512f"))]
    type Avx512F64Ops = avx512::DotProductAvx512F64LeafOps;

    #[cfg(all(feature = "simd", target_feature = "avx512f"))]
    type Avx512F32Ops = crate::dist::distance_metric_avx512::UnsupportedAvx512F32LeafOps;
}

impl<A, R> DistanceMetricAvx2<A> for DotProduct<R>
where
    A: Copy,
    R: Axis<Coord = R> + LossyFrom<A> + Mul<Output = R> + Add<Output = R>,
{
    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
    type Avx2F64Ops = crate::dist::distance_metric_avx2::UnsupportedAvx2F64LeafOps;

    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
    type Avx2F32Ops = crate::dist::distance_metric_avx2::UnsupportedAvx2F32LeafOps;
}

impl<A, R> DistanceMetricNeon<A> for DotProduct<R>
where
    A: Copy,
    R: Axis<Coord = R> + LossyFrom<A> + Mul<Output = R> + Add<Output = R>,
{
    #[cfg(all(feature = "simd", target_arch = "aarch64", target_feature = "neon"))]
    type NeonF64Ops = crate::dist::distance_metric_neon::UnsupportedNeonF64LeafOps;

    #[cfg(all(feature = "simd", target_arch = "aarch64", target_feature = "neon"))]
    type NeonF32Ops = crate::dist::distance_metric_neon::UnsupportedNeonF32LeafOps;
}
