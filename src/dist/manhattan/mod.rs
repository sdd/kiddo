use core::cmp::Ordering;
use core::ops::Add;

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

/// Manhattan / L1 distance metric, parameterized by output type `R`.
pub struct Manhattan<R>(core::marker::PhantomData<R>);

impl<A, R> DistanceMetricCore<A> for Manhattan<R>
where
    A: Copy,
    R: AxisUnified<Coord = R> + LossyFrom<A> + Add<Output = R>,
{
    type Output = R;
    const ORDERING: Ordering = Ordering::Less;

    #[inline(always)]
    fn widen_coord(a: A) -> R {
        R::lossy_from(a)
    }

    #[inline(always)]
    fn dist1(a: R, b: R) -> R {
        if a >= b {
            a - b
        } else {
            b - a
        }
    }
}

impl<A, R> DistanceMetricAvx512<A> for Manhattan<R>
where
    A: Copy,
    R: AxisUnified<Coord = R> + LossyFrom<A> + Add<Output = R>,
{
    #[cfg(all(feature = "simd", target_feature = "avx512f"))]
    type Avx512F64Ops = avx512::ManhattanAvx512F64LeafOps;
}

impl<A, R> DistanceMetricAvx2<A> for Manhattan<R>
where
    A: Copy,
    R: AxisUnified<Coord = R> + LossyFrom<A> + Add<Output = R>,
{
    #[cfg(all(feature = "simd", target_feature = "avx2"))]
    type Avx2LeafOps = avx2::ManhattanAvx2LeafOps;
}

impl<A, R> DistanceMetricNeon<A> for Manhattan<R>
where
    A: Copy,
    R: AxisUnified<Coord = R> + LossyFrom<A> + Add<Output = R>,
{
    #[cfg(all(feature = "simd", target_feature = "neon"))]
    type NeonLeafOps = neon::ManhattanNeonLeafOps;
}
