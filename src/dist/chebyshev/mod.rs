use std::cmp::Ordering;

use fixed::traits::LossyFrom;

use crate::Axis;

use crate::dist::distance_metric_core::DistanceMetricCore;
use crate::dist::{DistanceMetricAvx2, DistanceMetricAvx512, DistanceMetricNeon};

#[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
mod avx2;

#[cfg(all(feature = "simd", target_feature = "avx512f"))]
mod avx512;

#[cfg(all(feature = "simd", target_arch = "aarch64", target_feature = "neon"))]
mod neon;

/// Chebyshev / L-infinity distance metric, parameterized by output type `R`.
///
/// Defined as the maximum distance along any of the dimensions. AKA "chessboard distance".
///
/// Ref: <https://en.wikipedia.org/wiki/Chebyshev_distance>
#[doc(alias = "chessboard")]
#[doc(alias = "warehouse")]
#[doc(alias = "L∞")]
pub struct Chebyshev<R>(core::marker::PhantomData<R>);

impl<A, R> DistanceMetricCore<A> for Chebyshev<R>
where
    A: Copy,
    R: Axis<Coord = R> + LossyFrom<A>,
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

    #[inline(always)]
    fn combine_component(acc: &mut Self::Output, component: Self::Output) {
        if component > *acc {
            *acc = component;
        }
    }

    #[inline(always)]
    fn rect_dist_after_update<const K: usize>(
        _rd: Self::Output,
        off: &[Self::Output; K],
        dim: usize,
        new_off: Self::Output,
    ) -> Self::Output {
        let mut acc = Self::Output::zero();

        for axis in 0..K {
            let off_val = if axis == dim { new_off } else { off[axis] };
            Self::combine_component(&mut acc, Self::dist1(off_val, Self::Output::zero()));
        }

        acc
    }
}

impl<A, R> DistanceMetricAvx512<A> for Chebyshev<R>
where
    A: Copy,
    R: Axis<Coord = R> + LossyFrom<A>,
{
    #[cfg(all(feature = "simd", target_feature = "avx512f"))]
    type Avx512F64Ops = avx512::ChebyshevAvx512F64LeafOps;

    #[cfg(all(feature = "simd", target_feature = "avx512f"))]
    type Avx512F32Ops = avx512::ChebyshevAvx512F32LeafOps;
}

impl<A, R> DistanceMetricAvx2<A> for Chebyshev<R>
where
    A: Copy,
    R: Axis<Coord = R> + LossyFrom<A>,
{
    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
    type Avx2F64Ops = avx2::ChebyshevAvx2F64LeafOps;

    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
    type Avx2F32Ops = avx2::ChebyshevAvx2F32LeafOps;
}

impl<A, R> DistanceMetricNeon<A> for Chebyshev<R>
where
    A: Copy,
    R: Axis<Coord = R> + LossyFrom<A>,
{
    #[cfg(all(feature = "simd", target_arch = "aarch64", target_feature = "neon"))]
    type NeonF64Ops = neon::ChebyshevNeonF64LeafOps;

    #[cfg(all(feature = "simd", target_arch = "aarch64", target_feature = "neon"))]
    type NeonF32Ops = neon::ChebyshevNeonF32LeafOps;
}
