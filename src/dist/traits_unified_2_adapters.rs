use core::ops::{Add, Mul};

use fixed::traits::LossyFrom;

use crate::dist::{DotProduct, Manhattan, SquaredEuclidean};
use crate::stem_strategies::donnelly_2_blockmarker_simd::{
    DistanceMetricSimdBlock3, DistanceMetricSimdBlock4,
};
use crate::traits_unified_2::{
    AxisUnified, DistanceMetricUnified as DistanceMetricUnifiedV2, DotProduct as DotProductV2,
    Manhattan as ManhattanV2, SquaredEuclidean as SquaredEuclideanV2,
};

impl<A, R, const K: usize> DistanceMetricUnifiedV2<A, K> for SquaredEuclidean<R>
where
    A: Copy,
    R: AxisUnified<Coord = R> + LossyFrom<A> + Mul<Output = R> + Add<Output = R>,
{
    type Output = R;
    const ORDERING: core::cmp::Ordering = core::cmp::Ordering::Less;

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

impl<A, R, const K: usize> DistanceMetricUnifiedV2<A, K> for Manhattan<R>
where
    A: Copy,
    R: AxisUnified<Coord = R> + LossyFrom<A> + Mul<Output = R> + Add<Output = R>,
{
    type Output = R;
    const ORDERING: core::cmp::Ordering = core::cmp::Ordering::Less;

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

impl<A, R, const K: usize> DistanceMetricUnifiedV2<A, K> for DotProduct<R>
where
    A: Copy,
    R: AxisUnified<Coord = R> + LossyFrom<A> + Mul<Output = R> + Add<Output = R>,
{
    type Output = R;
    const ORDERING: core::cmp::Ordering = core::cmp::Ordering::Greater;

    #[inline(always)]
    fn widen_coord(a: A) -> R {
        R::lossy_from(a)
    }

    #[inline(always)]
    fn dist1(a: R, b: R) -> R {
        a * b
    }
}

macro_rules! impl_simd_backtrack_adapters {
    ($metric_v3:ty, $metric_v2:ty, $o:ty) => {
        impl<A: Copy, const K: usize> DistanceMetricSimdBlock3<A, K, $o> for $metric_v3
        where
            $metric_v2: DistanceMetricSimdBlock3<A, K, $o>,
        {
            #[inline(always)]
            fn backtrack_block3_autovec(
                query_wide: $o,
                stems_ptr: std::ptr::NonNull<u8>,
                block_base_idx: usize,
                old_off: $o,
                rd: $o,
                best_dist: $o,
            ) -> u8 {
                <$metric_v2 as DistanceMetricSimdBlock3<A, K, $o>>::backtrack_block3_autovec(
                    query_wide,
                    stems_ptr,
                    block_base_idx,
                    old_off,
                    rd,
                    best_dist,
                )
            }

            #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
            #[inline(always)]
            unsafe fn backtrack_block3_avx2(
                query_wide: $o,
                stems_ptr: std::ptr::NonNull<u8>,
                block_base_idx: usize,
                old_off: $o,
                rd: $o,
                best_dist: $o,
            ) -> u8 {
                <$metric_v2 as DistanceMetricSimdBlock3<A, K, $o>>::backtrack_block3_avx2(
                    query_wide,
                    stems_ptr,
                    block_base_idx,
                    old_off,
                    rd,
                    best_dist,
                )
            }

            #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
            #[inline(always)]
            unsafe fn backtrack_block3_avx512(
                query_wide: $o,
                stems_ptr: std::ptr::NonNull<u8>,
                block_base_idx: usize,
                old_off: $o,
                rd: $o,
                best_dist: $o,
            ) -> u8 {
                <$metric_v2 as DistanceMetricSimdBlock3<A, K, $o>>::backtrack_block3_avx512(
                    query_wide,
                    stems_ptr,
                    block_base_idx,
                    old_off,
                    rd,
                    best_dist,
                )
            }

            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            #[inline(always)]
            unsafe fn backtrack_block3_neon(
                query_wide: $o,
                stems_ptr: std::ptr::NonNull<u8>,
                block_base_idx: usize,
                old_off: $o,
                rd: $o,
                best_dist: $o,
            ) -> u8 {
                <$metric_v2 as DistanceMetricSimdBlock3<A, K, $o>>::backtrack_block3_neon(
                    query_wide,
                    stems_ptr,
                    block_base_idx,
                    old_off,
                    rd,
                    best_dist,
                )
            }
        }

        impl<A: Copy, const K: usize> DistanceMetricSimdBlock4<A, K, $o> for $metric_v3
        where
            $metric_v2: DistanceMetricSimdBlock4<A, K, $o>,
        {
            #[inline(always)]
            fn backtrack_block4_autovec(
                query_wide: $o,
                stems_ptr: std::ptr::NonNull<u8>,
                block_base_idx: usize,
                old_off: $o,
                rd: $o,
                best_dist: $o,
            ) -> u16 {
                <$metric_v2 as DistanceMetricSimdBlock4<A, K, $o>>::backtrack_block4_autovec(
                    query_wide,
                    stems_ptr,
                    block_base_idx,
                    old_off,
                    rd,
                    best_dist,
                )
            }

            #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
            #[inline(always)]
            unsafe fn backtrack_block4_avx2(
                query_wide: $o,
                stems_ptr: std::ptr::NonNull<u8>,
                block_base_idx: usize,
                old_off: $o,
                rd: $o,
                best_dist: $o,
            ) -> u16 {
                <$metric_v2 as DistanceMetricSimdBlock4<A, K, $o>>::backtrack_block4_avx2(
                    query_wide,
                    stems_ptr,
                    block_base_idx,
                    old_off,
                    rd,
                    best_dist,
                )
            }

            #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
            #[inline(always)]
            unsafe fn backtrack_block4_avx512(
                query_wide: $o,
                stems_ptr: std::ptr::NonNull<u8>,
                block_base_idx: usize,
                old_off: $o,
                rd: $o,
                best_dist: $o,
            ) -> u16 {
                <$metric_v2 as DistanceMetricSimdBlock4<A, K, $o>>::backtrack_block4_avx512(
                    query_wide,
                    stems_ptr,
                    block_base_idx,
                    old_off,
                    rd,
                    best_dist,
                )
            }

            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            #[inline(always)]
            unsafe fn backtrack_block4_neon(
                query_wide: $o,
                stems_ptr: std::ptr::NonNull<u8>,
                block_base_idx: usize,
                old_off: $o,
                rd: $o,
                best_dist: $o,
            ) -> u16 {
                <$metric_v2 as DistanceMetricSimdBlock4<A, K, $o>>::backtrack_block4_neon(
                    query_wide,
                    stems_ptr,
                    block_base_idx,
                    old_off,
                    rd,
                    best_dist,
                )
            }
        }
    };
}

impl_simd_backtrack_adapters!(SquaredEuclidean<f32>, SquaredEuclideanV2<f32>, f32);
impl_simd_backtrack_adapters!(SquaredEuclidean<f64>, SquaredEuclideanV2<f64>, f64);
impl_simd_backtrack_adapters!(Manhattan<f32>, ManhattanV2<f32>, f32);
impl_simd_backtrack_adapters!(Manhattan<f64>, ManhattanV2<f64>, f64);
impl_simd_backtrack_adapters!(DotProduct<f32>, DotProductV2<f32>, f32);
impl_simd_backtrack_adapters!(DotProduct<f64>, DotProductV2<f64>, f64);
