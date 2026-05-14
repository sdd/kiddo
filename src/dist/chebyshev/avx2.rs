use std::arch::x86_64::*;

use crate::dist::common::avx2::*;
use crate::dist::distance_metric_avx2::{Avx2F32LeafOps, Avx2F64LeafOps};

pub struct ChebyshevAvx2F64LeafOps;

impl Avx2F64LeafOps for ChebyshevAvx2F64LeafOps {
    #[inline(always)]
    unsafe fn dist_k0_f64x4(delta: __m256d) -> __m256d {
        custom_mm256_abs_pd(delta)
    }

    #[inline(always)]
    unsafe fn dist_kn_f64x4(acc: __m256d, delta: __m256d) -> __m256d {
        _mm256_max_pd(acc, custom_mm256_abs_pd(delta))
    }

    #[inline(always)]
    unsafe fn dist_k0_f64x2(delta: __m128d) -> __m128d {
        custom_mm_abs_pd(delta)
    }

    #[inline(always)]
    unsafe fn dist_kn_f64x2(acc: __m128d, delta: __m128d) -> __m128d {
        _mm_max_pd(acc, custom_mm_abs_pd(delta))
    }

    #[inline(always)]
    fn dist_k0_f64x1(delta: f64) -> f64 {
        delta.abs()
    }

    #[inline(always)]
    fn dist_kn_f64x1(acc: f64, delta: f64) -> f64 {
        acc.max(delta.abs())
    }
}

pub struct ChebyshevAvx2F32LeafOps;

impl Avx2F32LeafOps for ChebyshevAvx2F32LeafOps {
    #[inline(always)]
    unsafe fn dist_k0_f32x8(delta: __m256) -> __m256 {
        custom_mm256_abs_ps(delta)
    }

    #[inline(always)]
    unsafe fn dist_kn_f32x8(acc: __m256, delta: __m256) -> __m256 {
        _mm256_max_ps(acc, custom_mm256_abs_ps(delta))
    }

    #[inline(always)]
    unsafe fn dist_k0_f32x4(delta: __m128) -> __m128 {
        custom_mm_abs_ps(delta)
    }

    #[inline(always)]
    unsafe fn dist_kn_f32x4(acc: __m128, delta: __m128) -> __m128 {
        _mm_max_ps(acc, custom_mm_abs_ps(delta))
    }

    #[inline(always)]
    fn dist_k0_f32x1(delta: f32) -> f32 {
        delta.abs()
    }

    #[inline(always)]
    fn dist_kn_f32x1(acc: f32, delta: f32) -> f32 {
        acc.max(delta.abs())
    }
}
