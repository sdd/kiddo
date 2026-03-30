use std::arch::x86_64::*;

use crate::dist::distance_metric_avx512::Avx512F64LeafOps;

pub struct DotProductAvx512F64LeafOps;

impl Avx512F64LeafOps for DotProductAvx512F64LeafOps {
    #[inline(always)]
    unsafe fn dist_k0_f64x8(delta: __m512d) -> __m512d {
        delta
    }

    #[inline(always)]
    unsafe fn dist_kn_f64x8(acc: __m512d, delta: __m512d) -> __m512d {
        _mm512_add_pd(acc, delta)
    }

    #[inline(always)]
    unsafe fn dist_k0_f64x4(delta: __m256d) -> __m256d {
        delta
    }

    #[inline(always)]
    unsafe fn dist_kn_f64x4(acc: __m256d, delta: __m256d) -> __m256d {
        _mm256_add_pd(acc, delta)
    }

    #[inline(always)]
    fn dist_k0_f64x1(delta: f64) -> f64 {
        delta
    }

    #[inline(always)]
    fn dist_kn_f64x1(acc: f64, delta: f64) -> f64 {
        acc + delta
    }
}
