use std::arch::x86_64::*;

use crate::dist::distance_metric_avx512::Avx512F64LeafOps;

pub struct SquaredEuclideanAvx512F64LeafOps;

impl Avx512F64LeafOps for SquaredEuclideanAvx512F64LeafOps {
    #[inline(always)]
    unsafe fn dist_k0_f64x8(delta: __m512d) -> __m512d {
        _mm512_mul_pd(delta, delta)
    }

    #[inline(always)]
    unsafe fn dist_kn_f64x8(acc: __m512d, delta: __m512d) -> __m512d {
        _mm512_fmadd_pd(delta, delta, acc)
    }

    #[inline(always)]
    unsafe fn dist_k0_f64x4(delta: __m256d) -> __m256d {
        _mm256_mul_pd(delta, delta)
    }

    #[inline(always)]
    unsafe fn dist_kn_f64x4(acc: __m256d, delta: __m256d) -> __m256d {
        _mm256_fmadd_pd(delta, delta, acc)
    }

    #[inline(always)]
    fn dist_k0_f64x1(delta: f64) -> f64 {
        delta * delta
    }

    #[inline(always)]
    fn dist_kn_f64x1(acc: f64, delta: f64) -> f64 {
        acc + delta * delta
    }
}
