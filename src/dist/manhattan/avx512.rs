use std::arch::x86_64::*;

use crate::dist::distance_metric_avx512::Avx512F64LeafOps;
#[inline(always)]
unsafe fn abs_pd_512(x: __m512d) -> __m512d {
    let sign = _mm512_set1_pd(-0.0);
    _mm512_andnot_pd(sign, x)
}

#[inline(always)]
unsafe fn abs_pd_256(x: __m256d) -> __m256d {
    let sign = _mm256_set1_pd(-0.0);
    _mm256_andnot_pd(sign, x)
}

pub struct ManhattanAvx512F64LeafOps;

impl Avx512F64LeafOps for ManhattanAvx512F64LeafOps {
    #[inline(always)]
    unsafe fn dist_k0_f64x8(delta: __m512d) -> __m512d {
        abs_pd_512(delta)
    }

    #[inline(always)]
    unsafe fn dist_kn_f64x8(acc: __m512d, delta: __m512d) -> __m512d {
        _mm512_add_pd(acc, abs_pd_512(delta))
    }

    #[inline(always)]
    unsafe fn dist_k0_f64x4(delta: __m256d) -> __m256d {
        abs_pd_256(delta)
    }

    #[inline(always)]
    unsafe fn dist_kn_f64x4(acc: __m256d, delta: __m256d) -> __m256d {
        _mm256_add_pd(acc, abs_pd_256(delta))
    }

    #[inline(always)]
    fn dist_k0_f64x1(delta: f64) -> f64 {
        delta.abs()
    }

    #[inline(always)]
    fn dist_kn_f64x1(acc: f64, delta: f64) -> f64 {
        acc + delta.abs()
    }
}
