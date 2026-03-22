pub struct DotProductAvx512F64LeafOps;

impl crate::dist::distance_metric_avx512::Avx512F64LeafOps for DotProductAvx512F64LeafOps {
    #[inline(always)]
    unsafe fn dist_k0_f64x8(delta: core::arch::x86_64::__m512d) -> core::arch::x86_64::__m512d {
        delta
    }

    #[inline(always)]
    unsafe fn dist_kn_f64x8(
        acc: core::arch::x86_64::__m512d,
        delta: core::arch::x86_64::__m512d,
    ) -> core::arch::x86_64::__m512d {
        core::arch::x86_64::_mm512_add_pd(acc, delta)
    }

    #[inline(always)]
    unsafe fn dist_k0_f64x4(delta: core::arch::x86_64::__m256d) -> core::arch::x86_64::__m256d {
        delta
    }

    #[inline(always)]
    unsafe fn dist_kn_f64x4(
        acc: core::arch::x86_64::__m256d,
        delta: core::arch::x86_64::__m256d,
    ) -> core::arch::x86_64::__m256d {
        core::arch::x86_64::_mm256_add_pd(acc, delta)
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
