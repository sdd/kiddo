#[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
pub(crate) mod avx2 {
    use std::arch::x86_64::*;

    #[inline(always)]
    pub(crate) unsafe fn custom_mm256_abs_pd(x: __m256d) -> __m256d {
        let sign = _mm256_set1_pd(-0.0);
        _mm256_andnot_pd(sign, x)
    }

    #[inline(always)]
    pub(crate) unsafe fn custom_mm_abs_pd(x: __m128d) -> __m128d {
        let sign = _mm_set1_pd(-0.0);
        _mm_andnot_pd(sign, x)
    }

    #[inline(always)]
    pub(crate) unsafe fn custom_mm256_abs_ps(x: __m256) -> __m256 {
        let sign = _mm256_set1_ps(-0.0);
        _mm256_andnot_ps(sign, x)
    }

    #[inline(always)]
    pub(crate) unsafe fn custom_mm_abs_ps(x: __m128) -> __m128 {
        let sign = _mm_set1_ps(-0.0);
        _mm_andnot_ps(sign, x)
    }
}

#[cfg(all(feature = "simd", target_feature = "avx512f"))]
pub(crate) mod avx512 {
    use std::arch::x86_64::*;

    #[inline(always)]
    pub(crate) unsafe fn custom_mm512_abs_pd(x: __m512d) -> __m512d {
        let sign = _mm512_set1_pd(-0.0);
        _mm512_andnot_pd(sign, x)
    }

    #[inline(always)]
    pub(crate) unsafe fn custom_mm512_abs_ps(x: __m512) -> __m512 {
        let sign = _mm512_set1_ps(-0.0);
        _mm512_andnot_ps(sign, x)
    }
}
