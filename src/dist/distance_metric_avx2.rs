#![allow(clippy::missing_safety_doc)]

use core::arch::x86_64::{__m128, __m128d, __m256, __m256d};

/// AVX2 f64 leaf-kernel operations contract.
pub trait Avx2F64LeafOps {
    /// Calculate distance on 4 f64 lanes for the first dimension.
    unsafe fn dist_k0_f64x4(delta: __m256d) -> __m256d;

    /// Accumulate distance on 4 f64 lanes for subsequent dimensions.
    unsafe fn dist_kn_f64x4(acc: __m256d, delta: __m256d) -> __m256d;

    /// Calculate distance on 2 f64 lanes for the first dimension.
    unsafe fn dist_k0_f64x2(delta: __m128d) -> __m128d;

    /// Accumulate distance on 2 f64 lanes for subsequent dimensions.
    unsafe fn dist_kn_f64x2(acc: __m128d, delta: __m128d) -> __m128d;

    /// Calculate scalar f64 distance for the first dimension.
    fn dist_k0_f64x1(delta: f64) -> f64;

    /// Accumulate scalar f64 distance for subsequent dimensions.
    fn dist_kn_f64x1(acc: f64, delta: f64) -> f64;
}

/// Placeholder implementation for metrics without AVX2 f64 specializations.
pub struct UnsupportedAvx2F64LeafOps;

impl Avx2F64LeafOps for UnsupportedAvx2F64LeafOps {
    #[inline(always)]
    unsafe fn dist_k0_f64x4(_delta: __m256d) -> __m256d {
        panic!("AVX2 f64 leaf ops are not implemented for this metric")
    }

    #[inline(always)]
    unsafe fn dist_kn_f64x4(_acc: __m256d, _delta: __m256d) -> __m256d {
        panic!("AVX2 f64 leaf ops are not implemented for this metric")
    }

    #[inline(always)]
    unsafe fn dist_k0_f64x2(_delta: __m128d) -> __m128d {
        panic!("AVX2 f64 leaf ops are not implemented for this metric")
    }

    #[inline(always)]
    unsafe fn dist_kn_f64x2(_acc: __m128d, _delta: __m128d) -> __m128d {
        panic!("AVX2 f64 leaf ops are not implemented for this metric")
    }

    #[inline(always)]
    fn dist_k0_f64x1(_delta: f64) -> f64 {
        panic!("AVX2 f64 leaf ops are not implemented for this metric")
    }

    #[inline(always)]
    fn dist_kn_f64x1(_acc: f64, _delta: f64) -> f64 {
        panic!("AVX2 f64 leaf ops are not implemented for this metric")
    }
}

/// AVX2 f32 leaf-kernel operations contract.
pub trait Avx2F32LeafOps {
    /// Calculate distance on 8 f32 lanes for the first dimension.
    unsafe fn dist_k0_f32x8(delta: __m256) -> __m256;

    /// Accumulate distance on 8 f32 lanes for subsequent dimensions.
    unsafe fn dist_kn_f32x8(acc: __m256, delta: __m256) -> __m256;

    /// Calculate distance on 4 f32 lanes for the first dimension.
    unsafe fn dist_k0_f32x4(delta: __m128) -> __m128;

    /// Accumulate distance on 4 f32 lanes for subsequent dimensions.
    unsafe fn dist_kn_f32x4(acc: __m128, delta: __m128) -> __m128;

    /// Calculate scalar f32 distance for the first dimension.
    fn dist_k0_f32x1(delta: f32) -> f32;

    /// Accumulate scalar f32 distance for subsequent dimensions.
    fn dist_kn_f32x1(acc: f32, delta: f32) -> f32;
}

/// Placeholder implementation for metrics without AVX2 f32 specializations.
pub struct UnsupportedAvx2F32LeafOps;

impl Avx2F32LeafOps for UnsupportedAvx2F32LeafOps {
    #[inline(always)]
    unsafe fn dist_k0_f32x8(_delta: __m256) -> __m256 {
        panic!("AVX2 f32 leaf ops are not implemented for this metric")
    }

    #[inline(always)]
    unsafe fn dist_kn_f32x8(_acc: __m256, _delta: __m256) -> __m256 {
        panic!("AVX2 f32 leaf ops are not implemented for this metric")
    }

    #[inline(always)]
    unsafe fn dist_k0_f32x4(_delta: __m128) -> __m128 {
        panic!("AVX2 f32 leaf ops are not implemented for this metric")
    }

    #[inline(always)]
    unsafe fn dist_kn_f32x4(_acc: __m128, _delta: __m128) -> __m128 {
        panic!("AVX2 f32 leaf ops are not implemented for this metric")
    }

    #[inline(always)]
    fn dist_k0_f32x1(_delta: f32) -> f32 {
        panic!("AVX2 f32 leaf ops are not implemented for this metric")
    }

    #[inline(always)]
    fn dist_kn_f32x1(_acc: f32, _delta: f32) -> f32 {
        panic!("AVX2 f32 leaf ops are not implemented for this metric")
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Avx2F32LeafOps, Avx2F64LeafOps, UnsupportedAvx2F32LeafOps, UnsupportedAvx2F64LeafOps,
    };
    use core::arch::x86_64::{_mm256_set1_pd, _mm256_set1_ps, _mm_set1_pd, _mm_set1_ps};

    #[test]
    fn unsupported_avx2_f64_leaf_ops_panic() {
        assert!(std::panic::catch_unwind(|| unsafe {
            let _ = UnsupportedAvx2F64LeafOps::dist_k0_f64x4(_mm256_set1_pd(1.0));
        })
        .is_err());
        assert!(std::panic::catch_unwind(|| unsafe {
            let _ =
                UnsupportedAvx2F64LeafOps::dist_kn_f64x4(_mm256_set1_pd(1.0), _mm256_set1_pd(2.0));
        })
        .is_err());
        assert!(std::panic::catch_unwind(|| unsafe {
            let _ = UnsupportedAvx2F64LeafOps::dist_k0_f64x2(_mm_set1_pd(1.0));
        })
        .is_err());
        assert!(std::panic::catch_unwind(|| unsafe {
            let _ = UnsupportedAvx2F64LeafOps::dist_kn_f64x2(_mm_set1_pd(1.0), _mm_set1_pd(2.0));
        })
        .is_err());
        assert!(
            std::panic::catch_unwind(|| UnsupportedAvx2F64LeafOps::dist_k0_f64x1(1.0)).is_err()
        );
        assert!(
            std::panic::catch_unwind(|| UnsupportedAvx2F64LeafOps::dist_kn_f64x1(1.0, 2.0))
                .is_err()
        );
    }

    #[test]
    fn unsupported_avx2_f32_leaf_ops_panic() {
        assert!(std::panic::catch_unwind(|| unsafe {
            let _ = UnsupportedAvx2F32LeafOps::dist_k0_f32x8(_mm256_set1_ps(1.0));
        })
        .is_err());
        assert!(std::panic::catch_unwind(|| unsafe {
            let _ =
                UnsupportedAvx2F32LeafOps::dist_kn_f32x8(_mm256_set1_ps(1.0), _mm256_set1_ps(2.0));
        })
        .is_err());
        assert!(std::panic::catch_unwind(|| unsafe {
            let _ = UnsupportedAvx2F32LeafOps::dist_k0_f32x4(_mm_set1_ps(1.0));
        })
        .is_err());
        assert!(std::panic::catch_unwind(|| unsafe {
            let _ = UnsupportedAvx2F32LeafOps::dist_kn_f32x4(_mm_set1_ps(1.0), _mm_set1_ps(2.0));
        })
        .is_err());
        assert!(
            std::panic::catch_unwind(|| UnsupportedAvx2F32LeafOps::dist_k0_f32x1(1.0)).is_err()
        );
        assert!(
            std::panic::catch_unwind(|| UnsupportedAvx2F32LeafOps::dist_kn_f32x1(1.0, 2.0))
                .is_err()
        );
    }
}
