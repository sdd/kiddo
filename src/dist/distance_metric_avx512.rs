use core::arch::x86_64::{__m256d, __m512d};

/// AVX512 f64 leaf-kernel operations contract.
///
/// Mirrors the shape used by the hand-tuned `ultimate_*` AVX512 kernel.
pub trait Avx512F64LeafOps {
    unsafe fn dist_k0_f64x8(delta: __m512d) -> __m512d;
    unsafe fn dist_kn_f64x8(acc: __m512d, delta: __m512d) -> __m512d;
    unsafe fn dist_k0_f64x4(delta: __m256d) -> __m256d;
    unsafe fn dist_kn_f64x4(acc: __m256d, delta: __m256d) -> __m256d;
    fn dist_k0_f64x1(delta: f64) -> f64;
    fn dist_kn_f64x1(acc: f64, delta: f64) -> f64;
}

/// Placeholder implementation for metrics without AVX512 f64 specializations.
pub struct UnsupportedAvx512F64LeafOps;

impl Avx512F64LeafOps for UnsupportedAvx512F64LeafOps {
    #[inline(always)]
    unsafe fn dist_k0_f64x8(_delta: __m512d) -> __m512d {
        panic!("AVX512 f64 leaf ops are not implemented for this metric")
    }

    #[inline(always)]
    unsafe fn dist_kn_f64x8(_acc: __m512d, _delta: __m512d) -> __m512d {
        panic!("AVX512 f64 leaf ops are not implemented for this metric")
    }

    #[inline(always)]
    unsafe fn dist_k0_f64x4(_delta: __m256d) -> __m256d {
        panic!("AVX512 f64 leaf ops are not implemented for this metric")
    }

    #[inline(always)]
    unsafe fn dist_kn_f64x4(_acc: __m256d, _delta: __m256d) -> __m256d {
        panic!("AVX512 f64 leaf ops are not implemented for this metric")
    }

    #[inline(always)]
    fn dist_k0_f64x1(_delta: f64) -> f64 {
        panic!("AVX512 f64 leaf ops are not implemented for this metric")
    }

    #[inline(always)]
    fn dist_kn_f64x1(_acc: f64, _delta: f64) -> f64 {
        panic!("AVX512 f64 leaf ops are not implemented for this metric")
    }
}
