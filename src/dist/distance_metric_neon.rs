use core::arch::aarch64::{float32x4_t, float64x2_t};

/// NEON f64 leaf-kernel operations contract.
pub trait NeonF64LeafOps {
    /// Calculate distance on 2 f64 lanes for the first dimension.
    unsafe fn dist_k0_f64x2(delta: float64x2_t) -> float64x2_t;

    /// Accumulate distance on 2 f64 lanes for subsequent dimensions.
    unsafe fn dist_kn_f64x2(acc: float64x2_t, delta: float64x2_t) -> float64x2_t;

    /// Calculate scalar f64 distance for the first dimension.
    fn dist_k0_f64x1(delta: f64) -> f64;

    /// Accumulate scalar f64 distance for subsequent dimensions.
    fn dist_kn_f64x1(acc: f64, delta: f64) -> f64;
}

/// Placeholder implementation for metrics without NEON f64 specializations.
pub struct UnsupportedNeonF64LeafOps;

impl NeonF64LeafOps for UnsupportedNeonF64LeafOps {
    #[inline(always)]
    unsafe fn dist_k0_f64x2(_delta: float64x2_t) -> float64x2_t {
        panic!("NEON f64 leaf ops are not implemented for this metric")
    }

    #[inline(always)]
    unsafe fn dist_kn_f64x2(_acc: float64x2_t, _delta: float64x2_t) -> float64x2_t {
        panic!("NEON f64 leaf ops are not implemented for this metric")
    }

    #[inline(always)]
    fn dist_k0_f64x1(_delta: f64) -> f64 {
        panic!("NEON f64 leaf ops are not implemented for this metric")
    }

    #[inline(always)]
    fn dist_kn_f64x1(_acc: f64, _delta: f64) -> f64 {
        panic!("NEON f64 leaf ops are not implemented for this metric")
    }
}

/// NEON f32 leaf-kernel operations contract.
pub trait NeonF32LeafOps {
    /// Calculate distance on 4 f32 lanes for the first dimension.
    unsafe fn dist_k0_f32x4(delta: float32x4_t) -> float32x4_t;

    /// Accumulate distance on 4 f32 lanes for subsequent dimensions.
    unsafe fn dist_kn_f32x4(acc: float32x4_t, delta: float32x4_t) -> float32x4_t;

    /// Calculate scalar f32 distance for the first dimension.
    fn dist_k0_f32x1(delta: f32) -> f32;

    /// Accumulate scalar f32 distance for subsequent dimensions.
    fn dist_kn_f32x1(acc: f32, delta: f32) -> f32;
}

/// Placeholder implementation for metrics without NEON f32 specializations.
pub struct UnsupportedNeonF32LeafOps;

impl NeonF32LeafOps for UnsupportedNeonF32LeafOps {
    #[inline(always)]
    unsafe fn dist_k0_f32x4(_delta: float32x4_t) -> float32x4_t {
        panic!("NEON f32 leaf ops are not implemented for this metric")
    }

    #[inline(always)]
    unsafe fn dist_kn_f32x4(_acc: float32x4_t, _delta: float32x4_t) -> float32x4_t {
        panic!("NEON f32 leaf ops are not implemented for this metric")
    }

    #[inline(always)]
    fn dist_k0_f32x1(_delta: f32) -> f32 {
        panic!("NEON f32 leaf ops are not implemented for this metric")
    }

    #[inline(always)]
    fn dist_kn_f32x1(_acc: f32, _delta: f32) -> f32 {
        panic!("NEON f32 leaf ops are not implemented for this metric")
    }
}
