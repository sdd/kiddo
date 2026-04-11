use std::arch::aarch64::*;

use crate::dist::distance_metric_neon::{NeonF32LeafOps, NeonF64LeafOps};

pub struct SquaredEuclideanNeonF64LeafOps;

impl NeonF64LeafOps for SquaredEuclideanNeonF64LeafOps {
    #[inline(always)]
    unsafe fn dist_k0_f64x2(delta: float64x2_t) -> float64x2_t {
        vmulq_f64(delta, delta)
    }

    #[inline(always)]
    unsafe fn dist_kn_f64x2(acc: float64x2_t, delta: float64x2_t) -> float64x2_t {
        vaddq_f64(acc, vmulq_f64(delta, delta))
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

pub struct SquaredEuclideanNeonF32LeafOps;

impl NeonF32LeafOps for SquaredEuclideanNeonF32LeafOps {
    #[inline(always)]
    unsafe fn dist_k0_f32x4(delta: float32x4_t) -> float32x4_t {
        vmulq_f32(delta, delta)
    }

    #[inline(always)]
    unsafe fn dist_kn_f32x4(acc: float32x4_t, delta: float32x4_t) -> float32x4_t {
        vaddq_f32(acc, vmulq_f32(delta, delta))
    }

    #[inline(always)]
    fn dist_k0_f32x1(delta: f32) -> f32 {
        delta * delta
    }

    #[inline(always)]
    fn dist_kn_f32x1(acc: f32, delta: f32) -> f32 {
        acc + delta * delta
    }
}
