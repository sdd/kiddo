use std::arch::aarch64::*;

use crate::dist::distance_metric_neon::{NeonF32LeafOps, NeonF64LeafOps};

#[inline(always)]
unsafe fn pow_f64x2<const P: u32>(x: float64x2_t) -> float64x2_t {
    if P == 0 {
        return vdupq_n_f64(1.0);
    }
    if P == 1 {
        return x;
    }
    if P == 2 {
        return vmulq_f64(x, x);
    }
    if P == 3 {
        let x2 = vmulq_f64(x, x);
        return vmulq_f64(x2, x);
    }
    if P == 4 {
        let x2 = vmulq_f64(x, x);
        return vmulq_f64(x2, x2);
    }

    let mut acc = vdupq_n_f64(1.0);
    let mut base = x;
    let mut exp = P;
    while exp != 0 {
        if exp & 1 == 1 {
            acc = vmulq_f64(acc, base);
        }
        exp >>= 1;
        if exp != 0 {
            base = vmulq_f64(base, base);
        }
    }
    acc
}

#[inline(always)]
unsafe fn pow_f32x4<const P: u32>(x: float32x4_t) -> float32x4_t {
    if P == 0 {
        return vdupq_n_f32(1.0);
    }
    if P == 1 {
        return x;
    }
    if P == 2 {
        return vmulq_f32(x, x);
    }
    if P == 3 {
        let x2 = vmulq_f32(x, x);
        return vmulq_f32(x2, x);
    }
    if P == 4 {
        let x2 = vmulq_f32(x, x);
        return vmulq_f32(x2, x2);
    }

    let mut acc = vdupq_n_f32(1.0);
    let mut base = x;
    let mut exp = P;
    while exp != 0 {
        if exp & 1 == 1 {
            acc = vmulq_f32(acc, base);
        }
        exp >>= 1;
        if exp != 0 {
            base = vmulq_f32(base, base);
        }
    }
    acc
}

#[inline(always)]
fn pow_f64<const P: u32>(x: f64) -> f64 {
    if P == 0 {
        return 1.0;
    }
    if P == 1 {
        return x;
    }
    if P == 2 {
        return x * x;
    }
    if P == 3 {
        return x * x * x;
    }
    if P == 4 {
        let x2 = x * x;
        return x2 * x2;
    }
    x.powi(P as i32)
}

#[inline(always)]
fn pow_f32<const P: u32>(x: f32) -> f32 {
    if P == 0 {
        return 1.0;
    }
    if P == 1 {
        return x;
    }
    if P == 2 {
        return x * x;
    }
    if P == 3 {
        return x * x * x;
    }
    if P == 4 {
        let x2 = x * x;
        return x2 * x2;
    }
    x.powi(P as i32)
}

pub struct MinkowskiNeonF64LeafOps<const P: u32>;

impl<const P: u32> NeonF64LeafOps for MinkowskiNeonF64LeafOps<P> {
    #[inline(always)]
    unsafe fn dist_k0_f64x2(delta: float64x2_t) -> float64x2_t {
        pow_f64x2::<P>(vabsq_f64(delta))
    }

    #[inline(always)]
    unsafe fn dist_kn_f64x2(acc: float64x2_t, delta: float64x2_t) -> float64x2_t {
        vaddq_f64(acc, pow_f64x2::<P>(vabsq_f64(delta)))
    }

    #[inline(always)]
    fn dist_k0_f64x1(delta: f64) -> f64 {
        pow_f64::<P>(delta.abs())
    }

    #[inline(always)]
    fn dist_kn_f64x1(acc: f64, delta: f64) -> f64 {
        acc + pow_f64::<P>(delta.abs())
    }
}

pub struct MinkowskiNeonF32LeafOps<const P: u32>;

impl<const P: u32> NeonF32LeafOps for MinkowskiNeonF32LeafOps<P> {
    #[inline(always)]
    unsafe fn dist_k0_f32x4(delta: float32x4_t) -> float32x4_t {
        pow_f32x4::<P>(vabsq_f32(delta))
    }

    #[inline(always)]
    unsafe fn dist_kn_f32x4(acc: float32x4_t, delta: float32x4_t) -> float32x4_t {
        vaddq_f32(acc, pow_f32x4::<P>(vabsq_f32(delta)))
    }

    #[inline(always)]
    fn dist_k0_f32x1(delta: f32) -> f32 {
        pow_f32::<P>(delta.abs())
    }

    #[inline(always)]
    fn dist_kn_f32x1(acc: f32, delta: f32) -> f32 {
        acc + pow_f32::<P>(delta.abs())
    }
}
