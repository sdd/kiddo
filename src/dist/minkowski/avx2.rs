use std::arch::x86_64::*;

use crate::dist::common::avx2::*;
use crate::dist::distance_metric_avx2::{Avx2F32LeafOps, Avx2F64LeafOps};

#[inline(always)]
unsafe fn pow_pd_256<const P: u32>(x: __m256d) -> __m256d {
    if P == 0 {
        return _mm256_set1_pd(1.0);
    }
    if P == 1 {
        return x;
    }
    if P == 2 {
        return _mm256_mul_pd(x, x);
    }
    if P == 3 {
        let x2 = _mm256_mul_pd(x, x);
        return _mm256_mul_pd(x2, x);
    }
    if P == 4 {
        let x2 = _mm256_mul_pd(x, x);
        return _mm256_mul_pd(x2, x2);
    }

    let mut acc = _mm256_set1_pd(1.0);
    let mut base = x;
    let mut exp = P;
    while exp != 0 {
        if exp & 1 == 1 {
            acc = _mm256_mul_pd(acc, base);
        }
        exp >>= 1;
        if exp != 0 {
            base = _mm256_mul_pd(base, base);
        }
    }
    acc
}

#[inline(always)]
unsafe fn pow_pd_128<const P: u32>(x: __m128d) -> __m128d {
    if P == 0 {
        return _mm_set1_pd(1.0);
    }
    if P == 1 {
        return x;
    }
    if P == 2 {
        return _mm_mul_pd(x, x);
    }
    if P == 3 {
        let x2 = _mm_mul_pd(x, x);
        return _mm_mul_pd(x2, x);
    }
    if P == 4 {
        let x2 = _mm_mul_pd(x, x);
        return _mm_mul_pd(x2, x2);
    }

    let mut acc = _mm_set1_pd(1.0);
    let mut base = x;
    let mut exp = P;
    while exp != 0 {
        if exp & 1 == 1 {
            acc = _mm_mul_pd(acc, base);
        }
        exp >>= 1;
        if exp != 0 {
            base = _mm_mul_pd(base, base);
        }
    }
    acc
}

#[inline(always)]
unsafe fn pow_ps_256<const P: u32>(x: __m256) -> __m256 {
    if P == 0 {
        return _mm256_set1_ps(1.0);
    }
    if P == 1 {
        return x;
    }
    if P == 2 {
        return _mm256_mul_ps(x, x);
    }
    if P == 3 {
        let x2 = _mm256_mul_ps(x, x);
        return _mm256_mul_ps(x2, x);
    }
    if P == 4 {
        let x2 = _mm256_mul_ps(x, x);
        return _mm256_mul_ps(x2, x2);
    }

    let mut acc = _mm256_set1_ps(1.0);
    let mut base = x;
    let mut exp = P;
    while exp != 0 {
        if exp & 1 == 1 {
            acc = _mm256_mul_ps(acc, base);
        }
        exp >>= 1;
        if exp != 0 {
            base = _mm256_mul_ps(base, base);
        }
    }
    acc
}

#[inline(always)]
unsafe fn pow_ps_128<const P: u32>(x: __m128) -> __m128 {
    if P == 0 {
        return _mm_set1_ps(1.0);
    }
    if P == 1 {
        return x;
    }
    if P == 2 {
        return _mm_mul_ps(x, x);
    }
    if P == 3 {
        let x2 = _mm_mul_ps(x, x);
        return _mm_mul_ps(x2, x);
    }
    if P == 4 {
        let x2 = _mm_mul_ps(x, x);
        return _mm_mul_ps(x2, x2);
    }

    let mut acc = _mm_set1_ps(1.0);
    let mut base = x;
    let mut exp = P;
    while exp != 0 {
        if exp & 1 == 1 {
            acc = _mm_mul_ps(acc, base);
        }
        exp >>= 1;
        if exp != 0 {
            base = _mm_mul_ps(base, base);
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

pub struct MinkowskiAvx2F64LeafOps<const P: u32>;

impl<const P: u32> Avx2F64LeafOps for MinkowskiAvx2F64LeafOps<P> {
    #[inline(always)]
    unsafe fn dist_k0_f64x4(delta: __m256d) -> __m256d {
        pow_pd_256::<P>(custom_mm256_abs_pd(delta))
    }

    #[inline(always)]
    unsafe fn dist_kn_f64x4(acc: __m256d, delta: __m256d) -> __m256d {
        _mm256_add_pd(acc, pow_pd_256::<P>(custom_mm256_abs_pd(delta)))
    }

    #[inline(always)]
    unsafe fn dist_k0_f64x2(delta: __m128d) -> __m128d {
        pow_pd_128::<P>(custom_mm_abs_pd(delta))
    }

    #[inline(always)]
    unsafe fn dist_kn_f64x2(acc: __m128d, delta: __m128d) -> __m128d {
        _mm_add_pd(acc, pow_pd_128::<P>(custom_mm_abs_pd(delta)))
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

pub struct MinkowskiAvx2F32LeafOps<const P: u32>;

impl<const P: u32> Avx2F32LeafOps for MinkowskiAvx2F32LeafOps<P> {
    #[inline(always)]
    unsafe fn dist_k0_f32x8(delta: __m256) -> __m256 {
        pow_ps_256::<P>(custom_mm256_abs_ps(delta))
    }

    #[inline(always)]
    unsafe fn dist_kn_f32x8(acc: __m256, delta: __m256) -> __m256 {
        _mm256_add_ps(acc, pow_ps_256::<P>(custom_mm256_abs_ps(delta)))
    }

    #[inline(always)]
    unsafe fn dist_k0_f32x4(delta: __m128) -> __m128 {
        pow_ps_128::<P>(custom_mm_abs_ps(delta))
    }

    #[inline(always)]
    unsafe fn dist_kn_f32x4(acc: __m128, delta: __m128) -> __m128 {
        _mm_add_ps(acc, pow_ps_128::<P>(custom_mm_abs_ps(delta)))
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
