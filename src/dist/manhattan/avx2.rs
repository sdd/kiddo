use std::arch::x86_64::*;

use crate::dist::distance_metric_avx2::{Avx2F32LeafOps, Avx2F64LeafOps};

#[inline(always)]
unsafe fn abs_pd_256(x: __m256d) -> __m256d {
    let sign = _mm256_set1_pd(-0.0);
    _mm256_andnot_pd(sign, x)
}

#[inline(always)]
unsafe fn abs_pd_128(x: __m128d) -> __m128d {
    let sign = _mm_set1_pd(-0.0);
    _mm_andnot_pd(sign, x)
}

#[inline(always)]
unsafe fn abs_ps_256(x: __m256) -> __m256 {
    let sign = _mm256_set1_ps(-0.0);
    _mm256_andnot_ps(sign, x)
}

#[inline(always)]
unsafe fn abs_ps_128(x: __m128) -> __m128 {
    let sign = _mm_set1_ps(-0.0);
    _mm_andnot_ps(sign, x)
}

pub struct ManhattanAvx2F64LeafOps;

impl Avx2F64LeafOps for ManhattanAvx2F64LeafOps {
    #[inline(always)]
    unsafe fn dist_k0_f64x4(delta: __m256d) -> __m256d {
        abs_pd_256(delta)
    }

    #[inline(always)]
    unsafe fn dist_kn_f64x4(acc: __m256d, delta: __m256d) -> __m256d {
        _mm256_add_pd(acc, abs_pd_256(delta))
    }

    #[inline(always)]
    unsafe fn dist_k0_f64x2(delta: __m128d) -> __m128d {
        abs_pd_128(delta)
    }

    #[inline(always)]
    unsafe fn dist_kn_f64x2(acc: __m128d, delta: __m128d) -> __m128d {
        _mm_add_pd(acc, abs_pd_128(delta))
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

pub struct ManhattanAvx2F32LeafOps;

impl Avx2F32LeafOps for ManhattanAvx2F32LeafOps {
    #[inline(always)]
    unsafe fn dist_k0_f32x8(delta: __m256) -> __m256 {
        abs_ps_256(delta)
    }

    #[inline(always)]
    unsafe fn dist_kn_f32x8(acc: __m256, delta: __m256) -> __m256 {
        _mm256_add_ps(acc, abs_ps_256(delta))
    }

    #[inline(always)]
    unsafe fn dist_k0_f32x4(delta: __m128) -> __m128 {
        abs_ps_128(delta)
    }

    #[inline(always)]
    unsafe fn dist_kn_f32x4(acc: __m128, delta: __m128) -> __m128 {
        _mm_add_ps(acc, abs_ps_128(delta))
    }

    #[inline(always)]
    fn dist_k0_f32x1(delta: f32) -> f32 {
        delta.abs()
    }

    #[inline(always)]
    fn dist_kn_f32x1(acc: f32, delta: f32) -> f32 {
        acc + delta.abs()
    }
}

#[cfg(test)]
mod tests {
    use super::{
        abs_pd_128, abs_pd_256, abs_ps_128, abs_ps_256, ManhattanAvx2F32LeafOps,
        ManhattanAvx2F64LeafOps,
    };
    use crate::dist::distance_metric_avx2::{Avx2F32LeafOps, Avx2F64LeafOps};
    use std::arch::x86_64::*;

    #[cfg(target_feature = "avx2")]
    #[test]
    fn manhattan_avx2_f64_ops_compute_abs_and_accumulate() {
        unsafe {
            let v4 = _mm256_setr_pd(-1.0, 2.0, -3.5, 4.25);
            let abs4: [f64; 4] = std::mem::transmute(abs_pd_256(v4));
            assert_eq!(abs4, [1.0, 2.0, 3.5, 4.25]);

            let dist4: [f64; 4] = std::mem::transmute(ManhattanAvx2F64LeafOps::dist_k0_f64x4(v4));
            assert_eq!(dist4, [1.0, 2.0, 3.5, 4.25]);

            let acc4: [f64; 4] = std::mem::transmute(ManhattanAvx2F64LeafOps::dist_kn_f64x4(
                _mm256_set1_pd(10.0),
                v4,
            ));
            assert_eq!(acc4, [11.0, 12.0, 13.5, 14.25]);

            let v2 = _mm_setr_pd(-2.5, 7.0);
            let abs2: [f64; 2] = std::mem::transmute(abs_pd_128(v2));
            assert_eq!(abs2, [2.5, 7.0]);

            let dist2: [f64; 2] = std::mem::transmute(ManhattanAvx2F64LeafOps::dist_k0_f64x2(v2));
            assert_eq!(dist2, [2.5, 7.0]);

            let acc2: [f64; 2] =
                std::mem::transmute(ManhattanAvx2F64LeafOps::dist_kn_f64x2(_mm_set1_pd(3.0), v2));
            assert_eq!(acc2, [5.5, 10.0]);
        }

        assert_eq!(ManhattanAvx2F64LeafOps::dist_k0_f64x1(-3.0), 3.0);
        assert_eq!(ManhattanAvx2F64LeafOps::dist_kn_f64x1(5.0, -3.0), 8.0);
    }

    #[cfg(target_feature = "avx2")]
    #[test]
    fn manhattan_avx2_f32_ops_compute_abs_and_accumulate() {
        unsafe {
            let v8 = _mm256_setr_ps(-1.0, 2.0, -3.5, 4.25, -5.0, 6.0, -7.0, 8.0);
            let abs8: [f32; 8] = std::mem::transmute(abs_ps_256(v8));
            assert_eq!(abs8, [1.0, 2.0, 3.5, 4.25, 5.0, 6.0, 7.0, 8.0]);

            let dist8: [f32; 8] = std::mem::transmute(ManhattanAvx2F32LeafOps::dist_k0_f32x8(v8));
            assert_eq!(dist8, [1.0, 2.0, 3.5, 4.25, 5.0, 6.0, 7.0, 8.0]);

            let v4 = _mm_setr_ps(-1.5, 2.5, -3.0, 4.0);
            let abs4: [f32; 4] = std::mem::transmute(abs_ps_128(v4));
            assert_eq!(abs4, [1.5, 2.5, 3.0, 4.0]);
        }

        assert_eq!(ManhattanAvx2F32LeafOps::dist_k0_f32x1(-3.0), 3.0);
        assert_eq!(ManhattanAvx2F32LeafOps::dist_kn_f32x1(5.0, -3.0), 8.0);
    }
}
