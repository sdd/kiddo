//! Defines different distance metrics, in simplest case it defines the
//! euclidean distance which is no more than the square root of the sum of the
//! squares of the distances in each dimension.

use crate::tuned::f32::d4::kdtree::{A, K, PT};
#[cfg(any(target_arch = "x86_64"))]
use std::arch::x86_64::*;

#[cfg(any(target_arch = "x86_64"))]
union SimdToArray {
    array: [f32; 4],
    simd: __m128,
}

/// Returns the squared euclidean distance between two points. When you only
/// need to compare distances, rather than having the exact distance between
/// the points, this metric is beneficial because it avoids the expensive square
/// root computation.
///
/// # Examples
///
/// ```rust
/// use sok::distance::squared_euclidean;
///
/// assert!(0.0 == squared_euclidean(&[0.0, 0.0], &[0.0, 0.0]));
/// assert!(2.0 == squared_euclidean(&[0.0, 0.0], &[1.0, 1.0]));
/// assert!(1.0 == squared_euclidean(&[0.0, 0.0], &[1.0, 0.0]));
/// ```
pub fn squared_euclidean(a: &PT, b: &PT) -> A {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| ((*x) - (*y)) * ((*x) - (*y)))
        .fold(0.0, ::std::ops::Add::add)
}

pub fn squared_euclidean_simd_f32_d4(a: &PT, b: &PT) -> A {
    debug_assert!(a.as_ptr() as usize % 16 == 0);
    debug_assert!(b.as_ptr() as usize % 16 == 0);

    unsafe {
        let a = _mm_load_ps(a.as_ptr());
        let b = _mm_load_ps(b.as_ptr());

        let diff = _mm_sub_ps(a, b);

        let squared = _mm_mul_ps(diff, diff);

        let mut shuf = _mm_movehdup_ps(squared); // broadcast elements 3,1 to 2,0
        let mut sums = _mm_add_ps(squared, shuf);
        shuf = _mm_movehl_ps(shuf, sums); // high half -> low half
        sums = _mm_add_ss(sums, shuf);

        f32::from_bits(_mm_extract_ps::<0>(sums) as u32)
    }
}

pub fn squared_euclidean_simd_f32_d4_a_unaligned(a: &PT, b: &PT) -> A {
    debug_assert!(b.as_ptr() as usize % 16 == 0);

    unsafe {
        let a = _mm_loadu_ps(a.as_ptr());
        let b = _mm_load_ps(b.as_ptr());

        let diff = _mm_sub_ps(a, b);

        let squared = _mm_mul_ps(diff, diff);

        let mut shuf = _mm_movehdup_ps(squared); // broadcast elements 3,1 to 2,0
        let mut sums = _mm_add_ps(squared, shuf);
        shuf = _mm_movehl_ps(shuf, sums); // high half -> low half
        sums = _mm_add_ss(sums, shuf);

        f32::from_bits(_mm_extract_ps::<0>(sums) as u32)
    }
}

pub fn dot_product(a: &[f32; K], b: &[f32; K]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| ((*x) * (*y)))
        .fold(0f32, ::std::ops::Sub::sub)
}

#[cfg(any(target_arch = "x86_64"))]
#[target_feature(enable = "sse4.1")]
#[allow(clippy::missing_safety_doc)]
pub unsafe fn dot_sse(a: *const f32, b: *const f32) -> f32 {
    let a_mm = _mm_loadu_ps(a);
    let b_mm = _mm_loadu_ps(b);

    let res: SimdToArray = SimdToArray {
        simd: _mm_dp_ps::<113>(a_mm, b_mm),
    };
    res.array[0]
}

#[cfg(any(target_arch = "x86_64"))]
#[target_feature(enable = "sse4.1")]
#[allow(clippy::missing_safety_doc)]
pub unsafe fn dot_sse_aligned(a: *const f32, b: *const f32) -> f32 {
    let a_mm = _mm_load_ps(a);
    let b_mm = _mm_load_ps(b);

    let res: SimdToArray = SimdToArray {
        simd: _mm_dp_ps::<113>(a_mm, b_mm),
    };
    res.array[0]
}

#[cfg(any(target_arch = "x86_64"))]
pub fn dot_product_sse(a: &[f32; K], b: &[f32; K]) -> f32 {
    if K == 3 {
        dot_product_sse_3(&a[0..3], &a[0..3])
    } else if K == 4 {
        dot_product_sse_4(&a[0..4], &a[0..4])
    } else {
        dot_product(a, b)
    }
}

#[cfg(any(target_arch = "x86_64"))]
pub fn dot_product_sse_3(a: &[f32], b: &[f32]) -> f32 {
    let ap = [a[0], a[1], a[2], 0f32].as_ptr();
    let bp = [b[0], b[1], b[2], 0f32].as_ptr();
    unsafe { dot_sse(ap, bp) }
}

#[cfg(any(target_arch = "x86_64"))]
pub fn dot_product_sse_4(a: &[f32], b: &[f32]) -> f32 {
    unsafe { dot_sse(a.as_ptr(), b.as_ptr()) }
}

#[cfg(any(target_arch = "x86_64"))]
pub fn dot_product_sse_aligned(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let ap = a.as_ptr();
    let bp = b.as_ptr();
    unsafe { dot_sse_aligned(ap, bp) }
}
