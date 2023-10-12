//! Some experimental distance metrics work
#![allow(missing_docs)]

use num_traits::Float;

#[cfg(any(target_arch = "x86_64"))]
use std::arch::x86_64::*;

#[cfg(any(target_arch = "x86_64"))]
union SimdToArray {
    array: [f32; 4],
    simd: __m128,
}

pub fn squared_euclidean<T: Float, const K: usize>(a: &[T; K], b: &[T; K]) -> T {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| ((*x) - (*y)) * ((*x) - (*y)))
        .fold(T::zero(), ::std::ops::Add::add)
}

pub fn dot_product<const K: usize>(a: &[f32; K], b: &[f32; K]) -> f32 {
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
        simd: _mm_dp_ps(a_mm, b_mm, 0x71),
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
        simd: _mm_dp_ps(a_mm, b_mm, 0x71),
    };
    res.array[0]
}

#[cfg(any(target_arch = "x86_64"))]
pub fn dot_product_sse<const K: usize>(a: &[f32; K], b: &[f32; K]) -> f32 {
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
