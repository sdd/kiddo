//! Defines different distance metrics, in simplest case it defines the
//! euclidean distance which is no more than the square root of the sum of the
//! squares of the distances in each dimension.

// #[cfg(any(target_arch = "x86_64"))]
// use std::arch::x86_64::*;

use crate::float::kdtree::Axis;

/// Returns the squared euclidean distance between two points. When you only
/// need to compare distances, rather than having the exact distance between
/// the points, this metric is beneficial because it avoids the expensive square
/// root computation.
///
/// # Examples
///
/// ```rust
/// use sok::float::distance::manhattan;
///
/// assert_eq!(0f32, manhattan(&[0f32, 0f32], &[0f32, 0f32]));
/// assert_eq!(1f32, manhattan(&[0f32, 0f32], &[1f32, 0f32]));
/// assert_eq!(2f32, manhattan(&[0f32, 0f32], &[1f32, 1f32]));
/// ```

// #[inline(never)]
pub fn manhattan<A: Axis, const K: usize>(a: &[A; K], b: &[A; K]) -> A {
    a.iter()
        .zip(b.iter())
        .map(|(&a_val, &b_val)|
            (a_val - b_val).abs()
        )
        .fold(A::zero(), std::ops::Add::add)
}


// #[inline(never)]
pub fn squared_euclidean<A: Axis, const K: usize>(a: &[A; K], b: &[A; K]) -> A {
    a.iter()
        .zip(b.iter())
        .map(|(&a_val, &b_val)| {
            (a_val - b_val) * (a_val - b_val)
        })
        .fold(A::zero(), std::ops::Add::add)
}

