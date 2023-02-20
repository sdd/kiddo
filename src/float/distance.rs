//! Contains a selection of distance metrics that can be chosen from to measure the distance
//! between two points stored inside the tree.

// #[cfg(any(target_arch = "x86_64"))]
// use std::arch::x86_64::*;

use crate::float::kdtree::Axis;

/// Returns the squared euclidean distance between two points.
///
/// Faster than Euclidean distance due to not needing a square root, but still
/// preserves the same distance ordering as with Euclidean distance.
///
/// # Examples
///
/// ```rust
/// use kiddo::float::distance::squared_euclidean;
///
/// assert_eq!(0f32, squared_euclidean(&[0f32, 0f32], &[0f32, 0f32]));
/// assert_eq!(1f32, squared_euclidean(&[0f32, 0f32], &[1f32, 0f32]));
/// assert_eq!(2f32, squared_euclidean(&[0f32, 0f32], &[1f32, 1f32]));
/// ```
pub fn squared_euclidean<A: Axis, const K: usize>(a: &[A; K], b: &[A; K]) -> A {
    a.iter()
        .zip(b.iter())
        .map(|(&a_val, &b_val)| (a_val - b_val) * (a_val - b_val))
        .fold(A::zero(), std::ops::Add::add)
}

/// Returns the Manhattan / "taxi cab" distance between two points.
///
/// Faster than squared Euclidean, and just as effective if not more so in higher-dimensional spaces
///
/// # Examples
///
/// ```rust
/// use kiddo::float::distance::manhattan;
///
/// assert_eq!(0f32, manhattan(&[0f32, 0f32], &[0f32, 0f32]));
/// assert_eq!(1f32, manhattan(&[0f32, 0f32], &[1f32, 0f32]));
/// assert_eq!(2f32, manhattan(&[0f32, 0f32], &[1f32, 1f32]));
/// ```
pub fn manhattan<A: Axis, const K: usize>(a: &[A; K], b: &[A; K]) -> A {
    a.iter()
        .zip(b.iter())
        .map(|(&a_val, &b_val)| (a_val - b_val).abs())
        .fold(A::zero(), std::ops::Add::add)
}
