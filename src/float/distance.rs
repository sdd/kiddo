//! Contains a selection of distance metrics that can be chosen from to measure the distance
//! between two points stored inside the tree.

// #[cfg(any(target_arch = "x86_64"))]
// use std::arch::x86_64::*;

use crate::distance_metric::DistanceMetric;
use crate::float::kdtree::Axis;

/// Returns the Manhattan / "taxi cab" distance between two points.
///
/// Faster than squared Euclidean, and just as effective if not more so in higher-dimensional spaces
///
/// # Examples
///
/// ```rust
/// use kiddo::float::distance::{DistanceMetric, Manhattan};
///
/// assert_eq!(0f32, Manhattan::dist(&[0f32, 0f32], &[0f32, 0f32]));
/// assert_eq!(1f32, Manhattan::dist(&[0f32, 0f32], &[1f32, 0f32]));
/// assert_eq!(2f32, Manhattan::dist(&[0f32, 0f32], &[1f32, 1f32]));
/// ```
pub struct Manhattan {}

impl<A: Axis, const K: usize> DistanceMetric<A, K> for Manhattan {
    #[inline]
    fn dist(a: &[A; K], b: &[A; K]) -> A {
        a.iter()
            .zip(b.iter())
            .map(|(&a_val, &b_val)| (a_val - b_val).abs())
            .fold(A::zero(), std::ops::Add::add)
    }

    #[inline]
    fn dist1(a: A, b: A) -> A {
        (a - b).abs()
    }
}

/// Returns the squared euclidean distance between two points.
///
/// Faster than Euclidean distance due to not needing a square root, but still
/// preserves the same distance ordering as with Euclidean distance.
///
/// # Examples
///
/// ```rust
/// use kiddo::float::distance::{DistanceMetric, SquaredEuclidean};
///
/// assert_eq!(0f32, SquaredEuclidean::dist(&[0f32, 0f32], &[0f32, 0f32]));
/// assert_eq!(1f32, SquaredEuclidean::dist(&[0f32, 0f32], &[1f32, 0f32]));
/// assert_eq!(2f32, SquaredEuclidean::dist(&[0f32, 0f32], &[1f32, 1f32]));
/// ```
pub struct SquaredEuclidean {}

impl<A: Axis, const K: usize> DistanceMetric<A, K> for SquaredEuclidean {
    #[inline]
    fn dist(a: &[A; K], b: &[A; K]) -> A {
        a.iter()
            .zip(b.iter())
            .map(|(&a_val, &b_val)| (a_val - b_val) * (a_val - b_val))
            .fold(A::zero(), std::ops::Add::add)
    }

    #[inline]
    fn dist1(a: A, b: A) -> A {
        (a - b) * (a - b)
    }
}
