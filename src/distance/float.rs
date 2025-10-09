//! Contains a selection of distance metrics that can be chosen from to measure the distance
//! between two points stored inside the tree.

// #[cfg(any(target_arch = "x86_64"))]
// use std::arch::x86_64::*;

use crate::traits::{Axis, DistanceMetric};

/// Returns the Manhattan / "taxi cab" / L1 distance between two points.
///
/// Faster than squared Euclidean, and just as effective if not more so in higher-dimensional spaces
///
/// re-exported as `kiddo::Manhattan` for convenience
///
/// # Examples
///
/// ```rust
/// use kiddo::traits::DistanceMetric;
/// use kiddo::Manhattan;
///
/// assert_eq!(0f32, Manhattan::dist(&[0f32, 0f32], &[0f32, 0f32]));
/// assert_eq!(1f32, Manhattan::dist(&[0f32, 0f32], &[1f32, 0f32]));
/// assert_eq!(2f32, Manhattan::dist(&[0f32, 0f32], &[1f32, 1f32]));
/// ```
pub struct Manhattan {}

impl<A: Axis, const K: usize> DistanceMetric<A, K> for Manhattan {
    #[cfg_attr(not(feature = "no_inline"), inline)]
    fn dist(a: &[A; K], b: &[A; K]) -> A {
        a.iter()
            .zip(b.iter())
            .map(|(&a_val, &b_val)| (a_val - b_val).abs())
            .fold(A::zero(), std::ops::Add::add)
    }

    #[cfg_attr(not(feature = "no_inline"), inline)]
    fn dist1(a: A, b: A) -> A {
        (a - b).abs()
    }
}

/// Returns the squared Euclidean / L2 distance between two points.
///
/// Faster than Euclidean distance due to not needing a square root, but still
/// preserves the same distance ordering as with Euclidean distance.
///
/// re-exported as `kiddo::SquaredEuclidean` for convenience
///
/// # Examples
///
/// ```rust
/// use kiddo::traits::DistanceMetric;
/// use kiddo::SquaredEuclidean;
///
/// assert_eq!(0f32, SquaredEuclidean::dist(&[0f32, 0f32], &[0f32, 0f32]));
/// assert_eq!(1f32, SquaredEuclidean::dist(&[0f32, 0f32], &[1f32, 0f32]));
/// assert_eq!(2f32, SquaredEuclidean::dist(&[0f32, 0f32], &[1f32, 1f32]));
/// ```
pub struct SquaredEuclidean {}

impl<A: Axis, const K: usize> DistanceMetric<A, K> for SquaredEuclidean {
    #[cfg_attr(not(feature = "no_inline"), inline)]
    fn dist(a: &[A; K], b: &[A; K]) -> A {
        a.iter()
            .zip(b.iter())
            .map(|(&a_val, &b_val)| (a_val - b_val) * (a_val - b_val))
            .fold(A::zero(), std::ops::Add::add)
    }

    #[cfg_attr(not(feature = "no_inline"), inline)]
    fn dist1(a: A, b: A) -> A {
        (a - b) * (a - b)
    }
}

/// Returns the *Negated* dot product of the two points.
///
/// Only suitable as a distance metric when the points being stored
/// represent vectors of the same length, and ideally of unit length.
/// NOTE that we actually return the *negative* of the dot product here,
/// since our queries consider values less than others to be shorter distances than them.
pub struct DotProduct {}

impl<A: Axis, const K: usize> DistanceMetric<A, K> for DotProduct {
    #[cfg_attr(not(feature = "no_inline"), inline)]
    fn dist(a: &[A; K], b: &[A; K]) -> A {
        a.iter()
            .zip(b.iter())
            .map(|(&a_val, &b_val)| a_val * b_val)
            .fold(A::zero(), std::ops::Sub::sub)
    }

    #[cfg_attr(not(feature = "no_inline"), inline)]
    fn dist1(a: A, b: A) -> A {
        a * b
    }
}
