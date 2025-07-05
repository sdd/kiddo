//! Defines different distance metrics, in simplest case it defines the
//! euclidean distance which is no more than the square root of the sum of the
//! squares of the distances in each dimension.

// #[cfg(any(target_arch = "x86_64"))]
// use std::arch::x86_64::*;

use crate::mutable::fixed::kdtree::Axis;
use crate::traits::DistanceMetric;

/// Returns the squared euclidean distance between two points. When you only
/// need to compare distances, rather than having the exact distance between
/// the points, this metric is beneficial because it avoids the expensive square
/// root computation.
///
/// # Examples
///
/// ```rust
/// use fixed::types::extra::U0;
/// use fixed::FixedU16;
/// use kiddo::traits::DistanceMetric;
/// use kiddo::distance::fixed::Manhattan;
/// type Fxd = FixedU16<U0>;
///
/// let ZERO = Fxd::from_num(0);
/// let ONE = Fxd::from_num(1);
/// let TWO = Fxd::from_num(2);
///
/// assert_eq!(ZERO, Manhattan::dist(&[ZERO, ZERO], &[ZERO, ZERO]));
/// assert_eq!(ONE, Manhattan::dist(&[ZERO, ZERO], &[ONE, ZERO]));
/// assert_eq!(TWO, Manhattan::dist(&[ZERO, ZERO], &[ONE, ONE]));
/// ```
pub struct Manhattan {}

impl<A: Axis, const K: usize> DistanceMetric<A, K> for Manhattan {
    #[inline]
    fn dist(a: &[A; K], b: &[A; K]) -> A {
        a.iter()
            .zip(b.iter())
            .map(|(&a_val, &b_val)| {
                if a_val > b_val {
                    a_val - b_val
                } else {
                    b_val - a_val
                }
            })
            .fold(A::ZERO, |a, b| a.saturating_add(b))
    }

    #[inline]
    fn dist1(a: A, b: A) -> A {
        if a > b {
            a - b
        } else {
            b - a
        }
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
/// use fixed::types::extra::U0;
/// use fixed::FixedU16;
/// use kiddo::traits::DistanceMetric;
/// use kiddo::distance::fixed::SquaredEuclidean;
/// type Fxd = FixedU16<U0>;
///
/// let ZERO = Fxd::from_num(0);
/// let ONE = Fxd::from_num(1);
/// let TWO = Fxd::from_num(2);
/// let EIGHT = Fxd::from_num(8);
///
/// assert_eq!(SquaredEuclidean::dist(&[ZERO, ZERO], &[ZERO, ZERO]), ZERO);
/// assert_eq!(SquaredEuclidean::dist(&[ZERO, ZERO], &[ONE, ZERO]), ONE);
/// assert_eq!(SquaredEuclidean::dist(&[ZERO, ZERO], &[TWO, TWO]), EIGHT);
/// ```
pub struct SquaredEuclidean {}

impl<A: Axis, const K: usize> DistanceMetric<A, K> for SquaredEuclidean {
    #[inline]
    fn dist(a: &[A; K], b: &[A; K]) -> A {
        a.iter()
            .zip(b.iter())
            .map(|(&a_val, &b_val)| {
                let diff: A = a_val.dist(b_val);
                diff * diff
            })
            .fold(A::ZERO, |a, b| a.saturating_add(b))
    }

    #[inline]
    fn dist1(a: A, b: A) -> A {
        let diff: A = a.dist(b);
        diff * diff
    }
}
