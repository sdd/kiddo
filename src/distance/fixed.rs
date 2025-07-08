//! Defines different distance metrics.

// #[cfg(any(target_arch = "x86_64"))]
// use std::arch::x86_64::*;

use crate::traits::{AxisFixed, DistanceMetricFixed};

/// Returns the squared Euclidean distance between two points. When you only
/// need to compare distances, rather than having the exact distance between
/// the points, this metric is beneficial because it avoids the expensive square
/// root computation.
///
/// # Examples
///
/// ```rust
/// use fixed::types::extra::U0;
/// use fixed::{FixedU16, FixedU32};
/// use kiddo::traits::DistanceMetricFixed;
/// use kiddo::distance::fixed::Manhattan;
/// type Fxd = FixedU16<U0>;
/// type FxdR = FixedU32<U0>;
///
/// let ZERO_16 = Fxd::from_num(0);
/// let ONE_16 = Fxd::from_num(1);
/// let TWO_16 = Fxd::from_num(2);
/// let ZERO_32 = FxdR::from_num(0);
/// let ONE_32 = FxdR::from_num(1);
/// let TWO_32 = FxdR::from_num(2);
///
/// let result: FxdR =  Manhattan::dist(&[ZERO_16, ZERO_16], &[ZERO_16, ZERO_16]);
/// assert_eq!(result, ZERO_32);
/// let result: FxdR = Manhattan::dist(&[ZERO_16, ZERO_16], &[ONE_16, ZERO_16]);
/// assert_eq!(result, ONE_32);
/// let result: FxdR = Manhattan::dist(&[ZERO_16, ZERO_16], &[ONE_16, ONE_16]);
/// assert_eq!(result, TWO_32);
/// ```
pub struct Manhattan {}

impl<A: AxisFixed, R: AxisFixed, const K: usize> DistanceMetricFixed<A, K, R> for Manhattan {
    #[inline]
    fn dist(a: &[A; K], b: &[A; K]) -> R {
        a.iter()
            .zip(b.iter())
            .map(|(&a_val, &b_val)| {
                if a_val > b_val {
                    R::from_num(a_val) - R::from_num(b_val)
                } else {
                    R::from_num(b_val) - R::from_num(a_val)
                }
            })
            .fold(R::ZERO, |a, b| a.saturating_add(b))
    }

    #[inline]
    fn dist1(a: A, b: A) -> R {
        if a > b {
            R::from_num(a) - R::from_num(b)
        } else {
            R::from_num(b) - R::from_num(a)
        }
    }
}

/// Returns the squared Euclidean distance between two points.
///
/// Faster than Euclidean distance due to not needing a square root, but still
/// preserves the same distance ordering as with Euclidean distance.
///
/// # Examples
///
/// ```rust
/// use fixed::types::extra::U0;
/// use fixed::FixedU16;
/// use fixed::FixedU32;
/// use kiddo::traits::DistanceMetricFixed;
/// use kiddo::distance::fixed::SquaredEuclidean;
/// type Fxd16 = FixedU16<U0>;
/// type Fxd32 = FixedU32<U0>;
///
/// let ZERO_16 = Fxd16::from_num(0);
/// let ONE_16 = Fxd16::from_num(1);
/// let TWO_16 = Fxd16::from_num(2);
///
/// let ZERO_32 = Fxd32::from_num(0);
/// let ONE_32 = Fxd32::from_num(1);
/// let TWO_32 = Fxd32::from_num(2);
/// let EIGHT_32 = Fxd32::from_num(8);
///
/// let result: Fxd32 = SquaredEuclidean::dist(&[ZERO_16, ZERO_16], &[ZERO_16, ZERO_16]);
/// assert_eq!(result, ZERO_32);
/// let result: Fxd32 = SquaredEuclidean::dist(&[ZERO_16, ZERO_16], &[ONE_16, ZERO_16]);
/// assert_eq!(result, ONE_32);
/// let result: Fxd32 = SquaredEuclidean::dist(&[ZERO_16, ZERO_16], &[TWO_16, TWO_16]);
/// assert_eq!(result, EIGHT_32);
/// ```
pub struct SquaredEuclidean {}

impl<A: AxisFixed, R: AxisFixed, const K: usize> DistanceMetricFixed<A, K, R> for SquaredEuclidean {
    #[inline]
    fn dist(a: &[A; K], b: &[A; K]) -> R {
        a.iter()
            .zip(b.iter())
            .map(|(&a_val, &b_val)| {
                let diff: R = R::from_num(a_val).dist(R::from_num(b_val));
                diff * diff
            })
            .fold(R::ZERO, |a, b| a.saturating_add(b))
    }

    #[inline]
    fn dist1(a: A, b: A) -> R {
        let diff: R = R::from_num(a).dist(R::from_num(b));
        diff * diff
    }
}
