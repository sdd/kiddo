//! Defines different distance metrics, in simplest case it defines the
//! euclidean distance which is no more than the square root of the sum of the
//! squares of the distances in each dimension.

// #[cfg(any(target_arch = "x86_64"))]
// use std::arch::x86_64::*;

use crate::fixed::kdtree::Axis;

/// Returns the squared euclidean distance between two points.
///
/// When you only need to compare distances, rather than having the exact distance between
/// the points, this metric is beneficial because it avoids the expensive square
/// root computation.
///
/// # Examples
///
/// ```rust
/// use fixed::types::extra::U0;
/// use fixed::FixedU16;
/// use kiddo::fixed::distance::manhattan;
/// type FXD = FixedU16<U0>;
///
/// let ZERO = FXD::from_num(0);
/// let ONE = FXD::from_num(1);
/// let TWO = FXD::from_num(2);
///
/// assert!(ZERO == manhattan(&[ZERO, ZERO], &[ZERO, ZERO]));
/// assert!(ONE == manhattan(&[ZERO, ZERO], &[ONE, ZERO]));
/// assert!(TWO == manhattan(&[ZERO, ZERO], &[ONE, ONE]));
/// ```
pub fn manhattan<A: Axis, const K: usize>(a: &[A; K], b: &[A; K]) -> A {
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
/// use kiddo::fixed::distance::squared_euclidean;
/// type FXD = FixedU16<U0>;
///
/// let ZERO = FXD::from_num(0);
/// let ONE = FXD::from_num(1);
/// let TWO = FXD::from_num(2);
/// let EIGHT = FXD::from_num(8);
///
/// assert_eq!(squared_euclidean(&[ZERO, ZERO], &[ZERO, ZERO]), ZERO);
/// assert_eq!(squared_euclidean(&[ZERO, ZERO], &[ONE, ZERO]), ONE);
/// assert_eq!(squared_euclidean(&[ZERO, ZERO], &[TWO, TWO]), EIGHT);
/// ```
pub fn squared_euclidean<A: Axis, const K: usize>(a: &[A; K], b: &[A; K]) -> A {
    a.iter()
        .zip(b.iter())
        .map(|(&a_val, &b_val)| {
            // let diff: A = if a_val > b_val {
            //     a_val - b_val
            // } else {
            //     b_val - a_val
            // };
            let diff: A = a_val.dist(b_val);
            //let diff = a_val - b_val;
            //let diff = diff / A::from_num(4);

            //let res = diff.saturating_mul(diff);
            diff * diff

            // let res = (diff * A::from_num(0.5)).saturating_mul(diff); // / (A::from_num(K))
            // if res == A::MAX {
            //     println!("Dist saturated at mult (a={}, b={}, diff={})", &a_val, &b_val, &diff);
            // }
            //res
        })
        .fold(A::ZERO, |a, b| {
            // let res = a.saturating_add(b);
            a + b

            // let res = a.saturating_add(b.shr(2) );
            // if res == A::MAX {
            //     println!("Dist saturated at add")
            // }
            //res
        })
}
