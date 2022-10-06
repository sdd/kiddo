//! Defines different distance metrics, in simplest case it defines the
//! euclidean distance which is no more than the square root of the sum of the
//! squares of the distances in each dimension.

use crate::tuned::u16::d4::kdtree::{A, PT};
// #[cfg(any(target_arch = "x86_64"))]
// use std::arch::x86_64::*;


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
        .map(|(a_val, b_val)| {
            let diff  = if a_val > b_val {
                a_val - b_val
            } else {
                b_val - a_val
            };
            let res = diff * diff;
            let res = res.floor() + res.frac();
            res
        })
        .fold(A::from_num(0), |acc, v| acc.saturating_add(v))
}

