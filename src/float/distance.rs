//! Contains a selection of distance metrics that can be chosen from to measure the distance
//! between two points stored inside the tree.

// #[cfg(any(target_arch = "x86_64"))]
// use std::arch::x86_64::*;

use crate::float::kdtree::Axis;
use crate::traits::DistanceMetric;

/// Returns the Manhattan / "taxi cab" distance between two points.
///
/// Faster than squared Euclidean, and just as effective if not more so in higher-dimensional spaces
///
/// re-exported as `kiddo::Manhattan` for convenience
#[doc(alias = "minkowski")]
#[doc(alias = "l1")]
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

    #[inline]
    fn accumulate(rd: A, delta: A) -> A {
        rd + delta
    }

    const IS_MAX_BASED: bool = false;
}

/// Returns the Chebyshev / L-infinity distance between two points.
///
/// Chebyshev distance is the maximum absolute difference along any axis.
/// Also known as chessboard distance or L-infinity norm.
///
/// re-exported as `kiddo::Chebyshev` for convenience
///
/// # Examples
///
/// ```rust
/// use kiddo::traits::DistanceMetric;
/// use kiddo::Chebyshev;
///
/// assert_eq!(0f32, Chebyshev::dist(&[0f32, 0f32], &[0f32, 0f32]));
/// assert_eq!(1f32, Chebyshev::dist(&[0f32, 0f32], &[1f32, 0f32]));
/// assert_eq!(1f32, Chebyshev::dist(&[0f32, 0f32], &[1f32, 1f32]));
/// ```
pub struct Chebyshev {}

impl<A: Axis, const K: usize> DistanceMetric<A, K> for Chebyshev {
    #[inline]
    fn dist(a: &[A; K], b: &[A; K]) -> A {
        a.iter()
            .zip(b.iter())
            .map(|(&a_val, &b_val)| (a_val - b_val).abs())
            .fold(A::zero(), |acc, val| acc.max(val))
    }

    #[inline]
    fn dist1(a: A, b: A) -> A {
        (a - b).abs()
    }

    #[inline]
    fn accumulate(rd: A, delta: A) -> A {
        rd.max(delta)
    }
}

/// Returns the squared euclidean distance between two points.
///
/// Faster than Euclidean distance due to not needing a square root, but still
/// preserves the same distance ordering as with Euclidean distance.
///
/// re-exported as `kiddo::SquaredEuclidean` for convenience
#[doc(alias = "minkowski")]
#[doc(alias = "l2")]
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

    #[inline]
    fn accumulate(rd: A, delta: A) -> A {
        rd + delta
    }

    const IS_MAX_BASED: bool = false;
}

/// Returns the Minkowski distance (power distance) between two points.
///
/// This implementation returns the sum of the powers of the absolute differences:
/// `dist(p,q) = Σ |p[i] - q[i]|^P`.
/// To get the actual Minkowski distance (L_P norm), you must take the 1/P root
/// of the result.
///
/// This struct provides the generalised functionality for:
/// - P < 1: Fractional distance metrics
/// - P ∈ (1, 2): Interpolation between Manhattan and Euclidean
/// - P > 2: Distances emphasizing larger differences (can lead to overflow)
///
/// # Compile-time Warnings for Special Cases
///
/// Using `P=1` or `P=2` will result in a compile-time error:
///
/// - For `P=1`: Use [`Manhattan`] instead - it's faster and avoids the overhead
///   of `powi(1)`.
/// - For `P=2`: Use [`SquaredEuclidean`] instead.
///
/// This is enforced at compile time because these specific values have
/// specialised.
///
/// # Performance
/// Using `u32` for `P` is significantly faster than using a floating-point `P`,
/// as it uses `powi` instead of `powf`.
///
/// # Overflow
/// High values of `P` can easily result in overflows, especially with large
/// coordinate differences.
///
/// re-exported as `kiddo::Minkowski` for convenience
#[doc(alias = "taxicab")]
#[doc(alias = "l1")]
#[doc(alias = "l2")]
#[doc(alias = "euclidean")]
pub struct Minkowski<const P: u32> {}

impl<const P: u32> Minkowski<P> {
    const CHECK_P: () = {
        if P == 1 {
            panic!("Minkowski<1> is not recommended. Use `kiddo::Manhattan` metric instead.");
        }
        if P == 2 {
            panic!(
                "Minkowski<2> is not recommended. Use `kiddo::SquaredEuclidean` metric instead."
            );
        }
    };
}

impl<A: Axis, const K: usize, const P: u32> DistanceMetric<A, K> for Minkowski<P> {
    #[inline]
    #[allow(clippy::let_unit_value)]
    fn dist(a: &[A; K], b: &[A; K]) -> A {
        let _ = Self::CHECK_P;

        a.iter()
            .zip(b.iter())
            .map(|(&av, &bv)| (av - bv).abs().powi(P as i32))
            .fold(A::zero(), std::ops::Add::add)
    }

    #[inline]
    #[allow(clippy::let_unit_value)]
    fn dist1(a: A, b: A) -> A {
        let _ = Self::CHECK_P;

        (a - b).abs().powi(P as i32)
    }

    #[inline]
    fn accumulate(rd: A, delta: A) -> A {
        rd + delta
    }
}

/// Returns the Minkowski distance (power distance) between two points with a floating-point power `P`.
///
/// This implementation returns the sum of the powers of the absolute differences:
/// `dist(p,q) = Σ |p[i] - q[i]|^P`.
/// To get the actual Minkowski distance (L_P norm), you must take the 1/P root
/// of the result.
///
/// This struct provides the generalised functionality for:
/// - P < 1: Fractional distance metrics
/// - P ∈ (1, 2): Interpolation between Manhattan and Euclidean
/// - P > 2: Distances emphasizing larger differences (can lead to overflow)
///
/// # Const Generics Hack
/// Since Rust does not yet support `f64` in const generics, the power `P` is
/// passed as its `u64` bit representation via `P_BITS`. Use `f64::to_bits()`
/// to provide this value.
///
/// # Compile-time Warnings for Special Cases
///
/// Using `P=1` or `P=2` will result in a compile-time error:
///
/// - For `P=1`: Use [`Manhattan`] instead - it's faster and avoids the overhead
///   of `powi(1)`.
/// - For `P=2`: Use [`SquaredEuclidean`] instead.
/// - If `P` is an integer, consider using [`Minkowski<P>`] instead.
///
/// This is enforced at compile time because these specific values have
/// specialised.
///
/// # Performance
/// This version uses a floating-point power and is significantly slower than
/// `Minkowski<P>` which uses an integer power. Please use the latter if possible.
///
/// # Overflow
/// High values of `P` can easily result in overflows, especially with large
/// coordinate differences.
///
/// re-exported as `kiddo::MinkowskiF64` for convenience
#[doc(alias = "taxicab")]
#[doc(alias = "l1")]
#[doc(alias = "l2")]
#[doc(alias = "euclidean")]
pub struct MinkowskiF64<const P_BITS: u64> {}

impl<const P_BITS: u64> MinkowskiF64<P_BITS> {
    const CHECK_P: () = {
        let p = f64::from_bits(P_BITS);
        if (p - 1.0).abs() < f64::EPSILON {
            panic!(
                "MinkowskiF64<P=1.0> is not recommended. Use `kiddo::Manhattan` metric instead."
            );
        }
        if (p - 2.0).abs() < f64::EPSILON {
            panic!(
                "MinkowskiF64<P=2.0> is not recommended. Use `kiddo::SquaredEuclidean` metric instead."
            );
        }
        // Check if P is an integer. Then suggest to use Minkowski<P> instead.
        if p.fract() < f64::EPSILON {
            panic!(
                "MinkowskiF64<P as F64> with power that is basically integer. Consider using Minkowski<P as u32> instead.",
            );
        }
    };
}

impl<A: Axis, const K: usize, const P_BITS: u64> DistanceMetric<A, K> for MinkowskiF64<P_BITS> {
    #[inline]
    #[allow(clippy::let_unit_value)]
    fn dist(a: &[A; K], b: &[A; K]) -> A {
        let _ = Self::CHECK_P;

        let p = f64::from_bits(P_BITS);
        a.iter()
            .zip(b.iter())
            .map(|(&av, &bv)| {
                let diff = (av - bv).abs().to_f64().unwrap();
                A::from(diff.powf(p)).unwrap()
            })
            .fold(A::zero(), std::ops::Add::add)
    }

    #[inline]
    #[allow(clippy::let_unit_value)]
    fn dist1(a: A, b: A) -> A {
        let _ = Self::CHECK_P;

        let p = f64::from_bits(P_BITS);
        let diff = (a - b).abs().to_f64().unwrap();
        A::from(diff.powf(p)).unwrap()
    }

    #[inline]
    fn accumulate(rd: A, delta: A) -> A {
        rd + delta
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    mod common_metric_tests {
        use super::*;

        #[rstest]
        #[case::zeros_1d([0.0f32], [0.0f32])]
        #[case::normal_1d([1.0f32], [2.0f32])]
        #[case::neg_1d([-1.0f32], [1.0f32])]
        #[case::zeros_2d([0.0f32, 0.0f32], [0.0f32, 0.0f32])]
        #[case::normal_2d([1.0f32, 2.0f32], [3.0f32, 4.0f32])]
        #[case::large_2d([1e30f32, 1e30f32], [-1e30f32, -1e30f32])]
        #[case::zeros_3d([0.0f32, 0.0f32, 0.0f32], [0.0f32, 0.0f32, 0.0f32])]
        #[case::normal_3d([1.0f32, 2.0f32, 3.0f32], [4.0f32, 5.0f32, 6.0f32])]
        #[case::zeros_4d([0.0f32; 4], [0.0f32; 4])]
        #[case::normal_4d([1.0f32; 4], [2.0f32; 4])]
        #[case::zeros_5d([0.0f32; 5], [0.0f32; 5])]
        #[case::normal_5d([1.0f32; 5], [2.0f32; 5])]
        fn test_metric_non_negativity<A: Axis, const K: usize, D: DistanceMetric<A, K>>(
            #[values(
                Manhattan {},
                SquaredEuclidean {},
                Chebyshev {},
                Minkowski::<3> {},
                MinkowskiF64::<{ 0.5f64.to_bits() }> {}
            )]
            _metric: D,
            #[case] a: [A; K],
            #[case] b: [A; K],
        ) {
            let distance = D::dist(&a, &b);
            assert!(distance >= A::zero());
        }

        #[rstest]
        #[case::zeros_1d([0.0f32])]
        #[case::normal_1d([1.0f32])]
        #[case::zeros_2d([0.0f32, 0.0f32])]
        #[case::normal_2d([1.0f32, 2.0f32])]
        #[case::zeros_3d([0.0f32, 0.0f32, 0.0f32])]
        #[case::normal_3d([1.0f32, 2.0f32, 3.0f32])]
        #[case::zeros_4d([0.0f32; 4])]
        #[case::zeros_5d([0.0f32; 5])]
        fn test_metric_identity<A: Axis, const K: usize, D: DistanceMetric<A, K>>(
            #[values(
                Manhattan {},
                SquaredEuclidean {},
                Chebyshev {},
                Minkowski::<3> {},
                MinkowskiF64::<{ 0.5f64.to_bits() }> {}
            )]
            _metric: D,
            #[case] a: [A; K],
        ) {
            assert_eq!(D::dist(&a, &a), A::zero());
        }

        #[rstest]
        #[case::normal_1d([1.0f64], [2.0f64])]
        #[case::neg_1d([-1.0f64], [1.0f64])]
        #[case::normal_2d([1.0f64, 2.0f64], [3.0f64, 4.0f64])]
        #[case::normal_3d([1.0f64, 2.0f64, 3.0f64], [4.0f64, 5.0f64, 6.0f64])]
        #[case::normal_4d([1.0f64; 4], [2.0f64; 4])]
        #[case::normal_5d([1.0f64; 5], [2.0f64; 5])]
        fn test_metric_symmetry<A: Axis, const K: usize, D: DistanceMetric<A, K>>(
            #[values(
                Manhattan {},
                SquaredEuclidean {},
                Chebyshev {},
                Minkowski::<3> {},
                MinkowskiF64::<{ 0.5f64.to_bits() }> {}
            )]
            _metric: D,
            #[case] a: [A; K],
            #[case] b: [A; K],
        ) {
            assert_eq!(D::dist(&a, &b), D::dist(&b, &a));
        }
    }

    mod manhattan_tests {
        use super::*;

        #[rstest]
        #[case([0.0f32, 0.0f32], [0.0f32, 0.0f32], 0.0f32)] // identical points
        #[case([0.0f32, 0.0f32], [1.0f32, 0.0f32], 1.0f32)] // single axis difference
        #[case([0.0f32, 0.0f32], [0.0f32, 1.0f32], 1.0f32)] // single axis difference (other axis)
        #[case([0.0f32, 0.0f32], [1.0f32, 1.0f32], 2.0f32)] // diagonal
        #[case([-1.0f32, -1.0f32], [1.0f32, 1.0f32], 4.0f32)] // negative to positive
        #[case([1.5f32, 2.5f32], [3.5f32, 4.5f32], 4.0f32)] // fractional values
        fn test_manhattan_distance_2d(
            #[case] a: [f32; 2],
            #[case] b: [f32; 2],
            #[case] expected: f32,
        ) {
            assert_eq!(Manhattan::dist(&a, &b), expected);
        }

        #[rstest]
        #[case([0.0f64, 0.0f64, 0.0f64], [0.0f64, 0.0f64, 0.0f64], 0.0f64)] // identical points 3D
        #[case([0.0f64, 0.0f64, 0.0f64], [1.0f64, 2.0f64, 3.0f64], 6.0f64)] // 3D diagonal
        #[case([1.0f64, 2.0f64, 3.0f64], [4.0f64, 5.0f64, 6.0f64], 9.0f64)] // 3D offset
        fn test_manhattan_distance_3d(
            #[case] a: [f64; 3],
            #[case] b: [f64; 3],
            #[case] expected: f64,
        ) {
            assert_eq!(Manhattan::dist(&a, &b), expected);
        }

        #[rstest]
        #[case([0.0f32], [0.0f32], 0.0f32)] // 1D identical
        #[case([0.0f32], [5.0f32], 5.0f32)] // 1D positive
        #[case([5.0f32], [0.0f32], 5.0f32)] // 1D negative (reversed)
        #[case([-3.0f32], [7.0f32], 10.0f32)] // 1D negative to positive
        fn test_manhattan_distance_1d(
            #[case] a: [f32; 1],
            #[case] b: [f32; 1],
            #[case] expected: f32,
        ) {
            assert_eq!(Manhattan::dist(&a, &b), expected);
        }

        #[test]
        fn test_manhattan_distance_4d() {
            let a = [1.0f32, 2.0f32, 3.0f32, 4.0f32];
            let b = [5.0f32, 6.0f32, 7.0f32, 8.0f32];
            let expected = 16.0f32; // |5-1| + |6-2| + |7-3| + |8-4| = 4+4+4+4 = 16
            assert_eq!(Manhattan::dist(&a, &b), expected);
        }

        #[test]
        fn test_manhattan_distance_5d() {
            let a = [0.0f64, 1.0f64, 2.0f64, 3.0f64, 4.0f64];
            let b = [5.0f64, 6.0f64, 7.0f64, 8.0f64, 9.0f64];
            let expected = 25.0f64; // |5-0| + |6-1| + |7-2| + |8-3| + |9-4| = 5+5+5+5+5 = 25
            assert_eq!(Manhattan::dist(&a, &b), expected);
        }

        #[test]
        fn test_manhattan_dist1() {
            assert_eq!(
                <Manhattan as DistanceMetric<f32, 1>>::dist1(0.0f32, 0.0f32),
                0.0f32
            ); // zero difference
            assert_eq!(
                <Manhattan as DistanceMetric<f32, 1>>::dist1(1.0f32, 0.0f32),
                1.0f32
            ); // positive difference
            assert_eq!(
                <Manhattan as DistanceMetric<f32, 1>>::dist1(0.0f32, 1.0f32),
                1.0f32
            ); // negative difference (reversed)
            assert_eq!(
                <Manhattan as DistanceMetric<f32, 1>>::dist1(-2.5f32, 3.5f32),
                6.0f32
            ); // fractional negative to positive
            assert_eq!(
                <Manhattan as DistanceMetric<f32, 1>>::dist1(1000.0f32, -1000.0f32),
                2000.0f32
            ); // large values
        }
    }

    mod squared_euclidean_tests {
        use super::*;

        #[rstest]
        #[case([0.0f32, 0.0f32], [0.0f32, 0.0f32], 0.0f32)] // identical points
        #[case([0.0f32, 0.0f32], [1.0f32, 0.0f32], 1.0f32)] // single axis difference
        #[case([0.0f32, 0.0f32], [0.0f32, 1.0f32], 1.0f32)] // single axis difference (other axis)
        #[case([0.0f32, 0.0f32], [1.0f32, 1.0f32], 2.0f32)] // diagonal (1^2 + 1^2)
        #[case([-1.0f32, -1.0f32], [1.0f32, 1.0f32], 8.0f32)] // negative to positive (2^2 + 2^2)
        #[case([1.5f32, 2.5f32], [3.5f32, 4.5f32], 8.0f32)] // fractional values (2^2 + 2^2)
        #[case([0.0f32, 0.0f32], [3.0f32, 4.0f32], 25.0f32)] // 3-4-5 triangle
        fn test_squared_euclidean_distance_2d(
            #[case] a: [f32; 2],
            #[case] b: [f32; 2],
            #[case] expected: f32,
        ) {
            assert_eq!(SquaredEuclidean::dist(&a, &b), expected);
        }

        #[rstest]
        #[case([0.0f64, 0.0f64, 0.0f64], [0.0f64, 0.0f64, 0.0f64], 0.0f64)] // identical points 3D
        #[case([0.0f64, 0.0f64, 0.0f64], [1.0f64, 2.0f64, 2.0f64], 9.0f64)] // 3D (1^2 + 2^2 + 2^2)
        #[case([1.0f64, 2.0f64, 3.0f64], [4.0f64, 5.0f64, 6.0f64], 27.0f64)] // 3D offset (3^2 + 3^2 + 3^2)
        fn test_squared_euclidean_distance_3d(
            #[case] a: [f64; 3],
            #[case] b: [f64; 3],
            #[case] expected: f64,
        ) {
            assert_eq!(SquaredEuclidean::dist(&a, &b), expected);
        }

        #[rstest]
        #[case([0.0f32], [0.0f32], 0.0f32)] // 1D identical
        #[case([0.0f32], [5.0f32], 25.0f32)] // 1D positive (5^2)
        #[case([5.0f32], [0.0f32], 25.0f32)] // 1D negative (reversed)
        #[case([-3.0f32], [7.0f32], 100.0f32)] // 1D negative to positive (10^2)
        fn test_squared_euclidean_distance_1d(
            #[case] a: [f32; 1],
            #[case] b: [f32; 1],
            #[case] expected: f32,
        ) {
            assert_eq!(SquaredEuclidean::dist(&a, &b), expected);
        }

        #[test]
        fn test_squared_euclidean_dist1() {
            assert_eq!(
                <SquaredEuclidean as DistanceMetric<f32, 1>>::dist1(0.0f32, 0.0f32),
                0.0f32
            ); // zero difference
            assert_eq!(
                <SquaredEuclidean as DistanceMetric<f32, 1>>::dist1(1.0f32, 0.0f32),
                1.0f32
            ); // positive difference
            assert_eq!(
                <SquaredEuclidean as DistanceMetric<f32, 1>>::dist1(0.0f32, 1.0f32),
                1.0f32
            ); // negative difference (reversed)
            assert_eq!(
                <SquaredEuclidean as DistanceMetric<f32, 1>>::dist1(-2.5f32, 3.5f32),
                36.0f32
            ); // fractional negative to positive (6^2)
            assert_eq!(
                <SquaredEuclidean as DistanceMetric<f32, 1>>::dist1(10.0f32, -10.0f32),
                400.0f32
            ); // large values (20^2)
        }

        #[test]
        fn test_squared_euclidean_triangle_inequality_property() {
            // Test that squared Euclidean distance preserves ordering
            let a = [0.0f32, 0.0f32];
            let b = [1.0f32, 0.0f32];
            let c = [1.0f32, 1.0f32];

            let dist_ab = SquaredEuclidean::dist(&a, &b);
            let dist_ac = SquaredEuclidean::dist(&a, &c);
            let dist_bc = SquaredEuclidean::dist(&b, &c);

            // For these points: dist(a,b) = 1, dist(b,c) = 1, dist(a,c) = 2
            assert_eq!(dist_ab, 1.0f32);
            assert_eq!(dist_bc, 1.0f32);
            assert_eq!(dist_ac, 2.0f32);
        }
    }

    mod chebyshev_tests {
        use super::*;

        #[rstest]
        #[case([0.0f32, 0.0f32], [0.0f32, 0.0f32], 0.0f32)] // identical points
        #[case([0.0f32, 0.0f32], [1.0f32, 0.0f32], 1.0f32)] // single axis difference
        #[case([0.0f32, 0.0f32], [0.0f32, 1.0f32], 1.0f32)] // single axis difference (other axis)
        #[case([0.0f32, 0.0f32], [1.0f32, 1.0f32], 1.0f32)] // diagonal
        #[case([-1.0f32, -1.0f32], [1.0f32, 1.0f32], 2.0f32)] // negative to positive
        #[case([1.5f32, 2.5f32], [3.5f32, 4.5f32], 2.0f32)] // fractional values
        #[case([0.0f32, 0.0f32], [2.0f32, 1.0f32], 2.0f32)] // max on first axis
        #[case([0.0f32, 0.0f32], [1.0f32, 2.0f32], 2.0f32)] // max on second axis
        fn test_chebyshev_distance_2d(
            #[case] a: [f32; 2],
            #[case] b: [f32; 2],
            #[case] expected: f32,
        ) {
            assert_eq!(Chebyshev::dist(&a, &b), expected);
        }

        #[rstest]
        #[case([0.0f64, 0.0f64, 0.0f64], [0.0f64, 0.0f64, 0.0f64], 0.0f64)] // identical points 3D
        #[case([0.0f64, 0.0f64, 0.0f64], [1.0f64, 2.0f64, 3.0f64], 3.0f64)] // 3D diagonal (max is 3)
        #[case([1.0f64, 2.0f64, 3.0f64], [4.0f64, 5.0f64, 6.0f64], 3.0f64)] // 3D offset (max is 3)
        fn test_chebyshev_distance_3d(
            #[case] a: [f64; 3],
            #[case] b: [f64; 3],
            #[case] expected: f64,
        ) {
            assert_eq!(Chebyshev::dist(&a, &b), expected);
        }

        #[rstest]
        #[case([0.0f32], [0.0f32], 0.0f32)] // 1D identical
        #[case([0.0f32], [5.0f32], 5.0f32)] // 1D positive
        #[case([5.0f32], [0.0f32], 5.0f32)] // 1D negative (reversed)
        #[case([-3.0f32], [7.0f32], 10.0f32)] // 1D negative to positive
        fn test_chebyshev_distance_1d(
            #[case] a: [f32; 1],
            #[case] b: [f32; 1],
            #[case] expected: f32,
        ) {
            assert_eq!(Chebyshev::dist(&a, &b), expected);
        }

        #[test]
        fn test_chebyshev_distance_4d() {
            let a = [1.0f32, 2.0f32, 3.0f32, 4.0f32];
            let b = [5.0f32, 6.0f32, 7.0f32, 8.0f32];
            let expected = 4.0f32; // max(|5-1|, |6-2|, |7-3|, |8-4|) = max(4, 4, 4, 4) = 4
            assert_eq!(Chebyshev::dist(&a, &b), expected);
        }

        #[test]
        fn test_chebyshev_distance_5d() {
            let a = [0.0f64, 1.0f64, 2.0f64, 3.0f64, 4.0f64];
            let b = [5.0f64, 6.0f64, 7.0f64, 8.0f64, 9.0f64];
            let expected = 5.0f64; // max(|5-0|, |6-1|, |7-2|, |8-3|, |9-4|) = max(5, 5, 5, 5, 5) = 5
            assert_eq!(Chebyshev::dist(&a, &b), expected);
        }

        #[rstest]
        #[case(0.0f32, 0.0f32, 0.0f32)] // zero difference
        #[case(1.0f32, 0.0f32, 1.0f32)] // positive difference
        #[case(0.0f32, 1.0f32, 1.0f32)] // negative difference (reversed)
        #[case(-2.5f32, 3.5f32, 6.0f32)] // fractional negative to positive
        #[case(1000.0f32, -1000.0f32, 2000.0f32)] // large values
        fn test_chebyshev_dist1(#[case] a: f32, #[case] b: f32, #[case] expected: f32) {
            assert_eq!(<Chebyshev as DistanceMetric<f32, 1>>::dist1(a, b), expected);
        }

        #[test]
        fn test_chebyshev_symmetry() {
            let a = [1.0f64, 2.0f64, 3.0f64];
            let b = [4.0f64, 5.0f64, 6.0f64];
            assert_eq!(Chebyshev::dist(&a, &b), Chebyshev::dist(&b, &a));
        }

        #[test]
        fn test_chebyshev_identity() {
            let a = [1.0f32, 2.0f32, 3.0f32];
            assert_eq!(Chebyshev::dist(&a, &a), 0.0f32);
        }

        #[test]
        fn test_chebyshev_non_negativity() {
            let a = [1.0f32, 2.0f32];
            let b = [3.0f32, 4.0f32];
            let distance = Chebyshev::dist(&a, &b);
            assert!(distance >= 0.0f32);
        }

        #[test]
        fn test_chebyshev_max_property() {
            // Test that Chebyshev correctly finds the maximum difference
            let a = [0.0, 0.0];
            let b = [3.0, 1.0];

            let result = Chebyshev::dist(&a, &b);

            // max(|0-3|, |0-1|) = max(3, 1) = 3
            assert_eq!(result, 3.0);

            // Verify it's not Manhattan (which would be 4) or Euclidean (sqrt(10))
            assert_ne!(result, 4.0);
            assert_ne!(result, (10.0_f64).sqrt());
        }
    }

    mod minkowski_tests {
        use super::*;

        #[rstest]
        #[case([0.0f32, 0.0f32], [0.0f32, 0.0f32], 0.0f32)] // identical
        #[case([0.0f32, 0.0f32], [1.0f32, 1.0f32], 2.0f32)] // |0-1|^3 + |0-1|^3 = 1+1=2
        #[case([0.0f32, 0.0f32], [2.0f32, 0.0f32], 8.0f32)] // |0-2|^3 = 8
        #[case([-1.0f32, -1.0f32], [1.0f32, 1.0f32], 16.0f32)] // | -1 - 1|^3 + |-1 - 1|^3 = 8 + 8 = 16
        fn test_minkowski_3_distance_2d(
            #[case] a: [f32; 2],
            #[case] b: [f32; 2],
            #[case] expected: f32,
        ) {
            assert_eq!(Minkowski::<3>::dist(&a, &b), expected);
        }

        #[rstest]
        #[case([0.0f64, 0.0f64, 0.0f64], [1.0f64, 1.0f64, 1.0f64], 3.0f64)]
        #[case([1.0f64, 2.0f64, 3.0f64], [4.0f64, 5.0f64, 6.0f64], 81.0f64)] // 3^3 + 3^3 + 3^3 = 27*3 = 81
        fn test_minkowski_3_distance_3d(
            #[case] a: [f64; 3],
            #[case] b: [f64; 3],
            #[case] expected: f64,
        ) {
            assert_eq!(Minkowski::<3>::dist(&a, &b), expected);
        }

        #[rstest]
        #[case([0.0f32, 0.0f32], [1.0f32, 1.0f32], 2.0f32)] // |0-1|^0.5 + |0-1|^0.5 = 1+1=2
        #[case([0.0f32, 0.0f32], [4.0f32, 9.0f32], 5.0f32)] // |0-4|^0.5 + |0-9|^0.5 = 2+3=5
        #[case([1.0f32, 1.0f32], [5.0f32, 10.0f32], 5.0f32)] // |1-5|^0.5 + |1-10|^0.5 = 2+3=5
        fn test_minkowski_05_distance_2d(
            #[case] a: [f32; 2],
            #[case] b: [f32; 2],
            #[case] expected: f32,
        ) {
            assert_eq!(MinkowskiF64::<{ 0.5f64.to_bits() }>::dist(&a, &b), expected);
        }

        #[rstest]
        #[case([0.0f32], [0.0f32], 0.0f32)]
        #[case([0.0f32], [2.0f32], 8.0f32)]
        #[case([-2.0f32], [2.0f32], 64.0f32)] // | -2 - 2|^3 = 4^3 = 64
        fn test_minkowski_3_distance_1d(
            #[case] a: [f32; 1],
            #[case] b: [f32; 1],
            #[case] expected: f32,
        ) {
            assert_eq!(Minkowski::<3>::dist(&a, &b), expected);
        }

        #[rstest]
        #[case(0.0f32, 0.0f32, 0.0f32)]
        #[case(1.0f32, 0.0f32, 1.0f32)]
        #[case(0.0f32, 2.0f32, 8.0f32)]
        #[case(-2.5f32, 3.5f32, 216.0f32)] // | -2.5 - 3.5|^3 = 6^3 = 216
        fn test_minkowski_3_dist1(#[case] a: f32, #[case] b: f32, #[case] expected: f32) {
            assert_eq!(
                <Minkowski<3> as DistanceMetric<f32, 1>>::dist1(a, b),
                expected
            );
        }

        #[rstest]
        #[case(0.0f32, 0.0f32, 0.0f32)]
        #[case(1.0f32, 0.0f32, 1.0f32)]
        #[case(0.0f32, 4.0f32, 2.0f32)]
        #[case(10.0f32, 35.0f32, 5.0f32)] // | 10 - 35|^0.5 = 25^0.5 = 5
        fn test_minkowski_05_dist1(#[case] a: f32, #[case] b: f32, #[case] expected: f32) {
            assert_eq!(
                <MinkowskiF64<{ 0.5f64.to_bits() }> as DistanceMetric<f32, 1>>::dist1(a, b),
                expected
            );
        }
    }

    #[cfg(feature = "f16")]
    mod f16_tests {
        use super::*;
        use half::f16;

        #[test]
        fn test_manhattan_f16() {
            let a = [f16::from_f32(0.0), f16::from_f32(0.0)];
            let b = [f16::from_f32(1.0), f16::from_f32(1.0)];

            let result = Manhattan::dist(&a, &b);
            let expected = f16::from_f32(2.0);

            assert_eq!(result, expected);
        }

        #[test]
        fn test_squared_euclidean_f16() {
            let a = [f16::from_f32(0.0), f16::from_f32(0.0)];
            let b = [f16::from_f32(1.0), f16::from_f32(1.0)];

            let result = SquaredEuclidean::dist(&a, &b);
            let expected = f16::from_f32(2.0);

            assert_eq!(result, expected);
        }
    }

    mod integration_tests {
        use super::*;
        use crate::KdTree;
        use rand::prelude::*;
        use rand_distr::Normal;
        use rstest::rstest;

        #[derive(Debug, Clone, Copy)]
        enum DataScenario {
            NoTies,
            Ties,
            Gaussian,
        }

        #[derive(Debug, Clone, Copy)]
        enum TreeType {
            Mutable,
            Immutable,
        }

        impl DataScenario {
            /// Get data scenario
            ///
            /// Predefined data has input dimension (`dim`) and either
            /// with `DataScenario::NoTies` or `DataScenario::Ties`.
            ///
            /// # Parameters
            /// - `dim`: The dimensionality of the data to retrieve.
            ///   Must be a value between 1 and 4 (inclusive).
            ///
            /// # Returns
            /// - `Vec<Vec<f64>>`: A 2D vector where each inner vector represents a data point.
            fn get(&self, dim: usize) -> Vec<Vec<f64>> {
                match (self, dim) {
                    (DataScenario::NoTies, 1) => vec![
                        vec![1.0],
                        vec![2.0],
                        vec![4.0],
                        vec![7.0],
                        vec![-9.0],
                        vec![16.0],
                    ],
                    (DataScenario::NoTies, 2) => vec![
                        vec![0.0, 0.0],
                        vec![1.1, 0.1],
                        vec![2.3, 0.4],
                        vec![3.6, 0.9],
                        vec![5.0, 1.6],
                        vec![6.5, 2.5],
                    ],
                    (DataScenario::NoTies, 3) => vec![
                        vec![0.0, 0.0, 0.0],
                        vec![1.1, 0.1, 0.01],
                        vec![2.3, 0.4, 0.08],
                        vec![-3.6, -0.9, -0.27],
                        vec![5.0, 1.6, 0.64],
                        vec![6.5, 2.5, 1.25],
                    ],
                    (DataScenario::NoTies, 4) => vec![
                        vec![0.0, 0.0, 0.0, 1000.0],
                        vec![1.1, 0.1, 0.01, 1000.001],
                        vec![2.3, 0.4, 0.08, 1000.008],
                        vec![3.6, 0.9, 0.27, 1000.027],
                        vec![5.0, 1.6, 0.64, 1000.256],
                        vec![6.5, 2.5, 1.25, 1000.625],
                    ],
                    (DataScenario::Ties, 1) => vec![
                        vec![0.0],
                        vec![1.0],
                        vec![1.0],
                        vec![2.0],
                        vec![2.0],
                        vec![3.0],
                    ],
                    (DataScenario::Ties, 2) => vec![
                        vec![0.0, 0.0],
                        vec![1.0, 0.0],
                        vec![0.0, 1.0],
                        vec![-1.0, 0.0],
                        vec![0.0, -1.0],
                        vec![1.0, 1.0],
                    ],
                    (DataScenario::Ties, 3) => vec![
                        vec![0.0, 0.0, 0.0],
                        vec![1.0, 0.0, 0.0],
                        vec![0.0, 1.0, 0.0],
                        vec![0.0, 0.0, 1.0],
                        vec![-1.0, 0.0, 0.0],
                        vec![0.0, -1.0, 0.0],
                    ],
                    (DataScenario::Ties, 4) => vec![
                        vec![0.0, 0.0, 0.0, 0.0],
                        vec![1.0, 0.0, 0.0, 0.0],
                        vec![0.0, 1.0, 0.0, 0.0],
                        vec![0.0, 0.0, 1.0, 0.0],
                        vec![0.0, 0.0, 0.0, 1.0],
                        vec![-1.0, 0.0, 0.0, 0.0],
                    ],
                    (DataScenario::Gaussian, d) => {
                        let mut rng = StdRng::seed_from_u64(8757);
                        let normal = Normal::new(1.0, 10.0).unwrap();
                        let n_samples = 200;
                        let mut data = vec![vec![0.0; d]; n_samples];
                        for i in 0..n_samples {
                            for j in 0..d {
                                data[i][j] = normal.sample(&mut rng);
                            }
                        }
                        data
                    }
                    _ => panic!("Unsupported dimension {} for scenario {:?}", dim, self),
                }
            }
        }

        /// Helper function to test nearest_n queries for `D: DistanceMetric`
        ///
        /// Tests KD-tree Chebyshev distance queries across different tree types and
        /// data scenarios. This simplifies testing across different combinations.
        ///
        /// # What this function does
        /// 1. Get test data points based on a scenario (NoTies/Ties) and dimensionality
        /// 2. Builds either MutableKdTree (incremental) or ImmutableKdTree (bulk construction)
        /// 3. Performs nearest_n query with Chebyshev distance from point 0
        /// 4. Compares results against Brute-force distances,
        ///    calculated from `<D: DistanceMetric<f64, 6>>::dist`.
        ///
        /// # Choices
        /// - Fixed-size array `[f64; 6]`. For `dim<6` a subspace/padding is used for practicality
        ///
        /// # Assertions
        /// - Point 0 is always the query point (distance 0, index 0 expected first result)
        /// - NoTies scenario: checks distances and item IDs for points with unique distances
        /// - Ties scenario: checks distances (order among ties is non-deterministic)
        fn run_test_helper<D: DistanceMetric<f64, 6>>(
            dim: usize,
            tree_type: TreeType,
            scenario: DataScenario,
            n: usize,
        ) {
            let data = scenario.get(dim);
            let query_point = &data[0];

            let mut points: Vec<[f64; 6]> = Vec::with_capacity(data.len());
            for row in &data {
                let mut p = [0.0; 6];
                for (i, &val) in row.iter().enumerate() {
                    p[i] = val;
                }
                points.push(p);
            }

            let mut query_arr = [0.0; 6];
            for (i, &val) in query_point.iter().enumerate() {
                if i < 6 {
                    query_arr[i] = val;
                }
            }

            // Calculate ground truth with brute-force approach
            let mut expected: Vec<(usize, f64)> = points
                .iter()
                .enumerate()
                .map(|(i, &point)| {
                    let dist = D::dist(&query_arr, &point);
                    (i, dist)
                })
                .collect();

            expected.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            let expected_distances: Vec<f64> = expected.iter().map(|(_, d)| *d).collect();

            println!(
                "Query: {:?}, TreeType: {:?}, Scenario: {:?}, dim={}, n={}",
                query_point, tree_type, scenario, dim, n
            );

            // Query based on tree type
            let results = match tree_type {
                TreeType::Mutable => {
                    let mut tree: crate::float::kdtree::KdTree<f64, u64, 6, 128, u32> =
                        crate::float::kdtree::KdTree::new();
                    for (i, point) in points.iter().enumerate() {
                        tree.add(point, i as u64);
                    }
                    tree.nearest_n::<D>(&query_arr, n)
                }
                TreeType::Immutable => {
                    let tree: crate::immutable::float::kdtree::ImmutableKdTree<f64, u64, 6, 128> =
                        crate::immutable::float::kdtree::ImmutableKdTree::new_from_slice(&points);
                    tree.nearest_n::<D>(&query_arr, std::num::NonZero::new(n).unwrap())
                }
            };

            println!("Results (len: {}):", results.len());

            assert_eq!(results[0].item, 0, "First result should be the query point");
            assert_eq!(
                results[0].distance, 0.0,
                "First result distance should be 0.0"
            );

            for (i, result) in results.iter().enumerate() {
                assert_eq!(
                    result.distance, expected_distances[i],
                    "Distance at index {} should be {}, but was {}",
                    i, expected_distances[i], result.distance
                );
            }

            if matches!(scenario, DataScenario::NoTies) {
                for (i, result) in results.iter().enumerate() {
                    let expected_id = expected[i].0;
                    assert_eq!(
                        result.item, expected_id as u64,
                        "Result {}: item ID mismatch. Expected {}, got {}",
                        i, expected_id, result.item
                    );
                }
            }
        }

        /// Chebyshev distance nearest-neighbor query tests.
        ///
        /// Test matrix covering all combinations of mutable/immutable trees,
        /// data scenarios (with/out ties), dimensions, and neighbor query counts.
        #[rstest]
        fn test_nearest_n_chebyshev(
            #[values(TreeType::Mutable, TreeType::Immutable)] tree_type: TreeType,
            #[values(DataScenario::NoTies, DataScenario::Ties, DataScenario::Gaussian)]
            scenario: DataScenario,
            #[values(1, 2, 3, 4, 5, 6)] n: usize,
            #[values(1, 2, 3, 4)] dim: usize,
        ) {
            run_test_helper::<Chebyshev>(dim, tree_type, scenario, n);
        }

        #[rstest]
        fn test_nearest_n_squared_euclidean(
            #[values(TreeType::Mutable, TreeType::Immutable)] tree_type: TreeType,
            #[values(DataScenario::NoTies, DataScenario::Ties, DataScenario::Gaussian)]
            scenario: DataScenario,
            #[values(1, 2, 3, 4, 5, 6)] n: usize,
            #[values(1, 2, 3, 4)] dim: usize,
        ) {
            run_test_helper::<SquaredEuclidean>(dim, tree_type, scenario, n);
        }

        #[rstest]
        fn test_nearest_n_manhattan(
            #[values(TreeType::Mutable, TreeType::Immutable)] tree_type: TreeType,
            #[values(DataScenario::NoTies, DataScenario::Ties, DataScenario::Gaussian)]
            scenario: DataScenario,
            #[values(1, 2, 3, 4, 5, 6)] n: usize,
            #[values(1, 2, 3, 4)] dim: usize,
        ) {
            run_test_helper::<Manhattan>(dim, tree_type, scenario, n);
        }

        #[rstest]
        fn test_nearest_n_minkowski(
            #[values(TreeType::Mutable, TreeType::Immutable)] tree_type: TreeType,
            #[values(DataScenario::NoTies, DataScenario::Ties, DataScenario::Gaussian)]
            scenario: DataScenario,
            #[values(1, 2, 3, 4, 5, 6)] n: usize,
            #[values(1, 2, 3, 4)] dim: usize,
            #[values(3, 4)] p: u32,
        ) {
            match p {
                3 => run_test_helper::<Minkowski<3>>(dim, tree_type, scenario, n),
                4 => run_test_helper::<Minkowski<4>>(dim, tree_type, scenario, n),
                _ => unreachable!(),
            }
        }

        #[rstest]
        fn test_nearest_n_minkowski_f64(
            #[values(TreeType::Mutable, TreeType::Immutable)] tree_type: TreeType,
            #[values(DataScenario::NoTies, DataScenario::Ties, DataScenario::Gaussian)]
            scenario: DataScenario,
            #[values(1, 2, 3, 4, 5, 6)] n: usize,
            #[values(1, 2, 3, 4)] dim: usize,
            #[values(0.5, 1.5)] p: f64,
        ) {
            if (p - 0.5).abs() < f64::EPSILON {
                run_test_helper::<MinkowskiF64<{ 0.5f64.to_bits() }>>(dim, tree_type, scenario, n);
            } else if (p - 1.5).abs() < f64::EPSILON {
                run_test_helper::<MinkowskiF64<{ 1.5f64.to_bits() }>>(dim, tree_type, scenario, n);
            } else {
                unreachable!()
            }
        }

        #[test]
        fn test_nearest_n_manhattan_distance() {
            let mut kdtree: KdTree<f32, 2> = KdTree::new();

            // Add points in a simple pattern
            let points = [
                ([0.0f32, 0.0f32], 0), // distance 0 from query point
                ([1.0f32, 0.0f32], 1), // distance 1 from query point
                ([0.0f32, 1.0f32], 2), // distance 1 from query point
                ([2.0f32, 0.0f32], 3), // distance 2 from query point
                ([0.0f32, 2.0f32], 4), // distance 2 from query point
                ([3.0f32, 3.0f32], 5), // distance 6 from query point
            ];

            for (point, index) in points {
                kdtree.add(&point, index);
            }

            let query_point = [0.0f32, 0.0f32];
            let results = kdtree.nearest_n::<Manhattan>(&query_point, 4);

            // Expected order: [0], [1], [2], [3], [4]
            // Distances: 0, 1, 1, 2, 2
            // But we only ask for 4 nearest
            assert_eq!(results.len(), 4);

            // First result should be the query point itself
            assert_eq!(results[0].item, 0);
            assert_eq!(results[0].distance, 0.0);

            // Next two should be the points at Manhattan distance 1
            assert_eq!(results[1].item, 1);
            assert_eq!(results[1].distance, 1.0);
            assert_eq!(results[2].item, 2);
            assert_eq!(results[2].distance, 1.0);

            // Fourth should be one of the points at distance 2
            assert!(results[3].item == 3 || results[3].item == 4);
            assert_eq!(results[3].distance, 2.0);
        }

        #[test]
        fn test_nearest_n_squared_euclidean_distance() {
            let mut kdtree: KdTree<f64, 2> = KdTree::new();

            // Add points in a pattern where Euclidean and Manhattan differ
            let points = [
                ([0.0, 0.0], 0), // distance 0 from query point
                ([1.0, 0.0], 1), // Euclidean: 1, Manhattan: 1
                ([0.0, 1.0], 2), // Euclidean: 1, Manhattan: 1
                ([1.0, 1.0], 3), // Euclidean: 2, Manhattan: 2
                ([2.0, 0.0], 4), // Euclidean: 4, Manhattan: 2
                ([0.0, 2.0], 5), // Euclidean: 4, Manhattan: 2
                ([3.0, 4.0], 6), // Euclidean: 25, Manhattan: 7
            ];

            for (point, index) in points {
                kdtree.add(&point, index);
            }

            let query_point = [0.0, 0.0];
            let results = kdtree.nearest_n::<SquaredEuclidean>(&query_point, 5);

            assert_eq!(results.len(), 5);

            // First should be the query point itself
            assert_eq!(results[0].item, 0);
            assert_eq!(results[0].distance, 0.0);

            // Next two should be the points at Euclidean distance 1
            assert_eq!(results[1].item, 1);
            assert_eq!(results[1].distance, 1.0);
            assert_eq!(results[2].item, 2);
            assert_eq!(results[2].distance, 1.0);

            // Next two should be the points at Euclidean distance 2
            assert_eq!(results[3].item, 3);
            assert_eq!(results[3].distance, 2.0);
            assert_eq!(results[4].item, 4);
            assert_eq!(results[4].distance, 4.0);

            // Verify that points at squared Euclidean distance 4 are indeed farther
            // than points at squared Euclidean distance 2
            assert!(results[4].distance > results[3].distance);
        }

        #[test]
        fn test_nearest_n_different_metrics_produce_different_orderings() {
            let mut kdtree: KdTree<f32, 2> = KdTree::new();

            // Add points where Manhattan and Euclidean give different orderings
            let points = [
                ([0.0, 0.0], 0), // origin
                ([2.0, 1.0], 1), // Manhattan: 3, Euclidean^2: 5
                ([1.0, 2.0], 2), // Manhattan: 3, Euclidean^2: 5
                ([3.0, 0.0], 3), // Manhattan: 3, Euclidean^2: 9
                ([0.0, 3.0], 4), // Manhattan: 3, Euclidean^2: 9
            ];

            for (point, index) in points {
                kdtree.add(&point, index);
            }

            let query_point = [0.0, 0.0];

            let manhattan_results = kdtree.nearest_n::<Manhattan>(&query_point, 3);
            let euclidean_results = kdtree.nearest_n::<SquaredEuclidean>(&query_point, 3);

            // Both should include the origin as first result
            assert_eq!(manhattan_results[0].item, 0);
            assert_eq!(euclidean_results[0].item, 0);

            // For Manhattan: points 1, 2, 3, 4 are all at distance 3
            // The ordering among ties depends on tree structure, but they should all have same distance
            assert_eq!(manhattan_results[1].distance, 3.0);
            assert_eq!(manhattan_results[2].distance, 3.0);

            // For Euclidean: points 1 and 2 are at distance sqrt(5) ≈ 2.236 (squared: 5)
            // Points 3 and 4 are at distance 3 (squared: 9)
            assert_eq!(euclidean_results[1].distance, 5.0);
            assert_eq!(euclidean_results[2].distance, 5.0);

            // Verify that Euclidean ordering puts points 1 and 2 before 3 and 4
            let euclidean_items: Vec<u64> = euclidean_results
                .iter()
                .skip(1) // skip origin
                .take(2) // take next 2
                .map(|nn| nn.item)
                .collect();

            assert!(euclidean_items.contains(&1) || euclidean_items.contains(&2));

            // Calculate actual distances to verify our understanding
            let p1 = [2.0, 1.0];
            let p2 = [1.0, 2.0];
            let p3 = [3.0, 0.0];

            let manhattan_p1 = Manhattan::dist(&query_point, &p1);
            let manhattan_p2 = Manhattan::dist(&query_point, &p2);
            let manhattan_p3 = Manhattan::dist(&query_point, &p3);

            let euclidean_p1 = SquaredEuclidean::dist(&query_point, &p1);
            let euclidean_p2 = SquaredEuclidean::dist(&query_point, &p2);
            let euclidean_p3 = SquaredEuclidean::dist(&query_point, &p3);

            assert_eq!(manhattan_p1, 3.0);
            assert_eq!(manhattan_p2, 3.0);
            assert_eq!(manhattan_p3, 3.0);

            assert_eq!(euclidean_p1, 5.0);
            assert_eq!(euclidean_p2, 5.0);
            assert_eq!(euclidean_p3, 9.0);
        }

        #[test]
        fn test_nearest_n_3d_different_metrics() {
            let mut kdtree: KdTree<f64, 3> = KdTree::new();

            // Add points in 3D space
            let points = [
                ([1.0, 1.0, 1.0], 0), // origin
                ([2.0, 1.0, 1.0], 1), // 1 unit away on x-axis
                ([1.0, 2.0, 1.0], 2), // 1 unit away on y-axis
                ([1.0, 1.0, 2.0], 3), // 1 unit away on z-axis
                ([3.0, 1.0, 1.0], 4), // 2 units away on x-axis
                ([0.0, 0.0, 0.0], 5), // sqrt(3) ≈ 1.732 from origin
            ];

            for (point, index) in points {
                kdtree.add(&point, index);
            }

            let query_point = [1.0, 1.0, 1.0];
            let results = kdtree.nearest_n::<Manhattan>(&query_point, 4);

            assert_eq!(results.len(), 4);

            // First should be the query point itself
            assert_eq!(results[0].item, 0);
            assert_eq!(results[0].distance, 0.0);

            // Next three should be the points at Manhattan distance 1
            let nearby_items: Vec<u64> = results
                .iter()
                .skip(1) // skip origin
                .take(3) // take next 3
                .map(|nn| nn.item)
                .collect();

            assert!(nearby_items.contains(&1));
            assert!(nearby_items.contains(&2));
            assert!(nearby_items.contains(&3));

            // All nearby points should have distance 1
            for result in results.iter().skip(1).take(3) {
                assert_eq!(result.distance, 1.0);
            }

            // Point 4 should be farther (distance 2) and not in top 4
            let all_items: Vec<u64> = results.iter().map(|nn| nn.item).collect();
            assert!(!all_items.contains(&4));

            // Point 5 has Manhattan distance 3, so definitely not in top 4
            assert!(!all_items.contains(&5));
        }

        #[test]
        fn test_nearest_n_large_scale() {
            let mut kdtree: KdTree<f32, 2> = KdTree::new();

            // Create a grid of points
            let mut index = 0;
            for x in 0i32..10 {
                for y in 0i32..10 {
                    let point = [x as f32, y as f32];
                    kdtree.add(&point, index);
                    index += 1;
                }
            }

            // Query from center of grid
            let query_point = [5.0f32, 5.0f32];
            let results = kdtree.nearest_n::<SquaredEuclidean>(&query_point, 10);

            assert_eq!(results.len(), 10);

            // First result should be the center point itself (index 55)
            assert_eq!(results[0].item, 55);
            assert_eq!(results[0].distance, 0.0);

            // Results should be ordered by increasing distance
            for i in 1..10 {
                assert!(results[i].distance >= results[i - 1].distance);
            }

            // Verify distances make sense for a grid
            // The nearest points should be at squared distances: 0, 1, 1, 1, 1, 2, 2, 4, 4, 5...
            let expected_distances = [0.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 2.0f32];

            for (i, &expected_dist) in expected_distances.iter().enumerate() {
                if i < results.len() {
                    assert_eq!(results[i].distance, expected_dist);
                }
            }
        }

        #[test]
        fn test_nearest_n_chebyshev_distance() {
            let mut kdtree: KdTree<f32, 2> = KdTree::new();

            // Add points that show Chebyshev behavior
            let points = [
                ([0.0f32, 0.0f32], 0), // distance 0 from query point
                ([1.0f32, 0.0f32], 1), // Chebyshev: 1, Manhattan: 1, Euclidean^2: 1
                ([0.0f32, 1.0f32], 2), // Chebyshev: 1, Manhattan: 1, Euclidean^2: 1
                ([2.0f32, 0.0f32], 3), // Chebyshev: 2, Manhattan: 2, Euclidean^2: 4
                ([0.0f32, 2.0f32], 4), // Chebyshev: 2, Manhattan: 2, Euclidean^2: 4
                ([1.0f32, 1.0f32], 5), // Chebyshev: 1, Manhattan: 2, Euclidean^2: 2
            ];

            for (point, index) in points {
                kdtree.add(&point, index);
            }

            let query_point = [0.0f32, 0.0f32];
            let results = kdtree.nearest_n::<Chebyshev>(&query_point, 5);

            // With Chebyshev, points at (1,0), (0,1), and (1,1) all have distance 1
            // Points at (2,0) and (0,2) have distance 2
            assert_eq!(results.len(), 5);

            // First should be the query point itself
            assert_eq!(results[0].item, 0);
            assert_eq!(results[0].distance, 0.0);

            // Next should all be at Chebyshev distance 1
            let nearby_items: Vec<u64> = results
                .iter()
                .skip(1) // skip origin
                .take(4) // take next 4
                .filter(|r| (r.distance - 1.0).abs() < 0.001) // check for distance 1 (with some float tolerance)
                .map(|nn| nn.item)
                .collect();

            // All of these should be in the results: 1, 2, 5
            assert!(nearby_items.contains(&1));
            assert!(nearby_items.contains(&2));
            assert!(nearby_items.contains(&5));
        }

        #[test]
        fn test_within_chebyshev_distance() {
            let mut kdtree: KdTree<f32, 2> = KdTree::new();

            // Add points with varying Chebyshev distances
            let points = [
                ([0.0f32, 0.0f32], 0), // distance 0
                ([0.5f32, 0.5f32], 1), // Chebyshev: 0.5
                ([1.0f32, 0.0f32], 2), // Chebyshev: 1.0
                ([0.8f32, 0.9f32], 3), // Chebyshev: 0.9
                ([2.0f32, 0.0f32], 4), // Chebyshev: 2.0
                ([0.0f32, 2.0f32], 5), // Chebyshev: 2.0
                ([1.5f32, 1.5f32], 6), // Chebyshev: 1.5
            ];

            for (point, index) in points {
                kdtree.add(&point, index);
            }

            let query_point = [0.0f32, 0.0f32];
            let radius = 1.0; // radius 1 (not squared for Chebyshev)
            let mut results = kdtree.within::<Chebyshev>(&query_point, radius);

            // Sort by distance for easier verification
            results.sort_by(|a, b| {
                a.distance
                    .partial_cmp(&b.distance)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // These SHOULD be: 0, 1, 2, 3 (distances: 0, 0.5, 1.0, 0.9)
            // For `<=5.2.4` found: 0, 1, 3 (index 2 is missing due to dist1 pruning issue)
            let found_indices: Vec<u64> = results.iter().map(|r| r.item).collect();

            assert!(found_indices.contains(&0));
            assert!(found_indices.contains(&1));
            assert!(found_indices.contains(&2));
            assert!(found_indices.contains(&3));
            // Should NOT include points with Chebyshev distance > 1
            assert!(!found_indices.contains(&4));
            assert!(!found_indices.contains(&5));
            assert!(!found_indices.contains(&6));

            // Verify distances
            for result in results {
                assert!(result.distance <= 1.0 || (result.distance - 1.0).abs() < 0.001);
            }
        }

        #[test]
        fn test_chebyshev_vs_manhattan_ordering() {
            let mut kdtree: KdTree<f32, 2> = KdTree::new();

            // Points where Chebyshev and Manhattan differ significantly
            let points = [
                ([0.0f32, 0.0f32], 0), // origin
                ([3.0f32, 1.0f32], 1), // Chebyshev: 3, Manhattan: 4
                ([1.0f32, 3.0f32], 2), // Chebyshev: 3, Manhattan: 4
                ([2.0f32, 2.0f32], 3), // Chebyshev: 2, Manhattan: 4
                ([4.0f32, 0.5f32], 4), // Chebyshev: 4, Manhattan: 4.5
            ];

            for (point, index) in points {
                kdtree.add(&point, index);
            }

            let query_point = [0.0f32, 0.0f32];

            let chebyshev_results = kdtree.nearest_n::<Chebyshev>(&query_point, 4);
            let manhattan_results = kdtree.nearest_n::<Manhattan>(&query_point, 4);

            // Both should include the origin first
            assert_eq!(chebyshev_results[0].item, 0);
            assert_eq!(manhattan_results[0].item, 0);

            // With Chebyshev, nearest should be point 3 (distance 2)
            // With Manhattan, nearest should be points 1 and 2 (distance 4)
            assert_eq!(chebyshev_results[1].item, 3);
            assert_eq!(chebyshev_results[1].distance, 2.0);

            // With Manhattan, points 1 and 2 should come before point 3 (which is distance 4)
            let manhattan_items: Vec<u64> = manhattan_results
                .iter()
                .skip(1)
                .take(3)
                .map(|r| r.item)
                .collect();
            assert!(manhattan_items.contains(&1) || manhattan_items.contains(&2));

            // Verify the distance calculations are correct
            assert_eq!(chebyshev_results[1].distance, 2.0); // Chebyshev: max(|2-0|, |2-0|) = 2
            assert_eq!(manhattan_results[1].distance, 4.0); // Manhattan: |3-0| + |1-0| = 4
        }
    }
}
