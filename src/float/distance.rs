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
}

/// Returns the squared euclidean distance between two points.
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

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

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

        #[test]
        fn test_manhattan_symmetry() {
            let a = [1.0f64, 2.0f64, 3.0f64];
            let b = [4.0f64, 5.0f64, 6.0f64];

            assert_eq!(Manhattan::dist(&a, &b), Manhattan::dist(&b, &a));
        }

        #[test]
        fn test_manhattan_identity() {
            let a = [1.0f32, 2.0f32, 3.0f32];
            assert_eq!(Manhattan::dist(&a, &a), 0.0f32);
        }

        #[test]
        fn test_manhattan_non_negativity() {
            let a = [1.0f32, 2.0f32];
            let b = [3.0f32, 4.0f32];
            let distance = Manhattan::dist(&a, &b);
            assert!(distance >= 0.0f32);
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
        fn test_squared_euclidean_symmetry() {
            let a = [1.0f64, 2.0f64, 3.0f64];
            let b = [4.0f64, 5.0f64, 6.0f64];

            assert_eq!(
                SquaredEuclidean::dist(&a, &b),
                SquaredEuclidean::dist(&b, &a)
            );
        }

        #[test]
        fn test_squared_euclidean_identity() {
            let a = [1.0f32, 2.0f32, 3.0f32];
            assert_eq!(SquaredEuclidean::dist(&a, &a), 0.0f32);
        }

        #[test]
        fn test_squared_euclidean_non_negativity() {
            let a = [1.0f32, 2.0f32];
            let b = [3.0f32, 4.0f32];
            let distance = SquaredEuclidean::dist(&a, &b);
            assert!(distance >= 0.0f32);
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
}
