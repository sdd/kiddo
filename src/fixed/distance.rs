//! Defines different distance metrics, in simplest case it defines the
//! euclidean distance which is no more than the square root of the sum of the
//! squares of the distances in each dimension.

// #[cfg(any(target_arch = "x86_64"))]
// use std::arch::x86_64::*;

use crate::fixed::kdtree::Axis;
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
/// use kiddo::fixed::distance::Manhattan;
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

    #[inline]
    fn accumulate(rd: A, delta: A) -> A {
        rd.saturating_add(delta)
    }
}

/// Returns the Chebyshev distance (L-infinity norm) between two points.
///
/// This is the maximum of the absolute differences between coordinates of points.
///
/// # Examples
///
/// ```rust
/// use fixed::types::extra::U0;
/// use fixed::FixedU16;
/// use kiddo::traits::DistanceMetric;
/// use kiddo::fixed::distance::Chebyshev;
/// type Fxd = FixedU16<U0>;
///
/// let ZERO = Fxd::from_num(0);
/// let ONE = Fxd::from_num(1);
/// let TWO = Fxd::from_num(2);
///
/// assert_eq!(ZERO, Chebyshev::dist(&[ZERO, ZERO], &[ZERO, ZERO]));
/// assert_eq!(ONE, Chebyshev::dist(&[ZERO, ZERO], &[ONE, ZERO]));
/// assert_eq!(ONE, Chebyshev::dist(&[ZERO, ZERO], &[ONE, ONE]));
/// assert_eq!(TWO, Chebyshev::dist(&[ZERO, ZERO], &[TWO, ONE]));
/// ```
pub struct Chebyshev {}

impl<A: Axis, const K: usize> DistanceMetric<A, K> for Chebyshev {
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
            .reduce(|a, b| if a > b { a } else { b })
            .unwrap_or(A::ZERO)
    }

    #[inline]
    fn dist1(a: A, b: A) -> A {
        if a > b {
            a - b
        } else {
            b - a
        }
    }

    #[inline]
    fn accumulate(rd: A, delta: A) -> A {
        if rd > delta {
            rd
        } else {
            delta
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
/// use kiddo::fixed::distance::SquaredEuclidean;
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

    #[inline]
    fn accumulate(rd: A, delta: A) -> A {
        rd.saturating_add(delta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fixed::types::extra::U0;
    use rstest::rstest;

    type FxdU16 = fixed::FixedU16<U0>;

    const ZERO: FxdU16 = FxdU16::ZERO;
    const ONE: FxdU16 = FxdU16::lit("1");
    const TWO: FxdU16 = FxdU16::lit("2");
    const THREE: FxdU16 = FxdU16::lit("3");
    const FOUR: FxdU16 = FxdU16::lit("4");
    const FIVE: FxdU16 = FxdU16::lit("5");

    #[rstest]
    #[case([ZERO, ZERO], [ZERO, ZERO], ZERO)]
    #[case([ZERO, ZERO], [ONE, ZERO], ONE)]
    #[case([ZERO, ZERO], [ZERO, ONE], ONE)]
    #[case([ZERO, ZERO], [ONE, ONE], ONE)]
    #[case([ZERO, ZERO], [TWO, ONE], TWO)]
    #[case([ZERO, ZERO], [ONE, TWO], TWO)]
    fn test_chebyshev_distance_2d(
        #[case] a: [FxdU16; 2],
        #[case] b: [FxdU16; 2],
        #[case] expected: FxdU16,
    ) {
        assert_eq!(Chebyshev::dist(&a, &b), expected);
    }

    #[rstest]
    #[case([ZERO, ZERO, ZERO], [ZERO, ZERO, ZERO], ZERO)]
    #[case([ZERO, ZERO, ZERO], [ONE, TWO, THREE], THREE)]
    #[case([FIVE, FIVE, FIVE], [ONE, TWO, THREE], FOUR)]
    fn test_chebyshev_distance_3d(
        #[case] a: [FxdU16; 3],
        #[case] b: [FxdU16; 3],
        #[case] expected: FxdU16,
    ) {
        assert_eq!(Chebyshev::dist(&a, &b), expected);
    }

    #[rstest]
    #[case([ZERO, ZERO], [ZERO, ZERO], ZERO)]
    #[case([ZERO, ZERO], [ONE, ZERO], ONE)]
    #[case([ZERO, ZERO], [ZERO, ONE], ONE)]
    #[case([ZERO, ZERO], [ONE, ONE], TWO)]
    #[case([TWO, THREE], [ONE, ONE], THREE)]
    fn test_manhattan_distance_2d(
        #[case] a: [FxdU16; 2],
        #[case] b: [FxdU16; 2],
        #[case] expected: FxdU16,
    ) {
        assert_eq!(Manhattan::dist(&a, &b), expected);
    }

    #[rstest]
    #[case([ZERO, ZERO], [ZERO, ZERO], ZERO)]
    #[case([ZERO, ZERO], [ONE, ZERO], ONE)]
    #[case([ZERO, ZERO], [ZERO, ONE], ONE)]
    #[case([ZERO, ZERO], [ONE, ONE], TWO)]
    #[case([TWO, TWO], [ZERO, ZERO], FxdU16::lit("8"))]
    #[case([ONE, TWO], [TWO, ONE], TWO)]
    fn test_squared_euclidean_distance_2d(
        #[case] a: [FxdU16; 2],
        #[case] b: [FxdU16; 2],
        #[case] expected: FxdU16,
    ) {
        assert_eq!(SquaredEuclidean::dist(&a, &b), expected);
    }

    #[rstest]
    #[case([ZERO, ZERO, ZERO], [ZERO, ZERO, ZERO], ZERO)]
    #[case([ZERO, ZERO, ZERO], [ONE, ZERO, ZERO], ONE)]
    #[case([ONE, ONE, ONE], [TWO, TWO, TWO], THREE)]
    fn test_squared_euclidean_distance_3d(
        #[case] a: [FxdU16; 3],
        #[case] b: [FxdU16; 3],
        #[case] expected: FxdU16,
    ) {
        assert_eq!(SquaredEuclidean::dist(&a, &b), expected);
    }

    #[rstest]
    #[case::zero(ZERO, ZERO, ZERO)]
    #[case::pos(ONE, ZERO, ONE)]
    #[case::neg(ZERO, ONE, ONE)]
    #[case::diff(THREE, ONE, TWO)]
    fn test_manhattan_dist1(#[case] a: FxdU16, #[case] b: FxdU16, #[case] expected: FxdU16) {
        assert_eq!(
            <Manhattan as DistanceMetric<FxdU16, 1>>::dist1(a, b),
            expected
        );
    }

    #[rstest]
    #[case::zero(ZERO, ZERO, ZERO)]
    #[case::pos(ONE, ZERO, ONE)]
    #[case::neg(ZERO, ONE, ONE)]
    #[case::a_larger(TWO, ONE, ONE)]
    #[case::b_larger(ONE, TWO, ONE)]
    fn test_chebyshev_dist1(#[case] a: FxdU16, #[case] b: FxdU16, #[case] expected: FxdU16) {
        assert_eq!(
            <Chebyshev as DistanceMetric<FxdU16, 1>>::dist1(a, b),
            expected
        );
    }

    #[rstest]
    #[case::zero(ZERO, ZERO, ZERO)]
    #[case::pos(ONE, ZERO, ONE)]
    #[case::neg(ZERO, ONE, ONE)]
    #[case::a_larger(TWO, ONE, ONE)]
    #[case::b_larger(ONE, TWO, ONE)]
    fn test_squared_euclidean_dist1(
        #[case] a: FxdU16,
        #[case] b: FxdU16,
        #[case] expected: FxdU16,
    ) {
        assert_eq!(
            <SquaredEuclidean as DistanceMetric<FxdU16, 1>>::dist1(a, b),
            expected
        );
    }

    #[rstest]
    #[case::zero_one(ZERO, ONE, ONE)]
    #[case::one_zero(ONE, ZERO, ONE)]
    #[case::first_larger(ONE, TWO, TWO)]
    #[case::second_larger(TWO, ONE, TWO)]
    fn test_chebyshev_accumulate(
        #[case] rd: FxdU16,
        #[case] delta: FxdU16,
        #[case] expected: FxdU16,
    ) {
        assert_eq!(
            <Chebyshev as DistanceMetric<FxdU16, 1>>::accumulate(rd, delta),
            expected
        );
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::fixed::kdtree::KdTree;
    use fixed::types::extra::U0;
    use fixed::FixedU16;
    use rstest::rstest;

    type FxdU16 = FixedU16<U0>;

    const ZERO: FxdU16 = FxdU16::ZERO;
    const ONE: FxdU16 = FxdU16::lit("1");
    const TWO: FxdU16 = FxdU16::lit("2");
    const THREE: FxdU16 = FxdU16::lit("3");
    const FOUR: FxdU16 = FxdU16::lit("4");
    const FIVE: FxdU16 = FxdU16::lit("5");

    enum DataScenario {
        NoTies,
        Ties,
    }

    impl DataScenario {
        fn get(&self, dim: usize) -> Vec<Vec<FxdU16>> {
            match (self, dim) {
                (DataScenario::NoTies, 1) => {
                    vec![vec![ONE], vec![TWO], vec![THREE], vec![FOUR], vec![FIVE]]
                }
                (DataScenario::NoTies, 2) => vec![
                    vec![ZERO, ZERO],
                    vec![ONE, ZERO],
                    vec![TWO, ZERO],
                    vec![THREE, ZERO],
                    vec![FOUR, ZERO],
                    vec![FIVE, ZERO],
                ],
                (DataScenario::NoTies, 3) => vec![
                    vec![ZERO, ZERO, ZERO],
                    vec![ONE, ZERO, ZERO],
                    vec![TWO, ZERO, ZERO],
                    vec![THREE, ZERO, ZERO],
                    vec![FOUR, ZERO, ZERO],
                    vec![FIVE, ZERO, ZERO],
                ],
                (DataScenario::Ties, 1) => vec![
                    vec![ZERO],
                    vec![ONE],
                    vec![ONE],
                    vec![TWO],
                    vec![THREE],
                    vec![THREE],
                ],
                (DataScenario::Ties, 2) => vec![
                    vec![ZERO, ZERO],
                    vec![ONE, ZERO],
                    vec![ZERO, ONE],
                    vec![TWO, ZERO],
                    vec![ZERO, TWO],
                    vec![TWO, TWO],
                ],
                (DataScenario::Ties, 3) => vec![
                    vec![ZERO, ZERO, ZERO],
                    vec![ONE, ZERO, ZERO],
                    vec![ZERO, ONE, ZERO],
                    vec![ZERO, ZERO, ONE],
                    vec![TWO, ZERO, ZERO],
                    vec![ZERO, TWO, ZERO],
                ],
                _ => panic!("Unsupported dimension"),
            }
        }
    }

    fn run_test_helper<D: DistanceMetric<FxdU16, 6>>(dim: usize, scenario: DataScenario, n: usize) {
        let data = scenario.get(dim);
        let query_point = &data[0];

        let mut points: Vec<[FxdU16; 6]> = Vec::with_capacity(data.len());
        for row in &data {
            let mut p = [ZERO; 6];
            for (i, &val) in row.iter().enumerate() {
                if i < 6 {
                    p[i] = val;
                }
            }
            points.push(p);
        }

        let mut query_arr = [ZERO; 6];
        for (i, &val) in query_point.iter().enumerate() {
            if i < 6 {
                query_arr[i] = val;
            }
        }

        let expected: Vec<(usize, FxdU16)> = points
            .iter()
            .enumerate()
            .map(|(i, &point)| {
                let dist = D::dist(&query_arr, &point);
                (i, dist)
            })
            .collect();

        let expected_distances: Vec<FxdU16> = expected.iter().map(|(_, d)| *d).collect();

        let mut tree: KdTree<FxdU16, u32, 6, 32, u32> = KdTree::new();
        for (i, point) in points.iter().enumerate() {
            tree.add(point, i as u32);
        }

        let results = tree.nearest_n::<D>(&query_arr, n);

        assert_eq!(results[0].item, 0, "First result should be the query point");
        assert_eq!(
            results[0].distance, ZERO,
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
                    result.item, expected_id as u32,
                    "Result {}: item ID mismatch. Expected {}, got {}",
                    i, expected_id, result.item
                );
            }
        }
    }

    #[rstest]
    fn test_nearest_n_chebyshev(
        #[values(DataScenario::NoTies, DataScenario::Ties)] scenario: DataScenario,
        #[values(1, 2, 3, 4, 5, 6)] n: usize,
        #[values(1, 2, 3)] dim: usize,
    ) {
        run_test_helper::<Chebyshev>(dim, scenario, n);
    }

    #[rstest]
    fn test_nearest_n_squared_euclidean(
        #[values(DataScenario::NoTies, DataScenario::Ties)] scenario: DataScenario,
        #[values(1, 2, 3, 4, 5, 6)] n: usize,
        #[values(1, 2, 3)] dim: usize,
    ) {
        run_test_helper::<SquaredEuclidean>(dim, scenario, n);
    }

    #[rstest]
    fn test_nearest_n_manhattan(
        #[values(DataScenario::NoTies, DataScenario::Ties)] scenario: DataScenario,
        #[values(1, 2, 3, 4, 5, 6)] n: usize,
        #[values(1, 2, 3)] dim: usize,
    ) {
        run_test_helper::<Manhattan>(dim, scenario, n);
    }
}
