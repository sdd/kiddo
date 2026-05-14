use std::cmp::Ordering;

use crate::Axis;

/// Core distance metric behavior independent of architecture-specific SIMD.
///
/// `A` is the coordinate type stored in the tree/query.
/// `Output` is the widened accumulator / distance scalar type.
///
/// Dimensionality is method-generic (`const K: usize`) so callers do not need
/// to carry `K` on the metric type itself.
pub trait DistanceMetricCore<A: Copy> {
    /// Accumulator / distance scalar type.
    type Output: Axis<Coord = Self::Output>;

    /// Desired ordering semantics for this metric.
    /// - `Less`: smaller value is better (distance metrics)
    /// - `Greater`: larger value is better (similarity metrics)
    const ORDERING: Ordering;

    /// Widen a coordinate to the output type.
    fn widen_coord(a: A) -> Self::Output;

    /// Bulk widen hook.
    ///
    /// Default is a scalar loop. Implementers may override.
    #[inline(always)]
    fn widen_axis(axis: &[A], out: &mut [Self::Output]) {
        assert!(out.len() >= axis.len());
        for (dst, &src) in out.iter_mut().zip(axis.iter()) {
            *dst = Self::widen_coord(src);
        }
    }

    /// Single-axis contribution in widened coordinates.
    fn dist1(a: Self::Output, b: Self::Output) -> Self::Output;

    /// Combine a per-axis contribution into an accumulated point or box distance.
    ///
    /// Additive metrics keep the default `+` behavior; metrics such as
    /// Chebyshev override this to use `max`.
    #[inline(always)]
    fn combine_component(acc: &mut Self::Output, component: Self::Output) {
        *acc += component;
    }

    /// Single-axis contribution on raw coordinates.
    #[inline(always)]
    fn dist1_raw(a: A, b: A) -> Self::Output {
        Self::dist1(Self::widen_coord(a), Self::widen_coord(b))
    }

    /// Full point distance in widened coordinates.
    #[inline(always)]
    fn dist<const K: usize>(a: &[Self::Output; K], b: &[Self::Output; K]) -> Self::Output {
        let mut acc = Self::Output::zero();

        for dim in 0..K {
            Self::combine_component(&mut acc, Self::dist1(a[dim], b[dim]));
        }

        acc
    }

    /// Full point distance on raw coordinates.
    #[inline(always)]
    fn dist_raw<const K: usize>(a: &[A; K], b: &[A; K]) -> Self::Output {
        let mut acc = Self::Output::zero();

        for dim in 0..K {
            Self::combine_component(&mut acc, Self::dist1_raw(a[dim], b[dim]));
        }

        acc
    }

    /// Bounding-box distance derived from per-axis offsets to the query.
    #[inline(always)]
    fn rect_dist_from_off<const K: usize>(off: &[Self::Output; K]) -> Self::Output {
        let mut acc = Self::Output::zero();

        for off_val in off.iter().copied() {
            Self::combine_component(&mut acc, Self::dist1(off_val, Self::Output::zero()));
        }

        acc
    }

    /// Bounding-box distance after replacing a single axis offset.
    ///
    /// Additive metrics can update in O(1); metrics with different aggregation
    /// semantics can override.
    #[inline(always)]
    fn rect_dist_after_update<const K: usize>(
        rd: Self::Output,
        off: &[Self::Output; K],
        dim: usize,
        new_off: Self::Output,
    ) -> Self::Output {
        let new_dist1 = Self::dist1(new_off, Self::Output::zero());
        let old_dist1 = Self::dist1(off[dim], Self::Output::zero());
        Self::Output::saturating_add(rd - old_dist1, new_dist1)
    }

    /// Returns true if `a` is better than `b` under this metric ordering.
    #[inline(always)]
    fn better(a: Self::Output, b: Self::Output) -> bool {
        match Self::ORDERING {
            Ordering::Less => a < b,
            Ordering::Greater => a > b,
            Ordering::Equal => false,
        }
    }

    /// Ordering-compatible comparison helper.
    #[inline(always)]
    fn cmp(a: Self::Output, b: Self::Output) -> Ordering {
        a.partial_cmp(&b).unwrap_or(Ordering::Equal)
    }
}

#[cfg(test)]
mod tests {
    use super::DistanceMetricCore;
    use std::cmp::Ordering;

    struct DummyLessMetric;
    struct DummyGreaterMetric;

    impl DistanceMetricCore<i16> for DummyLessMetric {
        type Output = f64;
        const ORDERING: Ordering = Ordering::Less;

        fn widen_coord(a: i16) -> Self::Output {
            a as f64
        }

        fn dist1(a: Self::Output, b: Self::Output) -> Self::Output {
            (a - b).abs()
        }
    }

    impl DistanceMetricCore<i16> for DummyGreaterMetric {
        type Output = f64;
        const ORDERING: Ordering = Ordering::Greater;

        fn widen_coord(a: i16) -> Self::Output {
            a as f64
        }

        fn dist1(a: Self::Output, b: Self::Output) -> Self::Output {
            a - b
        }
    }

    #[test]
    fn default_widen_axis_bulk_widens() {
        let axis = [1i16, -2, 7];
        let mut out = [0.0f64; 3];
        DummyLessMetric::widen_axis(&axis, &mut out);
        assert_eq!(out, [1.0, -2.0, 7.0]);
    }

    #[test]
    fn default_better_respects_ordering() {
        assert!(DummyLessMetric::better(2.0, 5.0));
        assert!(!DummyLessMetric::better(5.0, 2.0));
        assert!(DummyGreaterMetric::better(5.0, 2.0));
        assert!(!DummyGreaterMetric::better(2.0, 5.0));
    }

    #[test]
    fn default_cmp_uses_partial_cmp() {
        assert_eq!(DummyLessMetric::cmp(2.0, 5.0), Ordering::Less);
        assert_eq!(DummyLessMetric::cmp(5.0, 2.0), Ordering::Greater);
        assert_eq!(DummyLessMetric::cmp(3.0, 3.0), Ordering::Equal);
    }
}
