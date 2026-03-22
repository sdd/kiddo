use core::cmp::Ordering;

use crate::traits_unified_2::AxisUnified;

/// Core distance metric behavior independent of architecture-specific SIMD.
///
/// `A` is the coordinate type stored in the tree/query.
/// `Output` is the widened accumulator / distance scalar type.
///
/// Dimensionality is method-generic (`const K: usize`) so callers do not need
/// to carry `K` on the metric type itself.
pub trait DistanceMetricCore<A: Copy> {
    /// Accumulator / distance scalar type.
    type Output: AxisUnified<Coord = Self::Output>;

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

    /// Full point distance in widened coordinates.
    #[inline(always)]
    fn dist<const K: usize>(a: &[Self::Output; K], b: &[Self::Output; K]) -> Self::Output {
        let mut acc = Self::dist1(a[0], b[0]);

        for dim in 1..K {
            acc += Self::dist1(a[dim], b[dim]);
        }

        acc
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
