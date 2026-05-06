use crate::Axis;

/// Trait for distance metrics used in spatial queries.
///
/// This trait supports both standard distance metrics (e.g., squared Euclidean)
/// and similarity metrics (e.g., dot product) through the ORDERING associated constant.
pub trait DistanceMetric<A: Copy, const K: usize> {
    /// Accumulator / distance scalar type.
    type Output: Axis<Coord = Self::Output>;

    /// Desired sort order on distances:
    /// - Less    => smaller value means closer distance (standard distances)
    /// - Greater => larger value means closer distance (e.g. dot product)
    const ORDERING: std::cmp::Ordering;

    // ---- Widening primitives ----

    /// Widen a single input coordinate into the Output type.
    fn widen_coord(a: A) -> Self::Output;

    /// Optional bulk-widen hook for a whole axis slice.
    /// Default is scalar loop; concrete impls can override for SIMD.
    fn widen_axis(axis: &[A], out: &mut [Self::Output]) {
        assert!(out.len() >= axis.len());
        for (dst, &src) in out.iter_mut().zip(axis.iter()) {
            *dst = Self::widen_coord(src);
        }
    }

    // ---- Core primitives on widened coords ----

    /// Distance contribution along a single axis, on already-widened coords.
    fn dist1(a: Self::Output, b: Self::Output) -> Self::Output;

    /// Distance between two K-d points, on widened coords.
    fn dist(a: &[Self::Output; K], b: &[Self::Output; K]) -> Self::Output {
        let mut acc = Self::Output::zero();
        for dim in 0..K {
            acc += Self::dist1(a[dim], b[dim]);
        }
        acc
    }

    /// Returns true if `a` is closer than `b` according to the metric's ordering.
    #[inline(always)]
    fn closer(a: Self::Output, b: Self::Output) -> bool {
        match Self::ORDERING {
            std::cmp::Ordering::Less => a < b,    // smaller is closer
            std::cmp::Ordering::Greater => a > b, // larger is closer (dot product)
            std::cmp::Ordering::Equal => false,
        }
    }

    /// Compares two distance values according to the metric's ordering.
    #[inline(always)]
    fn cmp(a: Self::Output, b: Self::Output) -> std::cmp::Ordering {
        a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Helper macro to implement SIMD block support (CompareBlock and SimdPrune) for a specific block size.
///
/// This is an internal macro used by impl_axis_float and impl_axis_fixed.
#[allow(unused_macros)]
macro_rules! impl_simd_block_support {
    ($t:ty, 3, $prune_fn:path, $compare_fn:path) => {
        impl crate::stem_strategy::CompareBlock3 for $t {
            #[inline(always)]
            fn compare_block3_impl(
                stems_ptr: std::ptr::NonNull<u8>,
                query_val: Self,
                block_base_idx: usize,
            ) -> u8 {
                $compare_fn(stems_ptr, block_base_idx, query_val)
            }
        }
    };

    ($t:ty, 4, $prune_fn:path, $compare_fn:path) => {
        impl crate::stem_strategy::CompareBlock4 for $t {
            #[inline(always)]
            fn compare_block4_impl(
                stems_ptr: std::ptr::NonNull<u8>,
                query_val: Self,
                block_base_idx: usize,
            ) -> u8 {
                $compare_fn(stems_ptr, block_base_idx, query_val)
            }
        }
    };

    ($t:ty, 5, $prune_fn:path, $compare_fn:path) => {
        compile_error!("Block5 support is not yet implemented");
    };
}

/// Calculates squared Euclidean distances for a batch of 64 points.
///
/// Used for benchmarking autovectorization with concrete types.
#[inline]
pub fn calc_dists(content_points: &[[f32; 64]; 3], acc: &mut [f32; 64], query: &[f32; 3]) {
    // AVX512: 4 loops of 32 iterations, each 4x unrolled, 5 instructions per pre-unrolled iteration
    (0..3).for_each(|dim| {
        (0..64).for_each(|idx| {
            acc[idx] +=
                (content_points[dim][idx] - query[dim]) * (content_points[dim][idx] - query[dim]);
        });
    });
}

#[cfg(test)]
mod tests {
    use super::{calc_dists, DistanceMetric};
    use std::cmp::Ordering;

    struct DummyLess;
    struct DummyGreater;

    impl DistanceMetric<i16, 3> for DummyLess {
        type Output = f64;
        const ORDERING: Ordering = Ordering::Less;

        fn widen_coord(a: i16) -> Self::Output {
            a as f64
        }

        fn dist1(a: Self::Output, b: Self::Output) -> Self::Output {
            (a - b).abs()
        }
    }

    impl DistanceMetric<i16, 3> for DummyGreater {
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
    fn default_widen_axis_dist_closer_and_cmp_are_exercised() {
        let axis = [1i16, -2, 7];
        let mut out = [0.0f64; 3];
        DummyLess::widen_axis(&axis, &mut out);
        assert_eq!(out, [1.0, -2.0, 7.0]);

        let a = [1.0f64, 2.0, 3.0];
        let b = [4.0f64, 4.0, 9.0];
        assert_eq!(DummyLess::dist(&a, &b), 11.0);

        assert!(DummyLess::closer(2.0, 5.0));
        assert!(!DummyLess::closer(5.0, 2.0));
        assert!(DummyGreater::closer(5.0, 2.0));
        assert_eq!(DummyLess::cmp(2.0, 5.0), Ordering::Less);
        assert_eq!(DummyLess::cmp(5.0, 2.0), Ordering::Greater);
        assert_eq!(DummyLess::cmp(3.0, 3.0), Ordering::Equal);
    }

    #[test]
    fn calc_dists_accumulates_squared_euclidean_components() {
        let mut content_points = [[0.0f32; 64]; 3];
        for idx in 0..64 {
            content_points[0][idx] = idx as f32;
            content_points[1][idx] = idx as f32 + 1.0;
            content_points[2][idx] = idx as f32 + 2.0;
        }

        let mut acc = [0.0f32; 64];
        let query = [1.0f32, 2.0, 3.0];
        calc_dists(&content_points, &mut acc, &query);

        assert_eq!(acc[0], 3.0);
        assert_eq!(acc[1], 0.0);
        assert_eq!(acc[2], 3.0);
        assert_eq!(acc[5], 48.0);
    }
}
