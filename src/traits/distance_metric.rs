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
