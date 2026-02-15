//! Definitions and implementations for some traits that are used by KdTree, LeafStrategies, StemStrategies and DistanceMEtrics

use crate::kd_tree::leaf_view::LeafView;
use crate::kd_tree::KdTree;
use crate::StemStrategy;
use fixed::traits::LossyFrom;
use fixed::types::extra::{U0, U16, U8};
use fixed::{FixedI32, FixedU16};
use nonmax::NonMaxUsize;
use ordered_float::Float;
use std::fmt::{Debug, Display};
use std::ops::{AddAssign, Sub};

/// Basic type requirements for items stored in the tree.
pub trait Basics: Copy + Debug + Default + Send + Sync + 'static {}
impl<T> Basics for T where T: Copy + Debug + Default + Send + Sync + 'static {}

mod sealed {
    pub trait Sealed {}
}

/// Marker trait indicating whether a leaf strategy supports mutation.
///
/// This trait is used to enable type-level distinction between mutable and
/// immutable leaf strategies, allowing for optimized monomorphization.
pub(crate) trait Mutability: sealed::Sealed + 'static {
    /// Returns true if this is a mutable strategy
    fn is_mutable() -> bool;

    /// Creates the appropriate StemLeafResolution for this mutability type
    fn initial_stem_leaf_resolution<SS: StemStrategy>(
        stems_depth: usize,
        leaf_count: usize,
    ) -> crate::kd_tree::StemLeafResolution;

    fn get_leaf_idx<A, T, SS, LS, const K: usize, const B: usize>(
        tree: &KdTree<A, T, SS, LS, K, B>,
        query: &[A; K],
    ) -> usize
    where
        A: AxisUnified<Coord = A>,
        T: Basics + Copy + Default + PartialOrd + PartialEq,
        SS: StemStrategy,
        LS: LeafStrategy<A, T, SS, K, B>;

    fn resolve_terminal_stem_idx<A, T, SS, LS, const K: usize, const B: usize>(
        tree: &KdTree<A, T, SS, LS, K, B>,
        stem_idx: usize,
    ) -> Option<usize>
    where
        A: AxisUnified<Coord = A>,
        T: Basics + Copy + Default + PartialOrd + PartialEq,
        SS: StemStrategy,
        LS: LeafStrategy<A, T, SS, K, B>;
}

fn build_mapped_stem_leaf_resolution<SS: StemStrategy>(
    stems_depth: usize,
    leaf_count: usize,
) -> crate::kd_tree::StemLeafResolution {
    if leaf_count == 0 {
        return crate::kd_tree::StemLeafResolution::Mapped {
            min_stem_leaf_idx: 0,
            leaf_idx_map: Vec::new(),
        };
    }

    let min_stem_leaf_idx = 0;

    // Determine highest stem index that can resolve to a leaf at this depth.
    let mut stem_strategy = SS::new_no_ptr();
    for bit_idx in (0..stems_depth).rev() {
        let is_right = (leaf_count - 1) & (1 << bit_idx) != 0;
        stem_strategy.traverse(is_right);
    }

    let mut leaf_idx_map: Vec<Option<NonMaxUsize>> = vec![None; stem_strategy.stem_idx() + 1];

    // Map each leaf index to the traversal endpoint that would resolve it.
    for leaf_idx in 0..leaf_count {
        let mut stem_strategy = SS::new_no_ptr();
        for bit_idx in (0..stems_depth).rev() {
            let is_right = leaf_idx & (1 << bit_idx) != 0;
            stem_strategy.traverse(is_right);
        }
        if let Some(existing_leaf_idx) = leaf_idx_map[stem_strategy.stem_idx()] {
            panic!(
                "Duplicate terminal stem index in initial mapped leaf_idx_map construction: stem_idx={} existing_leaf_idx={} new_leaf_idx={}",
                stem_strategy.stem_idx(),
                existing_leaf_idx.get(),
                leaf_idx
            );
        }
        leaf_idx_map[stem_strategy.stem_idx()] =
            Some(NonMaxUsize::new(leaf_idx).expect("leaf_idx overflow"));
    }

    crate::kd_tree::StemLeafResolution::Mapped {
        min_stem_leaf_idx,
        leaf_idx_map,
    }
}

/// Marker type for immutable leaf strategies.
///
/// Immutable strategies never mutate the tree structure after construction,
/// allowing for simpler and faster traversal logic.
#[derive(Debug, Clone, Copy)]
pub struct Immutable;
impl sealed::Sealed for Immutable {}
impl Mutability for Immutable {
    fn is_mutable() -> bool {
        false
    }

    fn initial_stem_leaf_resolution<SS: StemStrategy>(
        stems_depth: usize,
        leaf_count: usize,
    ) -> crate::kd_tree::StemLeafResolution {
        crate::kd_tree::StemLeafResolution::Arithmetic {
            stems_depth,
            leaf_count,
        }
    }

    fn get_leaf_idx<A, T, SS, LS, const K: usize, const B: usize>(
        tree: &KdTree<A, T, SS, LS, K, B>,
        query: &[A; K],
    ) -> usize
    where
        A: AxisUnified<Coord = A>,
        T: Basics + Copy + Default + PartialOrd + PartialEq,
        SS: StemStrategy,
        LS: LeafStrategy<A, T, SS, K, B>,
    {
        tree.get_leaf_idx_unmapped(query)
    }

    fn resolve_terminal_stem_idx<A, T, SS, LS, const K: usize, const B: usize>(
        _tree: &KdTree<A, T, SS, LS, K, B>,
        _stem_idx: usize,
    ) -> Option<usize>
    where
        A: AxisUnified<Coord = A>,
        T: Basics + Copy + Default + PartialOrd + PartialEq,
        SS: StemStrategy,
        LS: LeafStrategy<A, T, SS, K, B>,
    {
        None
    }
}

/// Marker type for mutable leaf strategies.
///
/// Mutable strategies support adding/removing points after construction,
/// requiring more complex traversal logic to handle non-uniform tree depths.
#[derive(Debug, Clone, Copy)]
pub struct Mutable;
impl sealed::Sealed for Mutable {}
impl Mutability for Mutable {
    fn is_mutable() -> bool {
        true
    }

    fn initial_stem_leaf_resolution<SS: StemStrategy>(
        stems_depth: usize,
        leaf_count: usize,
    ) -> crate::kd_tree::StemLeafResolution {
        // Start in Mapped state with min_stem_leaf_idx = 0 for simplicity.
        // TODO: Optimize later with Pristine state and dynamic min_stem_leaf_idx
        build_mapped_stem_leaf_resolution::<SS>(stems_depth, leaf_count)
    }

    fn get_leaf_idx<A, T, SS, LS, const K: usize, const B: usize>(
        tree: &KdTree<A, T, SS, LS, K, B>,
        query: &[A; K],
    ) -> usize
    where
        A: AxisUnified<Coord = A>,
        T: Basics + Copy + Default + PartialOrd + PartialEq,
        SS: StemStrategy,
        LS: LeafStrategy<A, T, SS, K, B>,
    {
        if tree.stem_leaf_resolution.uses_arithmetic() {
            tree.get_leaf_idx_unmapped(query)
        } else {
            tree.get_leaf_idx_mapped(query)
        }
    }

    fn resolve_terminal_stem_idx<A, T, SS, LS, const K: usize, const B: usize>(
        tree: &KdTree<A, T, SS, LS, K, B>,
        stem_idx: usize,
    ) -> Option<usize>
    where
        A: AxisUnified<Coord = A>,
        T: Basics + Copy + Default + PartialOrd + PartialEq,
        SS: StemStrategy,
        LS: LeafStrategy<A, T, SS, K, B>,
    {
        if tree.stem_leaf_resolution.uses_arithmetic() {
            None
        } else {
            tree.resolve_terminal_stem(stem_idx)
        }
    }
}

/// Trait for coordinate/axis types used in the k-d tree.
///
/// This trait defines the operations needed for coordinate values,
/// including comparison, distance calculation, and arithmetic operations.
pub trait AxisUnified:
    Copy
    + PartialEq
    + PartialOrd
    + Sub<Output = Self>
    + AddAssign<Self>
    + Debug
    + Display
    + crate::stem_strategies::CompareBlock3
    + crate::stem_strategies::CompareBlock4
{
    /// Coordinate scalar type stored in the tree and queries.
    type Coord: Copy;

    /// Zero coord.
    fn zero() -> Self::Coord;

    /// Maximum coord value.
    fn max_value() -> Self::Coord;

    /// Minimum coord value.
    fn min_value() -> Self::Coord;

    /// If coord is max value or not.
    fn is_max_value(coord: Self::Coord) -> bool;

    /// Compares two coordinate values.
    fn cmp(a: Self::Coord, b: Self::Coord) -> std::cmp::Ordering;

    /// Absolute/saturating difference along one axis, in coord units.
    fn saturating_dist(a: Self::Coord, b: Self::Coord) -> Self::Coord;

    /// Saturating addition of two coordinate values.
    fn saturating_add(a: Self::Coord, b: Self::Coord) -> Self::Coord;

    /// Returns the maximum of two coordinate values.
    fn max(a: Self::Coord, b: Self::Coord) -> Self::Coord;
}

/// Specifies whether a LeafStrategy's bucket size is a hard or soft limit
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BucketLimitType {
    /// Bucket size is completely fixed
    Hard,

    /// Bucket size is a target and can be larger than specified size if reqs
    Soft,
}

/// Strategy for how leaf storage is laid out and constructed.
/// - AX: Axis marker implementing AxisUnified (selects float or fixed semantics).
/// - T: item/content type stored alongside points.
/// - SS: stem layout strategy used to map split values into KdTree::stems (external).
/// - K: dimensionality.
/// - B: nominal bucket size (strategies may use or ignore it).
pub trait LeafStrategy<AX, T, SS, const K: usize, const B: usize>
where
    AX: AxisUnified,
    T: Basics,
    SS: StemStrategy,
{
    /// Coordinate scalar type.
    type Num;

    /// Marker type indicating whether this strategy supports mutation.
    #[allow(private_bounds)]
    type Mutability: Mutability;

    /// Whether bucket size is a hard or soft limit
    const BUCKET_LIMIT_TYPE: BucketLimitType;

    // ---- Construction ----

    /// Create a builder with an intended capacity (in points).
    fn new_with_capacity(capacity: usize) -> Self;

    /// Create a new LeafStrategy with a single, empty leaf
    fn new_with_empty_leaf() -> Self;

    // ---- Introspection / minimal accessors ----

    /// Total number of stored items.
    fn size(&self) -> usize;

    /// Number of leaves maintained by the strategy (buckets/extents).
    fn leaf_count(&self) -> usize;

    /// Number of items in a given leaf.
    fn leaf_len(&self, leaf_idx: usize) -> usize;

    /// Returns a view into the specified leaf's data.
    fn leaf_view(&self, leaf_idx: usize) -> LeafView<'_, AX, T, K, B>;

    /// Appends a new leaf to the storage.
    fn append_leaf(&mut self, leaf_points: &[&[AX]; K], leaf_items: &[T]);
}

/// Trait for leaf strategies that support mutation (adding/removing points).
pub trait MutableLeafStrategy<AX, T, SS, const K: usize, const B: usize>:
    LeafStrategy<AX, T, SS, K, B>
where
    AX: AxisUnified,
    T: Basics,
    SS: StemStrategy,
{
    /// Add an item to a leaf.
    ///
    /// * The leaf is expected to exist.
    /// * The leaf must not be full.
    /// * The caller must ensure that the right leaf is being added to.
    fn add_to_leaf(&mut self, leaf_idx: usize, point: &[AX; K], item: T);

    /// Remove an item (or items) from a leaf
    ///
    /// * The leaf is expected to exist.
    /// * The whole leaf is searched for any entries where both the point and the
    ///   value match. Any entries matching are removed.
    fn remove_from_leaf(&mut self, leaf_idx: usize, point: &[AX; K], item: T);

    /// Returns true if the specified leaf is full.
    fn is_leaf_full(&self, leaf_idx: usize) -> bool;

    /// Splits a full leaf, returning the pivot value and the index of
    /// the new leaf that the leaf was split into.
    fn split_leaf(&mut self, leaf_idx: usize, split_dim: usize) -> (AX, usize);
}

/// Trait for distance metrics used in spatial queries.
///
/// This trait supports both standard distance metrics (e.g., squared Euclidean)
/// and similarity metrics (e.g., dot product) through the ORDERING associated constant.
pub trait DistanceMetricUnified<A: Copy, const K: usize> {
    /// Accumulator / distance scalar type.
    type Output: AxisUnified<Coord = Self::Output>;

    /// Desired sort order on distances:
    /// - Less    => smaller is better (standard distances)
    /// - Greater => larger is better (dot product)
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

    /// Returns true if `a` is better than `b` according to the metric's ordering.
    #[inline(always)]
    fn better(a: Self::Output, b: Self::Output) -> bool {
        match Self::ORDERING {
            std::cmp::Ordering::Less => a < b,    // smaller is better
            std::cmp::Ordering::Greater => a > b, // larger is better (dot)
            std::cmp::Ordering::Equal => false,
        }
    }

    /// Compares two distance values according to the metric's ordering.
    #[inline(always)]
    fn cmp(a: Self::Output, b: Self::Output) -> std::cmp::Ordering {
        a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal)
        // match Self::ORDERING {
        //     std::cmp::Ordering::Less => base,
        //     std::cmp::Ordering::Greater => base.reverse(),
        //     std::cmp::Ordering::Equal => std::cmp::Ordering::Equal,
        // }
    }

    // ---- SIMD distance checks for backtracking ----
    // (moved to BacktrackBlock3/BacktrackBlock4 traits in stem_strategies)
}

/// Helper macro to implement SIMD block support (CompareBlock and SimdPrune) for a specific block size.
///
/// This is an internal macro used by impl_axis_float and impl_axis_fixed.
#[allow(unused_macros)]
macro_rules! impl_simd_block_support {
    ($t:ty, 3, $prune_fn:path, $compare_fn:path) => {
        impl crate::stem_strategies::CompareBlock3 for $t {
            #[inline(always)]
            fn compare_block3_impl(
                stems_ptr: std::ptr::NonNull<u8>,
                query_val: Self,
                block_base_idx: usize,
            ) -> u8 {
                $compare_fn(stems_ptr, block_base_idx, query_val)
            }
        }

        // SimdPrune implementation for Block3 will be added in Step 4
    };

    ($t:ty, 4, $prune_fn:path, $compare_fn:path) => {
        impl crate::stem_strategies::CompareBlock4 for $t {
            #[inline(always)]
            fn compare_block4_impl(
                stems_ptr: std::ptr::NonNull<u8>,
                query_val: Self,
                block_base_idx: usize,
            ) -> u8 {
                $compare_fn(stems_ptr, block_base_idx, query_val)
            }
        }

        // SimdPrune implementation for Block4 will be added in Step 4
    };

    // Block5 support placeholder for future
    ($t:ty, 5, $prune_fn:path, $compare_fn:path) => {
        // Block5 traits don't exist yet
        compile_error!("Block5 support is not yet implemented");
    };
}

/// Macro to implement AxisUnified for floating-point types.
#[macro_export]
macro_rules! impl_axis_float {
    // Pattern with SIMD block support
    ($t:ty, SIMD_BLOCK_SUPPORT => ( $( $block_size:literal => ($prune_fn:path, $compare_fn:path) ),* $(,)? )) => {
        impl_axis_float!($t); // First implement the basic AxisUnified trait

        // Then implement SIMD block support for each specified block size
        $(
            impl_simd_block_support!($t, $block_size, $prune_fn, $compare_fn);
        )*
    };

    // Base pattern without SIMD block support (uses default unimplemented!() from traits)
    ($t:ty) => {
        impl AxisUnified for $t {
            type Coord = $t;

            #[inline(always)]
            fn zero() -> Self::Coord {
                0.0
            }

            #[inline(always)]
            fn max_value() -> Self::Coord {
                <$t>::infinity()
            }

            #[inline(always)]
            fn min_value() -> Self::Coord {
                <$t>::neg_infinity()
            }

            #[inline(always)]
            fn is_max_value(coord: Self::Coord) -> bool {
                coord.is_infinite() && coord.is_sign_positive()
            }

            #[inline(always)]
            fn cmp(a: Self::Coord, b: Self::Coord) -> std::cmp::Ordering {
                // debug_assert!(
                //     a.is_finite() && b.is_finite(),
                //     "NaNs / Infinities should not be present in axis coordinates"
                // );
                if a < b {
                    std::cmp::Ordering::Less
                } else if b > a {
                    std::cmp::Ordering::Greater
                } else {
                    std::cmp::Ordering::Equal
                }
            }

            #[inline(always)]
            fn saturating_dist(a: Self::Coord, b: Self::Coord) -> Self::Coord {
                (a - b).abs()
            }

            #[inline(always)]
            fn saturating_add(a: Self::Coord, b: Self::Coord) -> Self::Coord {
                a + b
            }

            #[inline(always)]
            fn max(a: Self::Coord, b: Self::Coord) -> Self::Coord {
                a.max(b)
            }
        }
    };
}

/// Macro to implement AxisUnified for fixed-point types.
macro_rules! impl_axis_fixed {
    // Pattern with SIMD block support
    ($t:ty, SIMD_BLOCK_SUPPORT => ( $( $block_size:literal => ($prune_fn:path, $compare_fn:path) ),* $(,)? )) => {
        impl_axis_fixed!($t); // First implement the basic AxisUnified trait

        // Then implement SIMD block support for each specified block size
        $(
            impl_simd_block_support!($t, $block_size, $prune_fn, $compare_fn);
        )*
    };

    // Base pattern without SIMD block support (uses default unimplemented!() from traits)
    ($t:ty) => {
        impl AxisUnified for $t {
            type Coord = $t;

            #[inline(always)]
            fn zero() -> Self::Coord {
                <$t>::from_num(0)
            }

            #[inline(always)]
            fn max_value() -> Self::Coord {
                <Self::Coord>::MAX
            }

            #[inline(always)]
            fn min_value() -> Self::Coord {
                unimplemented!("min_value not yet implemented for fixed point types")
            }

            #[inline(always)]
            fn is_max_value(coord: Self::Coord) -> bool {
                coord == <$t>::max_value()
            }

            #[inline(always)]
            fn cmp(a: Self::Coord, b: Self::Coord) -> std::cmp::Ordering {
                a.cmp(&b)
            }

            #[inline(always)]
            fn saturating_dist(a: Self::Coord, b: Self::Coord) -> Self::Coord {
                if a >= b {
                    a - b
                } else {
                    b - a
                }
            }

            #[inline(always)]
            fn saturating_add(a: Self::Coord, b: Self::Coord) -> Self::Coord {
                a.saturating_add(b)
            }

            #[inline(always)]
            fn max(a: Self::Coord, b: Self::Coord) -> Self::Coord {
                a.max(b)
            }
        }
    };
}

/// Squared Euclidean distance metric, parameterized by output type R.
///
/// This metric computes the squared Euclidean distance between points,
/// widening input coordinates to the output type R.
pub struct SquaredEuclidean<R>(core::marker::PhantomData<R>);

impl<A, R, const K: usize> DistanceMetricUnified<A, K> for SquaredEuclidean<R>
where
    A: Copy,
    R: AxisUnified<Coord = R>
        + LossyFrom<A>
        + core::ops::Mul<Output = R>
        + core::ops::Add<Output = R>,
{
    type Output = R;
    const ORDERING: std::cmp::Ordering = std::cmp::Ordering::Less;

    #[inline(always)]
    fn widen_coord(a: A) -> R {
        R::lossy_from(a)
    }

    #[inline(always)]
    fn dist1(a: R, b: R) -> R {
        let d = if a >= b { a - b } else { b - a };
        d * d
    }

    // SIMD backtrack checks are now provided by BacktrackBlock3/BacktrackBlock4 traits.
}

/// Manhattan distance metric, parameterized by output type R.
///
/// This metric computes the Manhattan distance between points,
/// widening input coordinates to the output type R.
pub struct Manhattan<R>(core::marker::PhantomData<R>);

impl<A, R, const K: usize> DistanceMetricUnified<A, K> for Manhattan<R>
where
    A: Copy,
    R: AxisUnified<Coord = R>
        + LossyFrom<A>
        + core::ops::Mul<Output = R>
        + core::ops::Add<Output = R>,
{
    type Output = R;
    const ORDERING: std::cmp::Ordering = std::cmp::Ordering::Less;

    #[inline(always)]
    fn widen_coord(a: A) -> R {
        R::lossy_from(a)
    }

    #[inline(always)]
    fn dist1(a: R, b: R) -> R {
        if a >= b {
            a - b
        } else {
            b - a
        }
    }
}

// Axis impls stay as they are.
impl_axis_float!(f32);
impl_axis_float!(f64);

#[cfg(feature = "f16")]
impl AxisUnified for half::f16 {
    type Coord = half::f16;

    #[inline(always)]
    fn zero() -> Self::Coord {
        half::f16::from_f32(0.0)
    }

    #[inline(always)]
    fn max_value() -> Self::Coord {
        half::f16::from_f32(f32::INFINITY)
    }

    #[inline(always)]
    fn min_value() -> Self::Coord {
        half::f16::from_f32(f32::NEG_INFINITY)
    }

    #[inline(always)]
    fn is_max_value(coord: Self::Coord) -> bool {
        coord.is_infinite() && coord.is_sign_positive()
    }

    #[inline(always)]
    fn cmp(a: Self::Coord, b: Self::Coord) -> std::cmp::Ordering {
        if a < b {
            std::cmp::Ordering::Less
        } else if b > a {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    }

    #[inline(always)]
    fn saturating_dist(a: Self::Coord, b: Self::Coord) -> Self::Coord {
        (a - b).abs()
    }

    #[inline(always)]
    fn saturating_add(a: Self::Coord, b: Self::Coord) -> Self::Coord {
        a + b
    }

    #[inline(always)]
    fn max(a: Self::Coord, b: Self::Coord) -> Self::Coord {
        if a > b {
            a
        } else {
            b
        }
    }
}
impl_axis_fixed!(FixedI32<U16>);
impl_axis_fixed!(FixedI32<U0>);
impl_axis_fixed!(FixedU16<U8>);

/// Dot product similarity metric, parameterized by output type R.
///
/// This metric computes the dot product between points (higher is better),
/// widening input coordinates to the output type R.
pub struct DotProduct<R>(core::marker::PhantomData<R>);

impl<A, R, const K: usize> DistanceMetricUnified<A, K> for DotProduct<R>
where
    A: Copy,
    R: AxisUnified<Coord = R>
        + LossyFrom<A>
        + core::ops::Mul<Output = R>
        + core::ops::Add<Output = R>,
{
    type Output = R;
    const ORDERING: std::cmp::Ordering = std::cmp::Ordering::Greater;

    #[inline(always)]
    fn widen_coord(a: A) -> R {
        R::lossy_from(a)
    }

    #[inline(always)]
    fn dist1(a: R, b: R) -> R {
        a * b
    }
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

/// Updates the nearest neighbor from a batch of 64 distance calculations.
///
/// Used for benchmarking autovectorization with concrete types.
pub(crate) fn update_nearest(
    dists: &[f32; 64],
    items: &[usize; 64],
    best_dist: &mut f32,
    best_item: &mut usize,
) {
    // Autovectorizes with 256bit vectors on x86_64 where available
    // 341 loops (1 item per loop, unrolled x 3) of 4-8 instructions per item
    let (leaf_best_item, leaf_best_dist) = dists
        .iter()
        .enumerate()
        .min_by(|&(_, &a), (_, b)| a.partial_cmp(*b).unwrap())
        .unwrap();

    // 6 instructions, 1 branch
    if *leaf_best_dist < *best_dist {
        *best_dist = *leaf_best_dist;
        *best_item = items[leaf_best_item];
    }
}

/// Hook function for cargo-asm to verify autovectorization with concrete types.
/// This should vectorize perfectly on zen5 and serve as a baseline for the generic trait implementation.
#[inline(never)]
pub fn bench_update_nearest_f32_64(
    content_points: &[[f32; 64]; 3],
    items: &[usize; 64],
    query: &[f32; 3],
) -> (f32, usize) {
    let mut best_dist = f32::INFINITY;
    let mut best_item = 0usize;

    let mut acc = [0f32; 64];

    calc_dists(content_points, &mut acc, query);

    update_nearest(&acc, items, &mut best_dist, &mut best_item);

    (best_dist, best_item)
}
