use crate::kd_tree::leaf_view::LeafView;
use crate::StemStrategy;
use aligned_vec::AVec;
use fixed::traits::LossyFrom;
use fixed::types::extra::{U0, U16, U8};
use fixed::{FixedI32, FixedU16};
use ordered_float::Float;
use std::fmt::{Debug, Display};
use std::ops::{AddAssign, Sub};

pub trait Basics: Copy + Debug + Default + Send + Sync + 'static {}
impl<T> Basics for T where T: Copy + Debug + Default + Send + Sync + 'static {}

// Make AxisUnified a supertrait of the common numeric bounds you need everywhere.
pub trait AxisUnified:
    Copy + PartialEq + PartialOrd + Sub<Output = Self> + AddAssign<Self> + Debug + Display
{
    /// Coordinate scalar type stored in the tree and queries.
    type Coord: Copy;

    /// Zero coord.
    fn zero() -> Self::Coord;

    /// Maximum coord value.
    fn max_value() -> Self::Coord;

    /// If coord is max value or not.
    fn is_max_value(coord: Self::Coord) -> bool;

    fn cmp(a: Self::Coord, b: Self::Coord) -> std::cmp::Ordering;

    /// Absolute/saturating difference along one axis, in coord units.
    fn saturating_dist(a: Self::Coord, b: Self::Coord) -> Self::Coord;

    fn saturating_add(a: Self::Coord, b: Self::Coord) -> Self::Coord;
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

    // ---- Construction ----

    /// Create a builder with an intended capacity (in points).
    fn new_with_capacity(capacity: usize) -> Self;

    /// Bulk-build from a slice of points. Implementations should:
    /// - write split values into `stems` at indices determined by `stem_strategy`,
    /// - lay out leaf storage according to the strategy,
    /// - return the max stem level reached (for later traversal).
    fn bulk_build_from_slice(
        &mut self,
        source: &[[Self::Num; K]],
        stems: &mut AVec<Self::Num>,
        stem_strategy: SS,
    ) -> i32;

    /// Finalization hook (e.g., trim stems or compact internal buffers).
    fn finalize(
        &mut self,
        stems: &mut AVec<Self::Num>,
        stem_strategy: &mut SS,
        max_stem_level: i32,
    );

    // ---- Introspection / minimal accessors ----

    /// Total number of stored items.
    fn size(&self) -> usize;

    /// Number of leaves maintained by the strategy (buckets/extents).
    fn leaf_count(&self) -> usize;

    /// Number of items in a given leaf.
    fn leaf_len(&self, leaf_idx: usize) -> usize;

    fn leaf_view(&self, leaf_idx: usize) -> LeafView<'_, AX, T, K, B>;

    fn append_leaf(&mut self, leaf_points: &[&[AX]; K], leaf_items: &[T]);
}

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

    fn is_leaf_full(&self, leaf_idx: usize) -> bool;

    fn split_leaf(&mut self, leaf_idx: usize) -> usize;
}

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

    #[inline(always)]
    fn better(a: Self::Output, b: Self::Output) -> bool {
        match Self::ORDERING {
            std::cmp::Ordering::Less => a < b,    // smaller is better
            std::cmp::Ordering::Greater => a > b, // larger is better (dot)
            std::cmp::Ordering::Equal => false,
        }
    }

    #[inline(always)]
    fn cmp(a: Self::Output, b: Self::Output) -> std::cmp::Ordering {
        let base = a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal);
        match Self::ORDERING {
            std::cmp::Ordering::Less => base,
            std::cmp::Ordering::Greater => base.reverse(),
            std::cmp::Ordering::Equal => std::cmp::Ordering::Equal,
        }
    }
}

#[macro_export]
macro_rules! impl_axis_float {
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
            fn is_max_value(coord: Self::Coord) -> bool {
                coord.is_infinite()
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
        }
    };
}

macro_rules! impl_axis_fixed {
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
        }
    };
}

// Squared Euclidean metric, parameterized by Output type R.
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
}

// Axis impls stay as they are.
impl_axis_float!(f32);
impl_axis_float!(f64);
impl_axis_fixed!(FixedI32<U16>);
impl_axis_fixed!(FixedI32<U0>);
impl_axis_fixed!(FixedU16<U8>);

// Dot product metric, also parameterized by Output type R.
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
