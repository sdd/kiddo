use crate::StemStrategy;
use aligned_vec::AVec;
use fixed::traits::{Fixed, LossyFrom, LossyInto};
use fixed::types::extra::{U0, U16};
use fixed::FixedI32;
use ordered_float::Float;
use std::fmt::Debug;
use std::ops::{AddAssign, Sub};

pub trait Basics: Copy + Debug + Default + Send + Sync + 'static {}
impl<T> Basics for T where T: Copy + Debug + Default + Send + Sync + 'static {}

// Make AxisUnified a supertrait of the common numeric bounds you need everywhere.
pub trait AxisUnified:
    Copy + PartialEq + PartialOrd + Sub<Output = Self> + AddAssign<Self>
{
    /// Coordinate scalar type stored in the tree and queries.
    type Coord: Copy;

    /// Zero coord.
    fn zero() -> Self::Coord;

    /// Maximum coord value.
    fn max_value() -> Self::Coord;

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
    fn new_builder(capacity: usize) -> Self;

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

    // ---- Mutation (optional; immutable strategies may panic or be no-ops) ----

    /// Add a point/item pair into the structure, updating stems as necessary.
    fn add_point(
        &mut self,
        point: &[Self::Num; K],
        item: T,
        stems: &mut AVec<Self::Num>,
        stem_strategy: &mut SS,
    );

    /// Remove a specific point/item pair; returns number removed.
    fn remove_point(&mut self, point: &[Self::Num; K], item: T) -> usize;

    // ---- Introspection / minimal accessors ----

    /// Total number of stored items.
    fn size(&self) -> usize;

    /// Number of leaves maintained by the strategy (buckets/extents).
    fn leaf_count(&self) -> usize;

    /// Number of items in a given leaf.
    fn leaf_len(&self, leaf_idx: usize) -> usize;

    fn leaf_view(&self, leaf_idx: usize) -> LeafView<'_, AX, T, K>;
}

#[derive(Debug, Default)]
pub struct DummyLeafStrategy {}

impl<AX, T, SS, const K: usize, const B: usize> LeafStrategy<AX, T, SS, K, B> for DummyLeafStrategy
where
    AX: AxisUnified,
    T: Basics,
    SS: StemStrategy,
{
    type Num = ();

    fn new_builder(_capacity: usize) -> Self {
        unimplemented!()
    }

    fn bulk_build_from_slice(
        &mut self,
        _source: &[[Self::Num; K]],
        _stems: &mut AVec<Self::Num>,
        _stem_strategy: SS,
    ) -> i32 {
        unimplemented!()
    }

    fn finalize(
        &mut self,
        _stems: &mut AVec<Self::Num>,
        _stem_strategy: &mut SS,
        _max_stem_level: i32,
    ) {
        unimplemented!()
    }

    fn add_point(
        &mut self,
        _point: &[Self::Num; K],
        _item: T,
        _stems: &mut AVec<Self::Num>,
        _stem_strategy: &mut SS,
    ) {
        unimplemented!()
    }

    fn remove_point(&mut self, _point: &[Self::Num; K], _item: T) -> usize {
        unimplemented!()
    }

    fn size(&self) -> usize {
        unimplemented!()
    }

    fn leaf_count(&self) -> usize {
        unimplemented!()
    }

    fn leaf_len(&self, _leaf_idx: usize) -> usize {
        unimplemented!()
    }

    fn leaf_view(&self, _leaf_idx: usize) -> LeafView<'_, AX, T, K> {
        unimplemented!()
    }
}

pub type LeafView<'a, AX, T, const K: usize> = ([&'a [AX]; K], &'a [T]);

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
                <$t>::max_value()
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

pub trait DistanceMetricUnified<A, const K: usize> {
    // Output must itself be a valid axis-like type
    type Output: AxisUnified<Coord = Self::Output>;

    /// Desired sort order on Output to get "nearest" first:
    /// - Ordering::Less    => ascending (smaller is better)
    /// - Ordering::Greater => descending (larger is better)
    const ORDERING: std::cmp::Ordering;

    fn dist(a: &[A; K], b: &[A; K]) -> Self::Output;
    fn dist1(a: A, b: A) -> Self::Output;

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

pub struct SquaredEuclidean;
pub struct SquaredEuclideanWiden<R>(core::marker::PhantomData<R>);

#[macro_export]
macro_rules! impl_squared_euclidean_float {
    ($t:ty) => {
        impl<const K: usize> DistanceMetricUnified<$t, K> for SquaredEuclidean {
            type Output = $t;
            const ORDERING: std::cmp::Ordering = std::cmp::Ordering::Less;

            #[inline(always)]
            fn dist(a: &[$t; K], b: &[$t; K]) -> Self::Output {
                a.iter()
                    .zip(b.iter())
                    .map(|(&ai, &bi)| {
                        let d = ai - bi;
                        d * d
                    })
                    .fold(0.0, |acc, x| acc + x)
            }

            #[inline(always)]
            fn dist1(a: $t, b: $t) -> Self::Output {
                let d = a - b;
                d * d
            }
        }
    };
}

#[macro_export]
macro_rules! impl_squared_euclidean_fixed {
    ($t:ty) => {
        impl<const K: usize> DistanceMetricUnified<$t, K> for SquaredEuclidean {
            type Output = $t;
            const ORDERING: std::cmp::Ordering = std::cmp::Ordering::Less;

            #[inline(always)]
            fn dist(a: &[$t; K], b: &[$t; K]) -> Self::Output {
                a.iter()
                    .zip(b.iter())
                    .map(|(&ai, &bi)| {
                        let d = if ai >= bi { ai - bi } else { bi - ai };
                        d * d
                    })
                    .fold(<$t>::from_num(0), |acc, x| acc + x)
            }

            #[inline(always)]
            fn dist1(a: $t, b: $t) -> Self::Output {
                let d = if a >= b { a - b } else { b - a };
                d * d
            }
        }
    };
}

#[macro_export]
macro_rules! impl_squared_euclidean_fixed_widening {
    ($t:ty) => {
        impl<R, const K: usize> $crate::traits_unified_2::DistanceMetricUnified<$t, K>
            for $crate::traits_unified_2::SquaredEuclideanWiden<R>
        where
            R: $crate::traits_unified_2::AxisUnified<Coord = R>
                + fixed::traits::LossyFrom<$t>
                + std::ops::Mul<Output = R>
                + std::ops::Add<Output = R>,
        {
            type Output = R;
            const ORDERING: std::cmp::Ordering = std::cmp::Ordering::Less;

            #[inline(always)]
            fn dist(a: &[$t; K], b: &[$t; K]) -> Self::Output {
                a.iter()
                    .zip(b.iter())
                    .map(|(&ai, &bi)| {
                        let ai: R = fixed::traits::LossyFrom::<$t>::lossy_from(ai);
                        let bi: R = fixed::traits::LossyFrom::<$t>::lossy_from(bi);

                        let d = if ai >= bi { ai - bi } else { bi - ai };
                        d * d
                    })
                    .fold(R::zero(), |acc, x| acc + x)
            }

            #[inline(always)]
            fn dist1(a: $t, b: $t) -> Self::Output {
                let a: R = fixed::traits::LossyFrom::<$t>::lossy_from(a);
                let b: R = fixed::traits::LossyFrom::<$t>::lossy_from(b);

                let d: R = if a >= b { a - b } else { b - a };
                d * d
            }
        }
    };
}

#[macro_export]
macro_rules! impl_squared_euclidean_float_widening {
    ($src:ty => $dst:ty) => {
        impl<const K: usize> $crate::traits_unified_2::DistanceMetricUnified<$src, K>
            for $crate::traits_unified_2::SquaredEuclideanWiden<$dst>
        where
            $dst: $crate::traits_unified_2::AxisUnified + From<$src>,
        {
            type Output = $dst;
            const ORDERING: std::cmp::Ordering = std::cmp::Ordering::Less;

            #[inline(always)]
            fn dist(a: &[$src; K], b: &[$src; K]) -> Self::Output {
                a.iter()
                    .zip(b.iter())
                    .map(|(&ai, &bi)| {
                        let ai: $dst = <$dst as From<$src>>::from(ai);
                        let bi: $dst = <$dst as From<$src>>::from(bi);
                        let d = ai - bi;
                        d * d
                    })
                    .fold(<$dst as From<$src>>::from(0 as $src), |acc, x| acc + x)
            }

            #[inline(always)]
            fn dist1(a: $src, b: $src) -> Self::Output {
                let a: $dst = <$dst as From<$src>>::from(a);
                let b: $dst = <$dst as From<$src>>::from(b);
                let d = a - b;
                d * d
            }
        }
    };
}

impl_axis_float!(f32);
impl_axis_float!(f64);
impl_axis_fixed!(FixedI32<U16>);
impl_axis_fixed!(FixedI32<U0>);

impl_squared_euclidean_float!(f32);
impl_squared_euclidean_float!(f64);
impl_squared_euclidean_fixed!(FixedI32<U16>);
impl_squared_euclidean_fixed!(FixedI32<U0>);

impl_squared_euclidean_float_widening!(f32 => f64);

impl_squared_euclidean_fixed_widening!(FixedI32<U16>);
impl_squared_euclidean_fixed_widening!(FixedI32<U0>);

pub struct DotProduct;
pub struct DotProductWiden<R>(core::marker::PhantomData<R>);

#[macro_export]
macro_rules! impl_dot_product_float {
    ($t:ty) => {
        impl<const K: usize> DistanceMetricUnified<$t, K> for DotProduct {
            type Output = $t;
            const ORDERING: std::cmp::Ordering = std::cmp::Ordering::Greater;

            #[inline(always)]
            fn dist(a: &[$t; K], b: &[$t; K]) -> Self::Output {
                a.iter()
                    .zip(b.iter())
                    .map(|(&ai, &bi)| ai * bi)
                    .fold(0.0, |acc, x| acc + x)
            }

            #[inline(always)]
            fn dist1(a: $t, b: $t) -> Self::Output {
                a * b
            }
        }
    };
}

#[macro_export]
macro_rules! impl_dot_product_fixed {
    ($t:ty) => {
        impl<const K: usize> DistanceMetricUnified<$t, K> for DotProduct {
            type Output = $t;
            const ORDERING: std::cmp::Ordering = std::cmp::Ordering::Greater;

            #[inline(always)]
            fn dist(a: &[$t; K], b: &[$t; K]) -> Self::Output {
                a.iter()
                    .zip(b.iter())
                    .map(|(&ai, &bi)| ai * bi)
                    .fold(<Self::Output>::from_num(0), |acc, x| acc + x)
            }

            #[inline(always)]
            fn dist1(a: $t, b: $t) -> Self::Output {
                a * b
            }
        }
    };
}

#[macro_export]
macro_rules! impl_dot_product_fixed_widening {
    ($t:ty) => {
        impl<R, const K: usize> DistanceMetricUnified<$t, K> for DotProductWiden<R>
        where
            R: Fixed + LossyFrom<$t> + $crate::traits_unified_2::AxisUnified<Coord = R>,
        {
            type Output = R;
            const ORDERING: std::cmp::Ordering = std::cmp::Ordering::Greater;

            #[inline(always)]
            fn dist(a: &[$t; K], b: &[$t; K]) -> Self::Output {
                a.iter()
                    .zip(b.iter())
                    .map(|(&ai, &bi)| {
                        let ai: Self::Output = ai.lossy_into();
                        let bi: Self::Output = bi.lossy_into();

                        ai * bi
                    })
                    .fold(<Self::Output>::from_num(0), |acc, x| acc + x)
            }

            #[inline(always)]
            fn dist1(a: $t, b: $t) -> Self::Output {
                let a: Self::Output = a.lossy_into();
                let b: Self::Output = b.lossy_into();

                a * b
            }
        }
    };
}

impl_dot_product_float!(f32);
impl_dot_product_float!(f64);
impl_dot_product_fixed!(FixedI32<U16>);
impl_dot_product_fixed!(FixedI32<U0>);

impl_dot_product_fixed_widening!(FixedI32<U16>);
impl_dot_product_fixed_widening!(FixedI32<U0>);

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
