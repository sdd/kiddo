#![allow(clippy::missing_safety_doc)]

/// AVX2 distance metric trait
#[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
#[doc(hidden)]
pub mod distance_metric_avx2;

/// AVX512 distance metric trait
#[cfg(all(feature = "simd", target_feature = "avx512f"))]
#[doc(hidden)]
pub mod distance_metric_avx512;

/// Core distance metric trait
#[doc(hidden)]
pub mod distance_metric_core;

/// NEON distance metric trait
#[cfg(all(feature = "simd", target_arch = "aarch64", target_feature = "neon"))]
#[doc(hidden)]
pub mod distance_metric_neon;

/// Dot Product distance metric
#[doc(hidden)]
pub mod dot_product;

/// Chebyshev distance metric
#[doc(hidden)]
pub mod chebyshev;

/// Minkowski distance metric
#[doc(hidden)]
pub mod minkowski;

/// Manhattan distance metric
#[doc(hidden)]
pub mod manhattan;

/// Squared Euclidean distance metric
#[doc(hidden)]
pub mod squared_euclidean;

/// Shared distance metric functions
pub(crate) mod common;

#[doc(hidden)]
pub use distance_metric_core::DistanceMetricCore;
#[cfg(any(
    all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"),
    all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"),
    all(feature = "simd", target_arch = "aarch64", target_feature = "neon")
))]
use std::any::TypeId;

use crate::stem_strategy::donnelly_2_blockmarker_simd::{
    DistanceMetricSimdBlock3, DistanceMetricSimdBlock4,
};
use crate::Axis;
#[cfg(any(
    all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"),
    all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"),
    all(feature = "simd", target_arch = "aarch64", target_feature = "neon")
))]
use crate::{
    leaf_view::{LeafArena, LeafView},
    results::result_collection::{BestNeighbourResultCollection, ResultCollection},
    BestQueryResultItem, QueryResultItem,
};

#[doc(inline)]
pub use chebyshev::Chebyshev;
#[doc(inline)]
pub use dot_product::DotProduct;
#[doc(inline)]
pub use manhattan::Manhattan;
#[doc(inline)]
pub use minkowski::Minkowski;
#[doc(inline)]
pub use squared_euclidean::SquaredEuclidean;

#[cfg(feature = "simd")]
macro_rules! with_nearest_result_emitter {
    ($results:expr, $distance_ty:ty, $item_ty:ty, $emit:ident, $body:block) => {{
        let mut $emit = |candidate_dist: $distance_ty, item: $item_ty| {
            #[cfg(feature = "result_collection_stats")]
            crate::results::result_collection_stats::record_candidate_emitted();

            let candidate = QueryResultItem {
                point: (),
                distance: std::mem::transmute_copy::<$distance_ty, Self::Output>(&candidate_dist),
                item,
            };

            $results.add(candidate);
        };

        $body
    }};
}

#[cfg(feature = "simd")]
macro_rules! with_best_result_emitter {
    ($results:expr, $threshold_item:expr, $distance_ty:ty, $item_ty:ty, $emit:ident, $body:block) => {{
        let threshold_item = $threshold_item;
        let mut $emit = |candidate_dist: $distance_ty, item: $item_ty| {
            if threshold_item.is_some_and(|worst_item| item >= worst_item) {
                #[cfg(feature = "result_collection_stats")]
                crate::results::result_collection_stats::record_best_item_threshold_reject();
                return;
            }

            #[cfg(feature = "result_collection_stats")]
            crate::results::result_collection_stats::record_candidate_emitted();

            let candidate = BestQueryResultItem {
                point: (),
                distance: std::mem::transmute_copy::<$distance_ty, Self::Output>(&candidate_dist),
                item,
            };

            $results.add(candidate);
        };

        $body
    }};
}

/// AVX512 extension hooks.
///
/// Default behavior is "not specialized". Concrete metrics can override hook
/// methods in arch-specific code without changing public query bounds.
#[doc(hidden)]
pub trait DistanceMetricAvx512<A: Copy>: DistanceMetricCore<A> {
    /// Type that provides implementations of the AVX512 leaf ops
    #[cfg(all(feature = "simd", target_feature = "avx512f"))]
    type Avx512F64Ops: distance_metric_avx512::Avx512F64LeafOps + 'static;

    /// Type that provides implementations of the AVX512 f32 leaf ops.
    #[cfg(all(feature = "simd", target_feature = "avx512f"))]
    type Avx512F32Ops: distance_metric_avx512::Avx512F32LeafOps + 'static;

    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
    #[inline(always)]
    /// Try an AVX512-specialized `nearest_one` leaf kernel.
    unsafe fn try_nearest_one_leaf_avx512<T, const K: usize, const B: usize>(
        leaf: &LeafView<'_, A, T, K, B>,
        query_wide: &[Self::Output; K],
        best_dist: &mut Self::Output,
        best_item: &mut T,
    ) -> bool
    where
        A: Axis<Coord = A> + 'static,
        T: crate::Content,
        Self::Output: Axis<Coord = Self::Output> + 'static,
    {
        if TypeId::of::<A>() == TypeId::of::<f64>()
            && TypeId::of::<Self::Output>() == TypeId::of::<f64>()
            && TypeId::of::<Self::Avx512F64Ops>()
                != TypeId::of::<distance_metric_avx512::UnsupportedAvx512F64LeafOps>()
        {
            let leaf =
                &*(leaf as *const LeafView<'_, A, T, K, B> as *const LeafView<'_, f64, T, K, B>);
            let query_wide = &*(query_wide as *const [Self::Output; K] as *const [f64; K]);
            let best_dist = &mut *(best_dist as *mut Self::Output as *mut f64);

            crate::leaf_view_chunked::nearest_one::avx512::nearest_one_avx512_unchecked::<
                f64,
                Self::Avx512F64Ops,
                T,
                K,
                B,
            >(leaf, query_wide, best_dist, best_item);

            return true;
        }

        if TypeId::of::<A>() == TypeId::of::<f32>()
            && TypeId::of::<Self::Output>() == TypeId::of::<f32>()
            && TypeId::of::<Self::Avx512F32Ops>()
                != TypeId::of::<distance_metric_avx512::UnsupportedAvx512F32LeafOps>()
        {
            let leaf =
                &*(leaf as *const LeafView<'_, A, T, K, B> as *const LeafView<'_, f32, T, K, B>);
            let query_wide = &*(query_wide as *const [Self::Output; K] as *const [f32; K]);
            let best_dist = &mut *(best_dist as *mut Self::Output as *mut f32);

            crate::leaf_view_chunked::nearest_one::avx512::nearest_one_avx512_unchecked_f32::<
                Self::Avx512F32Ops,
                T,
                K,
                B,
            >(leaf, query_wide, best_dist, best_item);

            return true;
        }

        false
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
    #[inline(always)]
    /// Try an AVX512-specialized `nearest_one` arena kernel.
    unsafe fn try_nearest_one_arena_avx512<T, const K: usize>(
        arena: &LeafArena<'_, A, T, K>,
        query_wide: &[Self::Output; K],
        best_dist: &mut Self::Output,
        best_item: &mut T,
    ) -> bool
    where
        A: Axis<Coord = A> + 'static,
        T: crate::Content,
        Self::Output: Axis<Coord = Self::Output> + 'static,
    {
        if TypeId::of::<A>() == TypeId::of::<f64>()
            && TypeId::of::<Self::Output>() == TypeId::of::<f64>()
            && TypeId::of::<Self::Avx512F64Ops>()
                != TypeId::of::<distance_metric_avx512::UnsupportedAvx512F64LeafOps>()
        {
            let query_wide = &*(query_wide as *const [Self::Output; K] as *const [f64; K]);
            let best_dist = &mut *(best_dist as *mut Self::Output as *mut f64);
            let query_ptr = query_wide.as_ptr();
            let mut tile_base = arena.as_ptr();
            let mut remaining = arena.len();

            while remaining != 0 {
                let tile_len = crate::leaf_view::leaf_arena_tile_len(remaining);
                crate::leaf_view_chunked::nearest_one::avx512::nearest_one_avx512_arena_unchecked::<
                    f64,
                    Self::Avx512F64Ops,
                    T,
                    K,
                >(tile_base, tile_len, query_ptr, best_dist, best_item);
                let tile_bytes =
                    K * tile_len * std::mem::size_of::<f64>() + tile_len * std::mem::size_of::<T>();
                tile_base = tile_base.add(tile_bytes);
                remaining -= tile_len;
            }

            return true;
        }

        if TypeId::of::<A>() == TypeId::of::<f32>()
            && TypeId::of::<Self::Output>() == TypeId::of::<f32>()
            && TypeId::of::<Self::Avx512F32Ops>()
                != TypeId::of::<distance_metric_avx512::UnsupportedAvx512F32LeafOps>()
        {
            let query_wide = &*(query_wide as *const [Self::Output; K] as *const [f32; K]);
            let best_dist = &mut *(best_dist as *mut Self::Output as *mut f32);
            let query_ptr = query_wide.as_ptr();
            let mut tile_base = arena.as_ptr();
            let mut remaining = arena.len();

            while remaining != 0 {
                let tile_len = crate::leaf_view::leaf_arena_tile_len(remaining);
                crate::leaf_view_chunked::nearest_one::avx512::nearest_one_avx512_arena_unchecked_f32::<
                    Self::Avx512F32Ops,
                    T,
                    K,
                >(tile_base, tile_len, query_ptr, best_dist, best_item);
                let tile_bytes =
                    K * tile_len * std::mem::size_of::<f32>() + tile_len * std::mem::size_of::<T>();
                tile_base = tile_base.add(tile_bytes);
                remaining -= tile_len;
            }

            return true;
        }

        false
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
    #[inline(always)]
    /// Try an AVX512-specialized `nearest_n_within` leaf kernel.
    unsafe fn try_nearest_n_within_leaf_avx512<
        T,
        R,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    >(
        leaf: &LeafView<'_, A, T, K, B>,
        query_wide: &[Self::Output; K],
        max_dist: Self::Output,
        results: &mut R,
    ) -> bool
    where
        A: Axis<Coord = A> + 'static,
        T: crate::Content,
        Self::Output: Axis<Coord = Self::Output> + 'static,
        R: ResultCollection<Self::Output, QueryResultItem<(), T, Self::Output>>,
    {
        if TypeId::of::<A>() == TypeId::of::<f64>()
            && TypeId::of::<Self::Output>() == TypeId::of::<f64>()
            && TypeId::of::<Self::Avx512F64Ops>()
                != TypeId::of::<distance_metric_avx512::UnsupportedAvx512F64LeafOps>()
        {
            let leaf =
                &*(leaf as *const LeafView<'_, A, T, K, B> as *const LeafView<'_, f64, T, K, B>);
            let query_wide = &*(query_wide as *const [Self::Output; K] as *const [f64; K]);
            let max_dist = *(&max_dist as *const Self::Output as *const f64);
            with_nearest_result_emitter!(results, f64, T, emit, {
                crate::leaf_view_chunked::nearest_n_within::avx512::nearest_n_within_avx512_unchecked::<
                    Self::Avx512F64Ops,
                    T,
                    _,
                    EXCLUSIVE,
                    K,
                    B,
                >(leaf, query_wide, max_dist, &mut emit);
            });

            return true;
        }

        if TypeId::of::<A>() == TypeId::of::<f32>()
            && TypeId::of::<Self::Output>() == TypeId::of::<f32>()
            && TypeId::of::<Self::Avx512F32Ops>()
                != TypeId::of::<distance_metric_avx512::UnsupportedAvx512F32LeafOps>()
        {
            let leaf =
                &*(leaf as *const LeafView<'_, A, T, K, B> as *const LeafView<'_, f32, T, K, B>);
            let query_wide = &*(query_wide as *const [Self::Output; K] as *const [f32; K]);
            let max_dist = *(&max_dist as *const Self::Output as *const f32);
            with_nearest_result_emitter!(results, f32, T, emit, {
                crate::leaf_view_chunked::nearest_n_within::avx512::nearest_n_within_avx512_unchecked_f32::<
                    Self::Avx512F32Ops,
                    T,
                    _,
                    EXCLUSIVE,
                    K,
                    B,
                >(leaf, query_wide, max_dist, &mut emit);
            });

            return true;
        }

        false
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
    #[inline(always)]
    /// Try an AVX512-specialized `nearest_n_within` arena kernel.
    unsafe fn try_nearest_n_within_arena_avx512<T, R, const EXCLUSIVE: bool, const K: usize>(
        arena: &LeafArena<'_, A, T, K>,
        query_wide: &[Self::Output; K],
        max_dist: Self::Output,
        results: &mut R,
    ) -> bool
    where
        A: Axis<Coord = A> + 'static,
        T: crate::Content,
        Self::Output: Axis<Coord = Self::Output> + 'static,
        R: ResultCollection<Self::Output, QueryResultItem<(), T, Self::Output>>,
    {
        if TypeId::of::<A>() == TypeId::of::<f64>()
            && TypeId::of::<Self::Output>() == TypeId::of::<f64>()
            && TypeId::of::<Self::Avx512F64Ops>()
                != TypeId::of::<distance_metric_avx512::UnsupportedAvx512F64LeafOps>()
        {
            let query_wide = &*(query_wide as *const [Self::Output; K] as *const [f64; K]);
            let max_dist = *(&max_dist as *const Self::Output as *const f64);
            with_nearest_result_emitter!(results, f64, T, emit, {
                let mut tile_base = arena.as_ptr();
                let mut remaining = arena.len();

                while remaining != 0 {
                    let tile_len = crate::leaf_view::leaf_arena_tile_len(remaining);
                    crate::leaf_view_chunked::nearest_n_within::avx512::nearest_n_within_avx512_arena_unchecked::<
                        Self::Avx512F64Ops,
                        T,
                        _,
                        EXCLUSIVE,
                        K,
                    >(tile_base, tile_len, query_wide, max_dist, &mut emit);
                    let tile_bytes = K * tile_len * std::mem::size_of::<f64>()
                        + tile_len * std::mem::size_of::<T>();
                    tile_base = tile_base.add(tile_bytes);
                    remaining -= tile_len;
                }
            });

            return true;
        }

        if TypeId::of::<A>() == TypeId::of::<f32>()
            && TypeId::of::<Self::Output>() == TypeId::of::<f32>()
            && TypeId::of::<Self::Avx512F32Ops>()
                != TypeId::of::<distance_metric_avx512::UnsupportedAvx512F32LeafOps>()
        {
            let query_wide = &*(query_wide as *const [Self::Output; K] as *const [f32; K]);
            let max_dist = *(&max_dist as *const Self::Output as *const f32);
            with_nearest_result_emitter!(results, f32, T, emit, {
                let mut tile_base = arena.as_ptr();
                let mut remaining = arena.len();

                while remaining != 0 {
                    let tile_len = crate::leaf_view::leaf_arena_tile_len(remaining);
                    crate::leaf_view_chunked::nearest_n_within::avx512::nearest_n_within_avx512_arena_unchecked_f32::<
                        Self::Avx512F32Ops,
                        T,
                        _,
                        EXCLUSIVE,
                        K,
                    >(tile_base, tile_len, query_wide, max_dist, &mut emit);
                    let tile_bytes = K * tile_len * std::mem::size_of::<f32>()
                        + tile_len * std::mem::size_of::<T>();
                    tile_base = tile_base.add(tile_bytes);
                    remaining -= tile_len;
                }
            });

            return true;
        }

        false
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
    #[inline(always)]
    /// Try an AVX512-specialized `best_n_within` leaf kernel.
    unsafe fn try_best_n_within_leaf_avx512<
        T,
        R,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    >(
        leaf: &LeafView<'_, A, T, K, B>,
        query_wide: &[Self::Output; K],
        max_dist: Self::Output,
        threshold_item: Option<T>,
        results: &mut R,
    ) -> bool
    where
        A: Axis<Coord = A> + 'static,
        T: crate::Content + PartialOrd,
        Self::Output: Axis<Coord = Self::Output> + 'static,
        R: BestNeighbourResultCollection<Self::Output, T>,
    {
        if TypeId::of::<A>() == TypeId::of::<f64>()
            && TypeId::of::<Self::Output>() == TypeId::of::<f64>()
            && TypeId::of::<Self::Avx512F64Ops>()
                != TypeId::of::<distance_metric_avx512::UnsupportedAvx512F64LeafOps>()
        {
            let leaf =
                &*(leaf as *const LeafView<'_, A, T, K, B> as *const LeafView<'_, f64, T, K, B>);
            let query_wide = &*(query_wide as *const [Self::Output; K] as *const [f64; K]);
            let max_dist = *(&max_dist as *const Self::Output as *const f64);
            with_best_result_emitter!(results, threshold_item, f64, T, emit, {
                crate::leaf_view_chunked::best_n_within::avx512::best_n_within_avx512_unchecked::<
                    Self::Avx512F64Ops,
                    T,
                    _,
                    EXCLUSIVE,
                    K,
                    B,
                >(leaf, query_wide, max_dist, &mut emit);
            });

            return true;
        }

        if TypeId::of::<A>() == TypeId::of::<f32>()
            && TypeId::of::<Self::Output>() == TypeId::of::<f32>()
            && TypeId::of::<Self::Avx512F32Ops>()
                != TypeId::of::<distance_metric_avx512::UnsupportedAvx512F32LeafOps>()
        {
            let leaf =
                &*(leaf as *const LeafView<'_, A, T, K, B> as *const LeafView<'_, f32, T, K, B>);
            let query_wide = &*(query_wide as *const [Self::Output; K] as *const [f32; K]);
            let max_dist = *(&max_dist as *const Self::Output as *const f32);
            with_best_result_emitter!(results, threshold_item, f32, T, emit, {
                crate::leaf_view_chunked::best_n_within::avx512::best_n_within_avx512_unchecked_f32::<
                    Self::Avx512F32Ops,
                    T,
                    _,
                    EXCLUSIVE,
                    K,
                    B,
                >(leaf, query_wide, max_dist, &mut emit);
            });

            return true;
        }

        false
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
    #[inline(always)]
    /// Try an AVX512-specialized `best_n_within` arena kernel.
    unsafe fn try_best_n_within_arena_avx512<T, R, const EXCLUSIVE: bool, const K: usize>(
        arena: &LeafArena<'_, A, T, K>,
        query_wide: &[Self::Output; K],
        max_dist: Self::Output,
        threshold_item: Option<T>,
        results: &mut R,
    ) -> bool
    where
        A: Axis<Coord = A> + 'static,
        T: crate::Content + PartialOrd,
        Self::Output: Axis<Coord = Self::Output> + 'static,
        R: BestNeighbourResultCollection<Self::Output, T>,
    {
        if TypeId::of::<A>() == TypeId::of::<f64>()
            && TypeId::of::<Self::Output>() == TypeId::of::<f64>()
            && TypeId::of::<Self::Avx512F64Ops>()
                != TypeId::of::<distance_metric_avx512::UnsupportedAvx512F64LeafOps>()
        {
            let query_wide = &*(query_wide as *const [Self::Output; K] as *const [f64; K]);
            let max_dist = *(&max_dist as *const Self::Output as *const f64);
            with_best_result_emitter!(results, threshold_item, f64, T, emit, {
                let mut tile_base = arena.as_ptr();
                let mut remaining = arena.len();

                while remaining != 0 {
                    let tile_len = crate::leaf_view::leaf_arena_tile_len(remaining);
                    crate::leaf_view_chunked::best_n_within::avx512::best_n_within_avx512_arena_unchecked::<
                        Self::Avx512F64Ops,
                        T,
                        _,
                        EXCLUSIVE,
                        K,
                    >(tile_base, tile_len, query_wide, max_dist, &mut emit);
                    let tile_bytes = K * tile_len * std::mem::size_of::<f64>()
                        + tile_len * std::mem::size_of::<T>();
                    tile_base = tile_base.add(tile_bytes);
                    remaining -= tile_len;
                }
            });

            return true;
        }

        if TypeId::of::<A>() == TypeId::of::<f32>()
            && TypeId::of::<Self::Output>() == TypeId::of::<f32>()
            && TypeId::of::<Self::Avx512F32Ops>()
                != TypeId::of::<distance_metric_avx512::UnsupportedAvx512F32LeafOps>()
        {
            let query_wide = &*(query_wide as *const [Self::Output; K] as *const [f32; K]);
            let max_dist = *(&max_dist as *const Self::Output as *const f32);
            with_best_result_emitter!(results, threshold_item, f32, T, emit, {
                let mut tile_base = arena.as_ptr();
                let mut remaining = arena.len();

                while remaining != 0 {
                    let tile_len = crate::leaf_view::leaf_arena_tile_len(remaining);
                    crate::leaf_view_chunked::best_n_within::avx512::best_n_within_avx512_arena_unchecked_f32::<
                        Self::Avx512F32Ops,
                        T,
                        _,
                        EXCLUSIVE,
                        K,
                    >(tile_base, tile_len, query_wide, max_dist, &mut emit);
                    let tile_bytes = K * tile_len * std::mem::size_of::<f32>()
                        + tile_len * std::mem::size_of::<T>();
                    tile_base = tile_base.add(tile_bytes);
                    remaining -= tile_len;
                }
            });

            return true;
        }

        false
    }
}

/// AVX2 extension hooks.
#[doc(hidden)]
pub trait DistanceMetricAvx2<A: Copy>: DistanceMetricCore<A> {
    /// Whether a specialized AVX2 path is provided by this metric impl.
    const HAS_AVX2_SPECIALIZATION: bool = false;

    /// Type that provides implementations of the AVX2 f64 leaf ops.
    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
    type Avx2F64Ops: distance_metric_avx2::Avx2F64LeafOps + 'static;

    /// Type that provides implementations of the AVX2 f32 leaf ops.
    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
    type Avx2F32Ops: distance_metric_avx2::Avx2F32LeafOps + 'static;

    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
    #[inline(always)]
    /// Try an AVX2-specialized `nearest_n_within` leaf kernel.
    unsafe fn try_nearest_n_within_leaf_avx2<
        T,
        R,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    >(
        leaf: &LeafView<'_, A, T, K, B>,
        query_wide: &[Self::Output; K],
        max_dist: Self::Output,
        results: &mut R,
    ) -> bool
    where
        A: Axis<Coord = A> + 'static,
        T: crate::Content,
        Self::Output: Axis<Coord = Self::Output> + 'static,
        R: ResultCollection<Self::Output, QueryResultItem<(), T, Self::Output>>,
    {
        if TypeId::of::<A>() == TypeId::of::<f64>()
            && TypeId::of::<Self::Output>() == TypeId::of::<f64>()
            && TypeId::of::<Self::Avx2F64Ops>()
                != TypeId::of::<distance_metric_avx2::UnsupportedAvx2F64LeafOps>()
        {
            let leaf =
                &*(leaf as *const LeafView<'_, A, T, K, B> as *const LeafView<'_, f64, T, K, B>);
            let query_wide = &*(query_wide as *const [Self::Output; K] as *const [f64; K]);
            let max_dist = *(&max_dist as *const Self::Output as *const f64);
            with_nearest_result_emitter!(results, f64, T, emit, {
                crate::leaf_view_chunked::nearest_n_within::avx2::nearest_n_within_avx2_unchecked_f64::<
                    Self::Avx2F64Ops,
                    T,
                    _,
                    EXCLUSIVE,
                    K,
                    B,
                >(leaf, query_wide, max_dist, &mut emit);
            });

            return true;
        }

        if TypeId::of::<A>() == TypeId::of::<f32>()
            && TypeId::of::<Self::Output>() == TypeId::of::<f32>()
            && TypeId::of::<Self::Avx2F32Ops>()
                != TypeId::of::<distance_metric_avx2::UnsupportedAvx2F32LeafOps>()
        {
            let leaf =
                &*(leaf as *const LeafView<'_, A, T, K, B> as *const LeafView<'_, f32, T, K, B>);
            let query_wide = &*(query_wide as *const [Self::Output; K] as *const [f32; K]);
            let max_dist = *(&max_dist as *const Self::Output as *const f32);
            with_nearest_result_emitter!(results, f32, T, emit, {
                crate::leaf_view_chunked::nearest_n_within::avx2::nearest_n_within_avx2_unchecked_f32::<
                    Self::Avx2F32Ops,
                    T,
                    _,
                    EXCLUSIVE,
                    K,
                    B,
                >(leaf, query_wide, max_dist, &mut emit);
            });

            return true;
        }

        false
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
    #[inline(always)]
    /// Try an AVX2-specialized `nearest_n_within` arena kernel.
    unsafe fn try_nearest_n_within_arena_avx2<T, R, const EXCLUSIVE: bool, const K: usize>(
        arena: &LeafArena<'_, A, T, K>,
        query_wide: &[Self::Output; K],
        max_dist: Self::Output,
        results: &mut R,
    ) -> bool
    where
        A: Axis<Coord = A> + 'static,
        T: crate::Content,
        Self::Output: Axis<Coord = Self::Output> + 'static,
        R: ResultCollection<Self::Output, QueryResultItem<(), T, Self::Output>>,
    {
        if TypeId::of::<A>() == TypeId::of::<f64>()
            && TypeId::of::<Self::Output>() == TypeId::of::<f64>()
            && TypeId::of::<Self::Avx2F64Ops>()
                != TypeId::of::<distance_metric_avx2::UnsupportedAvx2F64LeafOps>()
        {
            let query_wide = &*(query_wide as *const [Self::Output; K] as *const [f64; K]);
            let max_dist = *(&max_dist as *const Self::Output as *const f64);
            with_nearest_result_emitter!(results, f64, T, emit, {
                let mut tile_base = arena.as_ptr();
                let mut remaining = arena.len();

                while remaining != 0 {
                    let tile_len = crate::leaf_view::leaf_arena_tile_len(remaining);
                    crate::leaf_view_chunked::nearest_n_within::avx2::nearest_n_within_avx2_arena_unchecked_f64::<
                        Self::Avx2F64Ops,
                        T,
                        _,
                        EXCLUSIVE,
                        K,
                    >(tile_base, tile_len, query_wide, max_dist, &mut emit);
                    let tile_bytes = K * tile_len * std::mem::size_of::<f64>()
                        + tile_len * std::mem::size_of::<T>();
                    tile_base = tile_base.add(tile_bytes);
                    remaining -= tile_len;
                }
            });

            return true;
        }

        if TypeId::of::<A>() == TypeId::of::<f32>()
            && TypeId::of::<Self::Output>() == TypeId::of::<f32>()
            && TypeId::of::<Self::Avx2F32Ops>()
                != TypeId::of::<distance_metric_avx2::UnsupportedAvx2F32LeafOps>()
        {
            let query_wide = &*(query_wide as *const [Self::Output; K] as *const [f32; K]);
            let max_dist = *(&max_dist as *const Self::Output as *const f32);
            with_nearest_result_emitter!(results, f32, T, emit, {
                let mut tile_base = arena.as_ptr();
                let mut remaining = arena.len();

                while remaining != 0 {
                    let tile_len = crate::leaf_view::leaf_arena_tile_len(remaining);
                    crate::leaf_view_chunked::nearest_n_within::avx2::nearest_n_within_avx2_arena_unchecked_f32::<
                        Self::Avx2F32Ops,
                        T,
                        _,
                        EXCLUSIVE,
                        K,
                    >(tile_base, tile_len, query_wide, max_dist, &mut emit);
                    let tile_bytes = K * tile_len * std::mem::size_of::<f32>()
                        + tile_len * std::mem::size_of::<T>();
                    tile_base = tile_base.add(tile_bytes);
                    remaining -= tile_len;
                }
            });

            return true;
        }

        false
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
    #[inline(always)]
    /// Try an AVX2-specialized `best_n_within` leaf kernel.
    unsafe fn try_best_n_within_leaf_avx2<
        T,
        R,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    >(
        leaf: &LeafView<'_, A, T, K, B>,
        query_wide: &[Self::Output; K],
        max_dist: Self::Output,
        threshold_item: Option<T>,
        results: &mut R,
    ) -> bool
    where
        A: Axis<Coord = A> + 'static,
        T: crate::Content + PartialOrd,
        Self::Output: Axis<Coord = Self::Output> + 'static,
        R: BestNeighbourResultCollection<Self::Output, T>,
    {
        if TypeId::of::<A>() == TypeId::of::<f64>()
            && TypeId::of::<Self::Output>() == TypeId::of::<f64>()
            && TypeId::of::<Self::Avx2F64Ops>()
                != TypeId::of::<distance_metric_avx2::UnsupportedAvx2F64LeafOps>()
        {
            let leaf =
                &*(leaf as *const LeafView<'_, A, T, K, B> as *const LeafView<'_, f64, T, K, B>);
            let query_wide = &*(query_wide as *const [Self::Output; K] as *const [f64; K]);
            let max_dist = *(&max_dist as *const Self::Output as *const f64);
            with_best_result_emitter!(results, threshold_item, f64, T, emit, {
                crate::leaf_view_chunked::best_n_within::avx2::best_n_within_avx2_unchecked_f64::<
                    Self::Avx2F64Ops,
                    T,
                    _,
                    EXCLUSIVE,
                    K,
                    B,
                >(leaf, query_wide, max_dist, &mut emit);
            });

            return true;
        }

        if TypeId::of::<A>() == TypeId::of::<f32>()
            && TypeId::of::<Self::Output>() == TypeId::of::<f32>()
            && TypeId::of::<Self::Avx2F32Ops>()
                != TypeId::of::<distance_metric_avx2::UnsupportedAvx2F32LeafOps>()
        {
            let leaf =
                &*(leaf as *const LeafView<'_, A, T, K, B> as *const LeafView<'_, f32, T, K, B>);
            let query_wide = &*(query_wide as *const [Self::Output; K] as *const [f32; K]);
            let max_dist = *(&max_dist as *const Self::Output as *const f32);
            with_best_result_emitter!(results, threshold_item, f32, T, emit, {
                crate::leaf_view_chunked::best_n_within::avx2::best_n_within_avx2_unchecked_f32::<
                    Self::Avx2F32Ops,
                    T,
                    _,
                    EXCLUSIVE,
                    K,
                    B,
                >(leaf, query_wide, max_dist, &mut emit);
            });

            return true;
        }

        false
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
    #[inline(always)]
    /// Try an AVX2-specialized `best_n_within` arena kernel.
    unsafe fn try_best_n_within_arena_avx2<T, R, const EXCLUSIVE: bool, const K: usize>(
        arena: &LeafArena<'_, A, T, K>,
        query_wide: &[Self::Output; K],
        max_dist: Self::Output,
        threshold_item: Option<T>,
        results: &mut R,
    ) -> bool
    where
        A: Axis<Coord = A> + 'static,
        T: crate::Content + PartialOrd,
        Self::Output: Axis<Coord = Self::Output> + 'static,
        R: BestNeighbourResultCollection<Self::Output, T>,
    {
        if TypeId::of::<A>() == TypeId::of::<f64>()
            && TypeId::of::<Self::Output>() == TypeId::of::<f64>()
            && TypeId::of::<Self::Avx2F64Ops>()
                != TypeId::of::<distance_metric_avx2::UnsupportedAvx2F64LeafOps>()
        {
            let query_wide = &*(query_wide as *const [Self::Output; K] as *const [f64; K]);
            let max_dist = *(&max_dist as *const Self::Output as *const f64);
            with_best_result_emitter!(results, threshold_item, f64, T, emit, {
                let mut tile_base = arena.as_ptr();
                let mut remaining = arena.len();

                while remaining != 0 {
                    let tile_len = crate::leaf_view::leaf_arena_tile_len(remaining);
                    crate::leaf_view_chunked::best_n_within::avx2::best_n_within_avx2_arena_unchecked_f64::<
                        Self::Avx2F64Ops,
                        T,
                        _,
                        EXCLUSIVE,
                        K,
                    >(tile_base, tile_len, query_wide, max_dist, &mut emit);
                    let tile_bytes = K * tile_len * std::mem::size_of::<f64>()
                        + tile_len * std::mem::size_of::<T>();
                    tile_base = tile_base.add(tile_bytes);
                    remaining -= tile_len;
                }
            });

            return true;
        }

        if TypeId::of::<A>() == TypeId::of::<f32>()
            && TypeId::of::<Self::Output>() == TypeId::of::<f32>()
            && TypeId::of::<Self::Avx2F32Ops>()
                != TypeId::of::<distance_metric_avx2::UnsupportedAvx2F32LeafOps>()
        {
            let query_wide = &*(query_wide as *const [Self::Output; K] as *const [f32; K]);
            let max_dist = *(&max_dist as *const Self::Output as *const f32);
            with_best_result_emitter!(results, threshold_item, f32, T, emit, {
                let mut tile_base = arena.as_ptr();
                let mut remaining = arena.len();

                while remaining != 0 {
                    let tile_len = crate::leaf_view::leaf_arena_tile_len(remaining);
                    crate::leaf_view_chunked::best_n_within::avx2::best_n_within_avx2_arena_unchecked_f32::<
                        Self::Avx2F32Ops,
                        T,
                        _,
                        EXCLUSIVE,
                        K,
                    >(tile_base, tile_len, query_wide, max_dist, &mut emit);
                    let tile_bytes = K * tile_len * std::mem::size_of::<f32>()
                        + tile_len * std::mem::size_of::<T>();
                    tile_base = tile_base.add(tile_bytes);
                    remaining -= tile_len;
                }
            });

            return true;
        }

        false
    }
}

/// NEON extension hooks.
#[doc(hidden)]
pub trait DistanceMetricNeon<A: Copy>: DistanceMetricCore<A> {
    /// Whether a specialized NEON path is provided by this metric impl.
    const HAS_NEON_SPECIALIZATION: bool = false;

    /// Type that provides implementations of the NEON f64 leaf ops.
    #[cfg(all(feature = "simd", target_arch = "aarch64", target_feature = "neon"))]
    type NeonF64Ops: distance_metric_neon::NeonF64LeafOps + 'static;

    /// Type that provides implementations of the NEON f32 leaf ops.
    #[cfg(all(feature = "simd", target_arch = "aarch64", target_feature = "neon"))]
    type NeonF32Ops: distance_metric_neon::NeonF32LeafOps + 'static;

    #[cfg(all(feature = "simd", target_arch = "aarch64", target_feature = "neon"))]
    #[inline(always)]
    /// Try a NEON-specialized `nearest_n_within` leaf kernel.
    unsafe fn try_nearest_n_within_leaf_neon<
        T,
        R,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    >(
        leaf: &LeafView<'_, A, T, K, B>,
        query_wide: &[Self::Output; K],
        max_dist: Self::Output,
        results: &mut R,
    ) -> bool
    where
        A: Axis<Coord = A> + 'static,
        T: crate::Content,
        Self::Output: Axis<Coord = Self::Output> + 'static,
        R: ResultCollection<Self::Output, QueryResultItem<(), T, Self::Output>>,
    {
        if TypeId::of::<A>() == TypeId::of::<f64>()
            && TypeId::of::<Self::Output>() == TypeId::of::<f64>()
            && TypeId::of::<Self::NeonF64Ops>()
                != TypeId::of::<distance_metric_neon::UnsupportedNeonF64LeafOps>()
        {
            let leaf =
                &*(leaf as *const LeafView<'_, A, T, K, B> as *const LeafView<'_, f64, T, K, B>);
            let query_wide = &*(query_wide as *const [Self::Output; K] as *const [f64; K]);
            let max_dist = *(&max_dist as *const Self::Output as *const f64);
            with_nearest_result_emitter!(results, f64, T, emit, {
                crate::leaf_view_chunked::nearest_n_within::neon::nearest_n_within_neon_unchecked_f64::<
                    Self::NeonF64Ops,
                    T,
                    _,
                    EXCLUSIVE,
                    K,
                    B,
                >(leaf, query_wide, max_dist, &mut emit);
            });

            return true;
        }

        if TypeId::of::<A>() == TypeId::of::<f32>()
            && TypeId::of::<Self::Output>() == TypeId::of::<f32>()
            && TypeId::of::<Self::NeonF32Ops>()
                != TypeId::of::<distance_metric_neon::UnsupportedNeonF32LeafOps>()
        {
            let leaf =
                &*(leaf as *const LeafView<'_, A, T, K, B> as *const LeafView<'_, f32, T, K, B>);
            let query_wide = &*(query_wide as *const [Self::Output; K] as *const [f32; K]);
            let max_dist = *(&max_dist as *const Self::Output as *const f32);
            with_nearest_result_emitter!(results, f32, T, emit, {
                crate::leaf_view_chunked::nearest_n_within::neon::nearest_n_within_neon_unchecked_f32::<
                    Self::NeonF32Ops,
                    T,
                    _,
                    EXCLUSIVE,
                    K,
                    B,
                >(leaf, query_wide, max_dist, &mut emit);
            });

            return true;
        }

        false
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64", target_feature = "neon"))]
    #[inline(always)]
    /// Try a NEON-specialized `nearest_n_within` arena kernel.
    unsafe fn try_nearest_n_within_arena_neon<T, R, const EXCLUSIVE: bool, const K: usize>(
        arena: &LeafArena<'_, A, T, K>,
        query_wide: &[Self::Output; K],
        max_dist: Self::Output,
        results: &mut R,
    ) -> bool
    where
        A: Axis<Coord = A> + 'static,
        T: crate::Content,
        Self::Output: Axis<Coord = Self::Output> + 'static,
        R: ResultCollection<Self::Output, QueryResultItem<(), T, Self::Output>>,
    {
        if TypeId::of::<A>() == TypeId::of::<f64>()
            && TypeId::of::<Self::Output>() == TypeId::of::<f64>()
            && TypeId::of::<Self::NeonF64Ops>()
                != TypeId::of::<distance_metric_neon::UnsupportedNeonF64LeafOps>()
        {
            let query_wide = &*(query_wide as *const [Self::Output; K] as *const [f64; K]);
            let max_dist = *(&max_dist as *const Self::Output as *const f64);
            with_nearest_result_emitter!(results, f64, T, emit, {
                let mut tile_base = arena.as_ptr();
                let mut remaining = arena.len();

                while remaining != 0 {
                    let tile_len = crate::leaf_view::leaf_arena_tile_len(remaining);
                    crate::leaf_view_chunked::nearest_n_within::neon::nearest_n_within_neon_arena_unchecked_f64::<
                        Self::NeonF64Ops,
                        T,
                        _,
                        EXCLUSIVE,
                        K,
                    >(tile_base, tile_len, query_wide, max_dist, &mut emit);
                    let tile_bytes = K * tile_len * std::mem::size_of::<f64>()
                        + tile_len * std::mem::size_of::<T>();
                    tile_base = tile_base.add(tile_bytes);
                    remaining -= tile_len;
                }
            });

            return true;
        }

        if TypeId::of::<A>() == TypeId::of::<f32>()
            && TypeId::of::<Self::Output>() == TypeId::of::<f32>()
            && TypeId::of::<Self::NeonF32Ops>()
                != TypeId::of::<distance_metric_neon::UnsupportedNeonF32LeafOps>()
        {
            let query_wide = &*(query_wide as *const [Self::Output; K] as *const [f32; K]);
            let max_dist = *(&max_dist as *const Self::Output as *const f32);
            with_nearest_result_emitter!(results, f32, T, emit, {
                let mut tile_base = arena.as_ptr();
                let mut remaining = arena.len();

                while remaining != 0 {
                    let tile_len = crate::leaf_view::leaf_arena_tile_len(remaining);
                    crate::leaf_view_chunked::nearest_n_within::neon::nearest_n_within_neon_arena_unchecked_f32::<
                        Self::NeonF32Ops,
                        T,
                        _,
                        EXCLUSIVE,
                        K,
                    >(tile_base, tile_len, query_wide, max_dist, &mut emit);
                    let tile_bytes = K * tile_len * std::mem::size_of::<f32>()
                        + tile_len * std::mem::size_of::<T>();
                    tile_base = tile_base.add(tile_bytes);
                    remaining -= tile_len;
                }
            });

            return true;
        }

        false
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64", target_feature = "neon"))]
    #[inline(always)]
    /// Try a NEON-specialized `best_n_within` leaf kernel.
    unsafe fn try_best_n_within_leaf_neon<
        T,
        R,
        const EXCLUSIVE: bool,
        const K: usize,
        const B: usize,
    >(
        leaf: &LeafView<'_, A, T, K, B>,
        query_wide: &[Self::Output; K],
        max_dist: Self::Output,
        threshold_item: Option<T>,
        results: &mut R,
    ) -> bool
    where
        A: Axis<Coord = A> + 'static,
        T: crate::Content + PartialOrd,
        Self::Output: Axis<Coord = Self::Output> + 'static,
        R: BestNeighbourResultCollection<Self::Output, T>,
    {
        if TypeId::of::<A>() == TypeId::of::<f64>()
            && TypeId::of::<Self::Output>() == TypeId::of::<f64>()
            && TypeId::of::<Self::NeonF64Ops>()
                != TypeId::of::<distance_metric_neon::UnsupportedNeonF64LeafOps>()
        {
            let leaf =
                &*(leaf as *const LeafView<'_, A, T, K, B> as *const LeafView<'_, f64, T, K, B>);
            let query_wide = &*(query_wide as *const [Self::Output; K] as *const [f64; K]);
            let max_dist = *(&max_dist as *const Self::Output as *const f64);
            with_best_result_emitter!(results, threshold_item, f64, T, emit, {
                crate::leaf_view_chunked::best_n_within::neon::best_n_within_neon_unchecked_f64::<
                    Self::NeonF64Ops,
                    T,
                    _,
                    EXCLUSIVE,
                    K,
                    B,
                >(leaf, query_wide, max_dist, &mut emit);
            });

            return true;
        }

        if TypeId::of::<A>() == TypeId::of::<f32>()
            && TypeId::of::<Self::Output>() == TypeId::of::<f32>()
            && TypeId::of::<Self::NeonF32Ops>()
                != TypeId::of::<distance_metric_neon::UnsupportedNeonF32LeafOps>()
        {
            let leaf =
                &*(leaf as *const LeafView<'_, A, T, K, B> as *const LeafView<'_, f32, T, K, B>);
            let query_wide = &*(query_wide as *const [Self::Output; K] as *const [f32; K]);
            let max_dist = *(&max_dist as *const Self::Output as *const f32);
            with_best_result_emitter!(results, threshold_item, f32, T, emit, {
                crate::leaf_view_chunked::best_n_within::neon::best_n_within_neon_unchecked_f32::<
                    Self::NeonF32Ops,
                    T,
                    _,
                    EXCLUSIVE,
                    K,
                    B,
                >(leaf, query_wide, max_dist, &mut emit);
            });

            return true;
        }

        false
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64", target_feature = "neon"))]
    #[inline(always)]
    /// Try a NEON-specialized `best_n_within` arena kernel.
    unsafe fn try_best_n_within_arena_neon<T, R, const EXCLUSIVE: bool, const K: usize>(
        arena: &LeafArena<'_, A, T, K>,
        query_wide: &[Self::Output; K],
        max_dist: Self::Output,
        threshold_item: Option<T>,
        results: &mut R,
    ) -> bool
    where
        A: Axis<Coord = A> + 'static,
        T: crate::Content + PartialOrd,
        Self::Output: Axis<Coord = Self::Output> + 'static,
        R: BestNeighbourResultCollection<Self::Output, T>,
    {
        if TypeId::of::<A>() == TypeId::of::<f64>()
            && TypeId::of::<Self::Output>() == TypeId::of::<f64>()
            && TypeId::of::<Self::NeonF64Ops>()
                != TypeId::of::<distance_metric_neon::UnsupportedNeonF64LeafOps>()
        {
            let query_wide = &*(query_wide as *const [Self::Output; K] as *const [f64; K]);
            let max_dist = *(&max_dist as *const Self::Output as *const f64);
            with_best_result_emitter!(results, threshold_item, f64, T, emit, {
                let mut tile_base = arena.as_ptr();
                let mut remaining = arena.len();

                while remaining != 0 {
                    let tile_len = crate::leaf_view::leaf_arena_tile_len(remaining);
                    crate::leaf_view_chunked::best_n_within::neon::best_n_within_neon_arena_unchecked_f64::<
                        Self::NeonF64Ops,
                        T,
                        _,
                        EXCLUSIVE,
                        K,
                    >(tile_base, tile_len, query_wide, max_dist, &mut emit);
                    let tile_bytes = K * tile_len * std::mem::size_of::<f64>()
                        + tile_len * std::mem::size_of::<T>();
                    tile_base = tile_base.add(tile_bytes);
                    remaining -= tile_len;
                }
            });

            return true;
        }

        if TypeId::of::<A>() == TypeId::of::<f32>()
            && TypeId::of::<Self::Output>() == TypeId::of::<f32>()
            && TypeId::of::<Self::NeonF32Ops>()
                != TypeId::of::<distance_metric_neon::UnsupportedNeonF32LeafOps>()
        {
            let query_wide = &*(query_wide as *const [Self::Output; K] as *const [f32; K]);
            let max_dist = *(&max_dist as *const Self::Output as *const f32);
            with_best_result_emitter!(results, threshold_item, f32, T, emit, {
                let mut tile_base = arena.as_ptr();
                let mut remaining = arena.len();

                while remaining != 0 {
                    let tile_len = crate::leaf_view::leaf_arena_tile_len(remaining);
                    crate::leaf_view_chunked::best_n_within::neon::best_n_within_neon_arena_unchecked_f32::<
                        Self::NeonF32Ops,
                        T,
                        _,
                        EXCLUSIVE,
                        K,
                    >(tile_base, tile_len, query_wide, max_dist, &mut emit);
                    let tile_bytes = K * tile_len * std::mem::size_of::<f32>()
                        + tile_len * std::mem::size_of::<T>();
                    tile_base = tile_base.add(tile_bytes);
                    remaining -= tile_len;
                }
            });

            return true;
        }

        false
    }
}

/// trait representing a DistanceMetric that can be used in a `KdTree` query.
#[doc(hidden)]
pub trait DistanceMetric<A: Copy>:
    DistanceMetricCore<A> + DistanceMetricAvx512<A> + DistanceMetricAvx2<A> + DistanceMetricNeon<A>
{
}

impl<T, A: Copy> DistanceMetric<A> for T where
    T: DistanceMetricCore<A>
        + DistanceMetricAvx512<A>
        + DistanceMetricAvx2<A>
        + DistanceMetricNeon<A>
{
}

/// V3-facing metric contract used by kd-tree query paths.
#[doc(hidden)]
pub trait DistanceMetricSimdBlock<A: Copy, const K: usize>:
    DistanceMetric<A>
    + DistanceMetricSimdBlock3<A, K, <Self as DistanceMetricCore<A>>::Output>
    + DistanceMetricSimdBlock4<A, K, <Self as DistanceMetricCore<A>>::Output>
{
}

impl<T, A: Copy, const K: usize> DistanceMetricSimdBlock<A, K> for T
where
    T: DistanceMetric<A>
        + DistanceMetricSimdBlock3<A, K, <T as DistanceMetricCore<A>>::Output>
        + DistanceMetricSimdBlock4<A, K, <T as DistanceMetricCore<A>>::Output>,
    <T as DistanceMetricCore<A>>::Output: Axis<Coord = <T as DistanceMetricCore<A>>::Output>,
{
}

#[cfg(test)]
mod tests {
    use super::{
        Chebyshev, DistanceMetric, DistanceMetricCore, DotProduct, Manhattan, Minkowski,
        SquaredEuclidean,
    };

    #[test]
    fn v3_squared_euclidean_f64_works() {
        type M = SquaredEuclidean<f64>;
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 2.0, -1.0];

        let aw = a.map(M::widen_coord);
        let bw = b.map(M::widen_coord);
        let d = <M as DistanceMetricCore<f64>>::dist::<3>(&aw, &bw);
        assert_eq!(d, 25.0);
    }

    #[test]
    fn v3_manhattan_f64_works() {
        type M = Manhattan<f64>;
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 2.0, -1.0];

        let aw = a.map(M::widen_coord);
        let bw = b.map(M::widen_coord);
        let d = <M as DistanceMetricCore<f64>>::dist::<3>(&aw, &bw);
        assert_eq!(d, 7.0);
    }

    #[test]
    fn v3_chebyshev_f64_works() {
        type M = Chebyshev<f64>;
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 2.5, -1.0];

        let aw = a.map(M::widen_coord);
        let bw = b.map(M::widen_coord);
        let d = <M as DistanceMetricCore<f64>>::dist::<3>(&aw, &bw);
        assert_eq!(d, 4.0);
    }

    #[test]
    fn v3_minkowski_3_f64_works() {
        type M = Minkowski<3, f64>;
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 2.5, -1.0];

        let aw = a.map(M::widen_coord);
        let bw = b.map(M::widen_coord);
        let d = <M as DistanceMetricCore<f64>>::dist::<3>(&aw, &bw);
        assert_eq!(d, 91.125);
    }

    #[test]
    fn v3_unified_bound_is_satisfied() {
        fn assert_unified<M: DistanceMetric<f64>>() {}
        assert_unified::<SquaredEuclidean<f64>>();
        assert_unified::<Chebyshev<f64>>();
        assert_unified::<Minkowski<3, f64>>();
    }

    #[test]
    fn v3_dot_product_f64_works() {
        type M = DotProduct<f64>;
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 2.0, -1.0];

        let aw = a.map(M::widen_coord);
        let bw = b.map(M::widen_coord);
        let d = <M as DistanceMetricCore<f64>>::dist::<3>(&aw, &bw);
        assert_eq!(d, 5.0);
    }
}
