mod fallback;

#[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
pub(crate) mod avx2;

#[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
pub(crate) mod avx512;

#[cfg(all(feature = "simd", target_arch = "aarch64", target_feature = "neon"))]
pub(crate) mod neon;

use crate::dist::KdTreeDistanceMetric;
use crate::leaf_view::{LeafArena, LeafView, TlsLeafScratch};
use crate::results::result_collection::BestNeighbourResultCollection;
use crate::traits_unified_2::{AxisUnified, Basics};

pub(crate) use fallback::{
    best_n_within_with_query_wide_arena_fallback, best_n_within_with_query_wide_fallback,
};

#[inline(always)]
pub(crate) fn best_n_within_with_query_wide_arena<AX, T, D, R, const K: usize>(
    arena: &LeafArena<'_, AX, T, K>,
    query_wide: &[D::Output; K],
    dist: D::Output,
    threshold_item: Option<T>,
    results: &mut R,
) where
    AX: AxisUnified<Coord = AX> + 'static,
    T: Basics + PartialOrd,
    D: KdTreeDistanceMetric<AX, K>,
    D::Output: AxisUnified<Coord = D::Output> + TlsLeafScratch + 'static,
    R: BestNeighbourResultCollection<D::Output, T>,
{
    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
    if unsafe {
        try_best_n_within_arena_avx512::<AX, T, D, R, K>(
            arena,
            query_wide,
            dist,
            threshold_item,
            results,
        )
    } {
        return;
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
    if unsafe {
        try_best_n_within_arena_avx2::<AX, T, D, R, K>(
            arena,
            query_wide,
            dist,
            threshold_item,
            results,
        )
    } {
        return;
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64", target_feature = "neon"))]
    if unsafe {
        try_best_n_within_arena_neon::<AX, T, D, R, K>(
            arena,
            query_wide,
            dist,
            threshold_item,
            results,
        )
    } {
        return;
    }

    best_n_within_with_query_wide_arena_fallback::<AX, T, D, R, K>(
        arena,
        query_wide,
        dist,
        threshold_item,
        results,
    );
}

#[inline(always)]
pub(crate) fn best_n_within_with_query_wide<AX, T, D, R, const K: usize, const B: usize>(
    leaf: &LeafView<'_, AX, T, K, B>,
    query_wide: &[D::Output; K],
    dist: D::Output,
    threshold_item: Option<T>,
    results: &mut R,
) where
    AX: AxisUnified<Coord = AX> + 'static,
    T: Basics + PartialOrd,
    D: KdTreeDistanceMetric<AX, K>,
    D::Output: AxisUnified<Coord = D::Output> + TlsLeafScratch + 'static,
    R: BestNeighbourResultCollection<D::Output, T>,
{
    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
    if unsafe {
        try_best_n_within_avx512::<AX, T, D, R, K, B>(
            leaf,
            query_wide,
            dist,
            threshold_item,
            results,
        )
    } {
        return;
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
    if unsafe {
        try_best_n_within_avx2::<AX, T, D, R, K, B>(leaf, query_wide, dist, threshold_item, results)
    } {
        return;
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64", target_feature = "neon"))]
    if unsafe {
        try_best_n_within_neon::<AX, T, D, R, K, B>(leaf, query_wide, dist, threshold_item, results)
    } {
        return;
    }

    best_n_within_with_query_wide_fallback::<AX, T, D, R, K, B>(
        leaf,
        query_wide,
        dist,
        threshold_item,
        results,
    );
}

#[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
#[inline(always)]
unsafe fn try_best_n_within_avx512<AX, T, D, R, const K: usize, const B: usize>(
    leaf: &LeafView<'_, AX, T, K, B>,
    query_wide: &[D::Output; K],
    dist: D::Output,
    threshold_item: Option<T>,
    results: &mut R,
) -> bool
where
    AX: AxisUnified<Coord = AX> + 'static,
    T: Basics + PartialOrd,
    D: KdTreeDistanceMetric<AX, K>,
    D::Output: AxisUnified<Coord = D::Output> + 'static,
    R: BestNeighbourResultCollection<D::Output, T>,
{
    D::try_best_n_within_leaf_avx512(leaf, query_wide, dist, threshold_item, results)
}

#[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
#[inline(always)]
unsafe fn try_best_n_within_arena_avx512<AX, T, D, R, const K: usize>(
    arena: &LeafArena<'_, AX, T, K>,
    query_wide: &[D::Output; K],
    dist: D::Output,
    threshold_item: Option<T>,
    results: &mut R,
) -> bool
where
    AX: AxisUnified<Coord = AX> + 'static,
    T: Basics + PartialOrd,
    D: KdTreeDistanceMetric<AX, K>,
    D::Output: AxisUnified<Coord = D::Output> + 'static,
    R: BestNeighbourResultCollection<D::Output, T>,
{
    D::try_best_n_within_arena_avx512(arena, query_wide, dist, threshold_item, results)
}

#[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
unsafe fn try_best_n_within_avx2<AX, T, D, R, const K: usize, const B: usize>(
    leaf: &LeafView<'_, AX, T, K, B>,
    query_wide: &[D::Output; K],
    dist: D::Output,
    threshold_item: Option<T>,
    results: &mut R,
) -> bool
where
    AX: AxisUnified<Coord = AX> + 'static,
    T: Basics + PartialOrd,
    D: KdTreeDistanceMetric<AX, K>,
    D::Output: AxisUnified<Coord = D::Output> + 'static,
    R: BestNeighbourResultCollection<D::Output, T>,
{
    D::try_best_n_within_leaf_avx2(leaf, query_wide, dist, threshold_item, results)
}

#[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
unsafe fn try_best_n_within_arena_avx2<AX, T, D, R, const K: usize>(
    arena: &LeafArena<'_, AX, T, K>,
    query_wide: &[D::Output; K],
    dist: D::Output,
    threshold_item: Option<T>,
    results: &mut R,
) -> bool
where
    AX: AxisUnified<Coord = AX> + 'static,
    T: Basics + PartialOrd,
    D: KdTreeDistanceMetric<AX, K>,
    D::Output: AxisUnified<Coord = D::Output> + 'static,
    R: BestNeighbourResultCollection<D::Output, T>,
{
    D::try_best_n_within_arena_avx2(arena, query_wide, dist, threshold_item, results)
}

#[cfg(all(feature = "simd", target_arch = "aarch64", target_feature = "neon"))]
#[inline(always)]
unsafe fn try_best_n_within_neon<AX, T, D, R, const K: usize, const B: usize>(
    leaf: &LeafView<'_, AX, T, K, B>,
    query_wide: &[D::Output; K],
    dist: D::Output,
    threshold_item: Option<T>,
    results: &mut R,
) -> bool
where
    AX: AxisUnified<Coord = AX> + 'static,
    T: Basics + PartialOrd,
    D: KdTreeDistanceMetric<AX, K>,
    D::Output: AxisUnified<Coord = D::Output> + 'static,
    R: BestNeighbourResultCollection<D::Output, T>,
{
    D::try_best_n_within_leaf_neon(leaf, query_wide, dist, threshold_item, results)
}

#[cfg(all(feature = "simd", target_arch = "aarch64", target_feature = "neon"))]
#[inline(always)]
unsafe fn try_best_n_within_arena_neon<AX, T, D, R, const K: usize>(
    arena: &LeafArena<'_, AX, T, K>,
    query_wide: &[D::Output; K],
    dist: D::Output,
    threshold_item: Option<T>,
    results: &mut R,
) -> bool
where
    AX: AxisUnified<Coord = AX> + 'static,
    T: Basics + PartialOrd,
    D: KdTreeDistanceMetric<AX, K>,
    D::Output: AxisUnified<Coord = D::Output> + 'static,
    R: BestNeighbourResultCollection<D::Output, T>,
{
    D::try_best_n_within_arena_neon(arena, query_wide, dist, threshold_item, results)
}
