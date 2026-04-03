mod fallback;

#[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
mod avx512;

#[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
use std::any::type_name;

#[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
use crate::dist::{Manhattan as V3Manhattan, SquaredEuclidean as V3SquaredEuclidean};
use crate::kd_tree::leaf_view::{LeafArena, LeafView};
use crate::traits_unified_2::{AxisUnified, Basics, DistanceMetricUnified};
#[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
use crate::traits_unified_2::{Manhattan as V2Manhattan, SquaredEuclidean as V2SquaredEuclidean};

// #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
// pub(crate) use avx512::nearest_one_avx512_unchecked;

pub(crate) use fallback::{
    nearest_one_with_query_wide_arena_fallback, nearest_one_with_query_wide_fallback,
};

#[inline(always)]
pub(crate) fn nearest_one_with_query_wide_arena<AX, T, D, const K: usize>(
    arena: &LeafArena<'_, AX, T, K>,
    query_wide: &[D::Output; K],
    best_dist: &mut D::Output,
    best_item: &mut T,
) where
    AX: AxisUnified<Coord = AX> + 'static,
    T: Basics,
    D: DistanceMetricUnified<AX, K>,
    D::Output: AxisUnified<Coord = D::Output> + 'static,
{
    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
    if unsafe {
        try_nearest_one_arena_avx512::<AX, T, D, K>(arena, query_wide, best_dist, best_item)
    } {
        return;
    }

    nearest_one_with_query_wide_arena_fallback::<AX, T, D, K>(
        arena, query_wide, best_dist, best_item,
    );
}

#[inline(always)]
pub(crate) fn nearest_one_with_query_wide<AX, T, D, const K: usize, const B: usize>(
    leaf: &LeafView<'_, AX, T, K, B>,
    query_wide: &[D::Output; K],
    best_dist: &mut D::Output,
    best_item: &mut T,
) where
    AX: AxisUnified<Coord = AX> + 'static,
    T: Basics,
    D: DistanceMetricUnified<AX, K>,
    D::Output: AxisUnified<Coord = D::Output> + 'static,
{
    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
    if unsafe { try_nearest_one_avx512::<AX, T, D, K, B>(leaf, query_wide, best_dist, best_item) } {
        return;
    }

    nearest_one_with_query_wide_fallback::<AX, T, D, K, B>(leaf, query_wide, best_dist, best_item);
}

#[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
#[inline(always)]
unsafe fn try_nearest_one_avx512<AX, T, D, const K: usize, const B: usize>(
    leaf: &LeafView<'_, AX, T, K, B>,
    query_wide: &[D::Output; K],
    best_dist: &mut D::Output,
    best_item: &mut T,
) -> bool
where
    AX: AxisUnified<Coord = AX> + 'static,
    T: Basics,
    D: DistanceMetricUnified<AX, K>,
    D::Output: AxisUnified<Coord = D::Output> + 'static,
{
    if type_name::<AX>() != type_name::<f64>() || type_name::<D::Output>() != type_name::<AX>() {
        return false;
    }

    let leaf = &*(leaf as *const LeafView<'_, AX, T, K, B> as *const LeafView<'_, f64, T, K, B>);
    let query_wide = &*(query_wide as *const [D::Output; K] as *const [f64; K]);
    let best_dist = &mut *(best_dist as *mut D::Output as *mut f64);

    if type_name::<D>() == type_name::<V3SquaredEuclidean<f64>>()
        || type_name::<D>() == type_name::<V2SquaredEuclidean<f64>>()
    {
        avx512::nearest_one_avx512_unchecked::<f64, T, V3SquaredEuclidean<f64>, K, B>(
            leaf, query_wide, best_dist, best_item,
        );
        return true;
    }

    if type_name::<D>() == type_name::<V3Manhattan<f64>>()
        || type_name::<D>() == type_name::<V2Manhattan<f64>>()
    {
        avx512::nearest_one_avx512_unchecked::<f64, T, V3Manhattan<f64>, K, B>(
            leaf, query_wide, best_dist, best_item,
        );
        return true;
    }

    false
}

#[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
#[inline(always)]
unsafe fn try_nearest_one_arena_avx512<AX, T, D, const K: usize>(
    arena: &LeafArena<'_, AX, T, K>,
    query_wide: &[D::Output; K],
    best_dist: &mut D::Output,
    best_item: &mut T,
) -> bool
where
    AX: AxisUnified<Coord = AX> + 'static,
    T: Basics,
    D: DistanceMetricUnified<AX, K>,
    D::Output: AxisUnified<Coord = D::Output> + 'static,
{
    if type_name::<AX>() != type_name::<f64>() || type_name::<D::Output>() != type_name::<AX>() {
        return false;
    }

    let query_wide = &*(query_wide as *const [D::Output; K] as *const [f64; K]);
    let best_dist = &mut *(best_dist as *mut D::Output as *mut f64);

    if type_name::<D>() == type_name::<V3SquaredEuclidean<f64>>()
        || type_name::<D>() == type_name::<V2SquaredEuclidean<f64>>()
    {
        let query_ptr = query_wide.as_ptr();
        let mut tile_base = arena.as_ptr();
        let mut remaining = arena.len();
        while remaining != 0 {
            let tile_len = crate::kd_tree::leaf_view::leaf_arena_tile_len(remaining);
            avx512::nearest_one_avx512_arena_unchecked::<f64, T, V3SquaredEuclidean<f64>, K>(
                tile_base, tile_len, query_ptr, best_dist, best_item,
            );
            let tile_bytes =
                K * tile_len * std::mem::size_of::<f64>() + tile_len * std::mem::size_of::<T>();
            tile_base = tile_base.add(tile_bytes);
            remaining -= tile_len;
        }
        return true;
    }

    if type_name::<D>() == type_name::<V3Manhattan<f64>>()
        || type_name::<D>() == type_name::<V2Manhattan<f64>>()
    {
        let query_ptr = query_wide.as_ptr();
        let mut tile_base = arena.as_ptr();
        let mut remaining = arena.len();
        while remaining != 0 {
            let tile_len = crate::kd_tree::leaf_view::leaf_arena_tile_len(remaining);
            avx512::nearest_one_avx512_arena_unchecked::<f64, T, V3Manhattan<f64>, K>(
                tile_base, tile_len, query_ptr, best_dist, best_item,
            );
            let tile_bytes =
                K * tile_len * std::mem::size_of::<f64>() + tile_len * std::mem::size_of::<T>();
            tile_base = tile_base.add(tile_bytes);
            remaining -= tile_len;
        }
        return true;
    }

    false
}
