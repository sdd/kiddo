mod fallback;

#[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
mod avx512;

use crate::kd_tree::leaf_view::LeafView;
use crate::traits_unified_2::{AxisUnified, Basics, DistanceMetricUnified};

#[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
pub(crate) use avx512::nearest_one_avx512_unchecked;

pub(crate) use fallback::nearest_one_with_query_wide_fallback;

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
    nearest_one_with_query_wide_fallback::<AX, T, D, K, B>(leaf, query_wide, best_dist, best_item);
}
