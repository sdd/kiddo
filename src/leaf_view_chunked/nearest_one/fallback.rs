use crate::dist::DistanceMetricUnified;
use crate::leaf_view::{try_identity_widen_axis, LeafArena, LeafView};
use crate::{Axis, Content};

#[inline(always)]
pub(crate) fn nearest_one_with_query_wide_fallback<AX, T, D, const K: usize, const B: usize>(
    leaf: &LeafView<'_, AX, T, K, B>,
    query_wide: &[D::Output; K],
    best_dist: &mut D::Output,
    best_item: &mut T,
) where
    AX: Axis<Coord = AX> + 'static,
    T: Content,
    D: DistanceMetricUnified<AX>,
    D::Output: Axis<Coord = D::Output> + 'static,
{
    let points = leaf.points();
    let items = leaf.items();

    if items.is_empty() {
        return;
    }

    let widened_points = points.map(try_identity_widen_axis::<AX, D::Output>);
    if widened_points.iter().all(Option::is_some) {
        let widened_points = widened_points.map(|axis| axis.expect("checked is_some above"));
        for idx in 0..items.len() {
            let mut dist = D::Output::zero();
            for dim in 0..K {
                unsafe {
                    dist += D::dist1(
                        *widened_points.get_unchecked(dim).get_unchecked(idx),
                        *query_wide.get_unchecked(dim),
                    );
                }
            }
            if dist < *best_dist {
                *best_dist = dist;
                *best_item = unsafe { *items.get_unchecked(idx) };
            }
        }
        return;
    }

    for idx in 0..items.len() {
        let mut dist = D::Output::zero();

        for dim in 0..K {
            dist += D::dist1(
                D::widen_coord(unsafe { *points.get_unchecked(dim).get_unchecked(idx) }),
                unsafe { *query_wide.get_unchecked(dim) },
            );
        }

        if dist < *best_dist {
            *best_dist = dist;
            *best_item = unsafe { *items.get_unchecked(idx) };
        }
    }
}

#[inline(always)]
pub(crate) fn nearest_one_with_query_wide_arena_fallback<AX, T, D, const K: usize>(
    arena: &LeafArena<'_, AX, T, K>,
    query_wide: &[D::Output; K],
    best_dist: &mut D::Output,
    best_item: &mut T,
) where
    AX: Axis<Coord = AX> + 'static,
    T: Content,
    D: DistanceMetricUnified<AX>,
    D::Output: Axis<Coord = D::Output> + 'static,
{
    if arena.is_empty() {
        return;
    }

    arena.for_each_tiled_chunk(|tile| {
        for idx in 0..tile.len() {
            let mut dist = D::Output::zero();

            for dim in 0..K {
                let coord = unsafe { tile.point_unaligned(dim, idx) };
                dist += D::dist1(D::widen_coord(coord), unsafe {
                    *query_wide.get_unchecked(dim)
                });
            }

            if dist < *best_dist {
                *best_dist = dist;
                *best_item = unsafe { tile.item_unaligned(idx) };
            }
        }
    });
}

#[cfg(feature = "cargo_asm")]
pub mod cargo_asm {
    use super::nearest_one_with_query_wide_arena_fallback;
    use crate::dist::SquaredEuclidean;
    use crate::leaf_view::LeafArena;

    /// Hook for cargo-asm to render the scalar arena nearest-one fallback directly.
    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn v6_nearest_one_with_query_wide_arena_fallback_cargo_asm_hook(
        arena: &LeafArena<'_, f64, usize, 3>,
        query: [f64; 3],
        best_dist: &mut f64,
        best_item: &mut usize,
    ) {
        nearest_one_with_query_wide_arena_fallback::<f64, usize, SquaredEuclidean<f64>, 3>(
            arena, &query, best_dist, best_item,
        );
    }
}
