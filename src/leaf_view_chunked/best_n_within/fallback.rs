use crate::dist::KdTreeDistanceMetric;
use crate::leaf_view::{LeafArena, LeafView, TlsLeafScratch};
use crate::results::result_collection::BestNeighbourResultCollection;
use crate::{Axis, BestNeighbour, Content};

#[inline(always)]
pub(crate) fn best_n_within_with_query_wide_fallback<AX, T, D, R, const K: usize, const B: usize>(
    leaf: &LeafView<'_, AX, T, K, B>,
    query_wide: &[D::Output; K],
    dist: D::Output,
    threshold_item: Option<T>,
    results: &mut R,
) where
    AX: Axis<Coord = AX> + 'static,
    T: Content + PartialOrd,
    D: KdTreeDistanceMetric<AX, K>,
    D::Output: Axis<Coord = D::Output> + TlsLeafScratch + 'static,
    R: BestNeighbourResultCollection<D::Output, T>,
{
    leaf.with_dists_for_slice_wide::<D, _>(query_wide, |dists| {
        LeafView::<AX, T, K, B>::update_best_dists(
            dists,
            leaf.items(),
            dist,
            threshold_item,
            results,
        );
    });
}

#[inline(always)]
pub(crate) fn best_n_within_with_query_wide_arena_fallback<AX, T, D, R, const K: usize>(
    arena: &LeafArena<'_, AX, T, K>,
    query_wide: &[D::Output; K],
    dist: D::Output,
    threshold_item: Option<T>,
    results: &mut R,
) where
    AX: Axis<Coord = AX> + 'static,
    T: Content + PartialOrd,
    D: KdTreeDistanceMetric<AX, K>,
    D::Output: Axis<Coord = D::Output> + 'static,
    R: BestNeighbourResultCollection<D::Output, T>,
{
    if arena.is_empty() {
        return;
    }

    arena.for_each_tiled_chunk(|tile| {
        for idx in 0..tile.len() {
            let mut candidate_dist = D::Output::zero();

            for dim in 0..K {
                let coord = unsafe { tile.point_unaligned(dim, idx) };
                D::combine_component(
                    &mut candidate_dist,
                    D::dist1(D::widen_coord(coord), unsafe {
                        *query_wide.get_unchecked(dim)
                    }),
                );
            }

            if candidate_dist <= dist {
                let item = unsafe { tile.item_unaligned(idx) };
                if threshold_item.is_some_and(|worst_item| item >= worst_item) {
                    #[cfg(feature = "result_collection_stats")]
                    crate::results::result_collection_stats::record_best_item_threshold_reject();
                    continue;
                }
                #[cfg(feature = "result_collection_stats")]
                crate::results::result_collection_stats::record_candidate_emitted();

                let candidate = BestNeighbour {
                    distance: candidate_dist,
                    item,
                };

                results.add(candidate);
            }
        }
    });
}
