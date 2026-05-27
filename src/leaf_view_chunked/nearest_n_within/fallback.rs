use crate::dist::KdTreeDistanceMetric;
use crate::leaf_view::{LeafArena, LeafView, TlsLeafScratch};
use crate::results::result_collection::ResultCollection;
use crate::{Axis, Content, NearestNeighbour};

#[inline(always)]
pub(crate) fn nearest_n_within_with_query_wide_fallback<
    AX,
    T,
    D,
    R,
    const EXCLUSIVE: bool,
    const K: usize,
    const B: usize,
>(
    leaf: &LeafView<'_, AX, T, K, B>,
    query_wide: &[D::Output; K],
    dist: D::Output,
    results: &mut R,
) where
    AX: Axis<Coord = AX> + 'static,
    T: Content,
    D: KdTreeDistanceMetric<AX, K>,
    D::Output: Axis<Coord = D::Output> + TlsLeafScratch + 'static,
    R: ResultCollection<D::Output, NearestNeighbour<D::Output, T>>,
{
    leaf.with_dists_for_slice_wide::<D, _>(query_wide, |dists| {
        LeafView::<AX, T, K, B>::update_nearest_dists::<_, _, EXCLUSIVE>(
            dists,
            leaf.items(),
            dist,
            results,
        );
    });
}

#[inline(always)]
pub(crate) fn nearest_n_within_with_query_wide_arena_fallback<
    AX,
    T,
    D,
    R,
    const EXCLUSIVE: bool,
    const K: usize,
>(
    arena: &LeafArena<'_, AX, T, K>,
    query_wide: &[D::Output; K],
    dist: D::Output,
    results: &mut R,
) where
    AX: Axis<Coord = AX> + 'static,
    T: Content,
    D: KdTreeDistanceMetric<AX, K>,
    D::Output: Axis<Coord = D::Output> + 'static,
    R: ResultCollection<D::Output, NearestNeighbour<D::Output, T>>,
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

            let is_within_dist = if EXCLUSIVE {
                candidate_dist < dist
            } else {
                candidate_dist <= dist
            };

            if is_within_dist {
                #[cfg(feature = "result_collection_stats")]
                crate::results::result_collection_stats::record_candidate_emitted();

                let candidate = NearestNeighbour {
                    point: (),
                    distance: candidate_dist,
                    item: unsafe { tile.item_unaligned(idx) },
                };

                results.add(candidate);
            }
        }
    });
}
