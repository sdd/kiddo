use crate::dist::KdTreeDistanceMetric;
use crate::kd_tree::leaf_view::{LeafArena, LeafView, TlsLeafScratch};
use crate::kd_tree::result_collection::ResultCollection;
use crate::traits_unified_2::{AxisUnified, Basics};
use crate::BestNeighbour;

#[inline(always)]
pub(crate) fn best_n_within_with_query_wide_fallback<AX, T, D, R, const K: usize, const B: usize>(
    leaf: &LeafView<'_, AX, T, K, B>,
    query_wide: &[D::Output; K],
    dist: D::Output,
    results: &mut R,
) where
    AX: AxisUnified<Coord = AX> + 'static,
    T: Basics + Ord,
    D: KdTreeDistanceMetric<AX, K>,
    D::Output: AxisUnified<Coord = D::Output> + TlsLeafScratch + 'static,
    R: ResultCollection<D::Output, BestNeighbour<D::Output, T>>,
{
    leaf.with_dists_for_slice_wide::<D, _>(query_wide, |dists| {
        LeafView::<AX, T, K, B>::update_best_dists(dists, leaf.items(), dist, results);
    });
}

#[inline(always)]
pub(crate) fn best_n_within_with_query_wide_arena_fallback<AX, T, D, R, const K: usize>(
    arena: &LeafArena<'_, AX, T, K>,
    query_wide: &[D::Output; K],
    dist: D::Output,
    results: &mut R,
) where
    AX: AxisUnified<Coord = AX> + 'static,
    T: Basics + Ord,
    D: KdTreeDistanceMetric<AX, K>,
    D::Output: AxisUnified<Coord = D::Output> + 'static,
    R: ResultCollection<D::Output, BestNeighbour<D::Output, T>>,
{
    if arena.is_empty() {
        return;
    }

    #[cfg(feature = "buffered_result_collection")]
    let mut buffer = crate::kd_tree::result_collection::ResultBuffer::new();

    arena.for_each_tiled_chunk(|tile| {
        for idx in 0..tile.len() {
            let mut candidate_dist = D::Output::zero();

            for dim in 0..K {
                let coord = unsafe { tile.point_unaligned(dim, idx) };
                candidate_dist += D::dist1(D::widen_coord(coord), unsafe {
                    *query_wide.get_unchecked(dim)
                });
            }

            if candidate_dist <= dist {
                let candidate = BestNeighbour {
                    distance: candidate_dist,
                    item: unsafe { tile.item_unaligned(idx) },
                };

                #[cfg(feature = "buffered_result_collection")]
                {
                    buffer.push(candidate);
                }

                #[cfg(not(feature = "buffered_result_collection"))]
                {
                    results.add(candidate);
                }
            }
        }
    });

    #[cfg(feature = "buffered_result_collection")]
    crate::kd_tree::result_collection::flush_result_buffer(results, &mut buffer);
}
