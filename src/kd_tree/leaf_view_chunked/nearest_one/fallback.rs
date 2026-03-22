use crate::kd_tree::leaf_view::{try_identity_widen_axis, LeafView};
use crate::traits_unified_2::{AxisUnified, Basics, DistanceMetricUnified};

#[inline(always)]
pub(crate) fn nearest_one_with_query_wide_fallback<AX, T, D, const K: usize, const B: usize>(
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
