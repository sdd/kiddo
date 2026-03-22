use crate::kd_tree::leaf_view::LeafView;
use crate::traits_unified_2::{AxisUnified, Basics, DistanceMetricUnified};
use super::LeafViewChunked;


// TODO: turn this into a macro so that we can stamp out implementations
//       for all AxisUnified types without a NEON-specific implementation
impl<'a, T, D, const K: usize> LeafViewChunked<'a, f32, T, K>
where
    T: Basics,
    D: DistanceMetricUnified<f32, K>,
{
    pub fn nearest_one_neon(
        leaf: &LeafView<'_, f32, T, K>,
        query: &[f32; K],
        best_dist: &mut f32,
        best_item: &mut T,
    ) {
        nearest_one_fallback(
            leaf,
            query,
            best_dist,
            best_item
        )
    }
}
