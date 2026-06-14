use super::LeafViewChunked;
use crate::dist::DistanceMetric;
use crate::leaf_view::LeafView;
use crate::{AxisUnified, Basics};


// TODO: turn this into a macro so that we can stamp out implementations
//       for all AxisUnified types without a NEON-specific implementation
impl<'a, T, D, const K: usize> LeafViewChunked<'a, f32, T, K>
where
    T: Basics,
    D: DistanceMetric<f32>,
{
    pub fn nearest_one_neon(
        leaf: &LeafView<'_, f32, T, K>,
        query: &[f32; K],
        best_dist: &mut f32,
        best_item: &mut T,
    ) {
        nearest_one_fallback(leaf, query, best_dist, best_item)
    }
}
