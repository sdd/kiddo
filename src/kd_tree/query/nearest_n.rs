use crate::kd_tree::KdTree;
use crate::traits_unified_2::{AxisUnified, Basics, DistanceMetricUnified, LeafStrategy};
use crate::{NearestNeighbour, StemStrategy};
use std::num::NonZero;

impl<A, T, SS, LS, const K: usize, const B: usize> KdTree<A, T, SS, LS, K, B>
where
    A: AxisUnified<Coord = A>,
    T: Basics + Ord,
    LS: LeafStrategy<A, T, SS, K, B>,
    SS: StemStrategy,
{
    pub fn nearest_n<D>(
        &self,
        query: &[A; K],
        max_qty: NonZero<usize>,
        sorted: bool,
    ) -> Vec<NearestNeighbour<D::Output, T>>
    where
        D: DistanceMetricUnified<A, K>,
    {
        self.nearest_n_within::<D>(query, D::Output::max_value(), max_qty.get(), sorted)
    }
}
