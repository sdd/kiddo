use crate::kd_tree::traits::QueryContext;
use crate::kd_tree::KdTree;
use crate::traits_unified_2::{AxisUnified, Basics, DistanceMetricUnified, LeafStrategy};
use crate::{BestNeighbour, StemStrategy};
use std::collections::BinaryHeap;
use std::num::NonZero;

impl<A, T, SS, LS, const K: usize, const B: usize> KdTree<A, T, SS, LS, K, B>
where
    A: AxisUnified<Coord = A>,
    T: Basics + Ord,
    LS: LeafStrategy<A, T, SS, K, B>,
    SS: StemStrategy,
{
    pub fn best_n_within<D>(
        &self,
        query: &[A; K],
        max_dist: <D as DistanceMetricUnified<A, K>>::Output,
        max_qty: NonZero<usize>,
    ) -> BinaryHeap<BestNeighbour<<D as DistanceMetricUnified<A, K>>::Output, T>>
    where
        D: DistanceMetricUnified<A, K>,
    {
        let max_qty = max_qty.into();
        let req_ctx = BestNWithinReqCtx {
            query,
            max_dist,
            max_qty,
        };
        let mut results = BinaryHeap::with_capacity(max_qty);

        self.backtracking_query(req_ctx, |leaf| {
            leaf.best_n_within::<D>(query, max_dist, &mut results);
        });

        results
    }
}

#[derive(Debug, Copy, Clone)]
struct BestNWithinReqCtx<'a, A, O, const K: usize> {
    query: &'a [A; K],
    max_dist: O,
    max_qty: usize,
}

impl<'a, A, DOut, const K: usize> QueryContext<A, K> for BestNWithinReqCtx<'a, A, DOut, K> {
    fn query(&self) -> &[A; K] {
        self.query
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kd_tree::leaf_strategies::dummy::DummyLeafStrategy;
    use crate::traits_unified_2::SquaredEuclidean;

    use crate::Eytzinger;

    #[test]
    fn test_get_leaf_idx() {
        let tree: KdTree<f32, u32, Eytzinger<3>, DummyLeafStrategy, 3, 32> = KdTree::default();

        let query = [0.0f32; 3];

        let result = tree
            .best_n_within::<SquaredEuclidean<_>>(&query, 0.5f32, NonZero::new(3).unwrap())
            .len();

        assert_eq!(result, 3);
    }
}
