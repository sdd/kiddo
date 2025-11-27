use std::num::NonZero;
use std::ops::Sub;
use crate::kd_tree::KdTree;
use crate::kd_tree::traits::QueryContext;
use crate::mutable::float::result_collection::ResultCollection;
use crate::StemStrategy;
use crate::traits_unified_2::{AxisUnified, Basics, LeafStrategy};

impl<A, T, SS, LS, const K: usize, const B: usize> KdTree<A, T, SS, LS, K, B>
where
    A: AxisUnified<Coord = A>,
    T: Basics + Copy + Default,
    LS: LeafStrategy<A, T, SS, K, B>,
    SS: StemStrategy,
{
    pub fn nearest_n_within<D>(
        &self,
        query: &[A; K],
        max_dist: A,
        max_qty: NonZero<usize>,
        sorted: bool,
    ) -> (A, T) {
        let max_qty = max_qty.into();
        
        let req_ctx = NeaarestNWithinReqCtx {
            query,
            max_dist,
            max_qty,
            sorted
        };

        let mut best_dist = A::max_value();
        let mut best_item = T::default();
        
        self.backtracking_query(
            req_ctx,
            |leaf, _l| {
                // TODO: real impl
                best_dist = A::zero();
                best_item = leaf.1[0];
                
                true // continue processing
            }
        );

        (best_dist, best_item)
    }

    fn nearest_n_within_inner<D, H: ResultCollection<A, T>>(
        &self,
        req_ctx: NeaarestNWithinReqCtx<A, K>,
    ) -> H {
        

        self.backtracking_query(
            req_ctx,
            |leaf, _l| {
                // TODO: real impl
                best_dist = A::zero();
                best_item = leaf.1[0];

                true // continue processing
            }
        );

        (best_dist, best_item)
    }
}

struct NeaarestNWithinReqCtx<'a, A, const K: usize> {
    query: &'a [A; K],
    max_dist: A,
    max_qty: usize,
    sorted: bool,
}


impl<A, const K: usize> QueryContext<A, K> for NeaarestNWithinReqCtx<'_, A, K> {
    fn query(&self) -> &[A; K] {
        self.query
    }
}

#[cfg(test)]
mod tests {
    use crate::Eytzinger;
    use crate::traits_unified_2::DummyLeafStrategy;
    use super::*;

    #[test]
    fn test_get_leaf_idx() {
        let tree: KdTree<f32, u32, Eytzinger<3>, DummyLeafStrategy, 3, 32> = KdTree::default();

        let query = [0.0f32; 3];

        let result = tree.get_leaf_idx(&query);

        assert_eq!(result, 3);
    }
}