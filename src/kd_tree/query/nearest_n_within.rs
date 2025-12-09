use crate::kd_tree::traits::QueryContext;
use crate::kd_tree::KdTree;
use crate::mutable::float::result_collection::ResultCollection;
use crate::traits_unified_2::{AxisUnified, Basics, DistanceMetricUnified, LeafStrategy};
use crate::{NearestNeighbour, StemStrategy};
use sorted_vec::SortedVec;
use std::collections::BinaryHeap;

const MAX_VEC_RESULT_SIZE: usize = 20;

impl<A, T, SS, LS, const K: usize, const B: usize> KdTree<A, T, SS, LS, K, B>
where
    A: AxisUnified<Coord = A>,
    T: Basics + Ord,
    LS: LeafStrategy<A, T, SS, K, B>,
    SS: StemStrategy,
{
    pub fn nearest_n_within<D>(
        &self,
        query: &[A; K],
        max_dist: D::Output,
        max_qty: usize,
        sorted: bool,
    ) -> Vec<NearestNeighbour<D::Output, T>>
    where
        D: DistanceMetricUnified<A, K>,
    {
        if sorted && max_qty < usize::MAX {
            if max_qty <= MAX_VEC_RESULT_SIZE {
                self.nearest_n_within_inner::<D, SortedVec<NearestNeighbour<D::Output, T>>>(
                    query, max_dist, max_qty, sorted,
                )
            } else {
                self.nearest_n_within_inner::<D, BinaryHeap<NearestNeighbour<D::Output, T>>>(
                    query, max_dist, max_qty, sorted,
                )
            }
        } else {
            self.nearest_n_within_inner::<D, Vec<NearestNeighbour<D::Output, T>>>(
                query, max_dist, 0, sorted,
            )
        }
    }

    fn nearest_n_within_inner<D, R>(
        &self,
        query: &[A; K],
        max_dist: D::Output,
        max_qty: usize,
        sorted: bool,
    ) -> Vec<NearestNeighbour<D::Output, T>>
    where
        D: DistanceMetricUnified<A, K>,
        R: ResultCollection<D::Output, T>,
    {
        let req_ctx = NearestNWithinReqCtx {
            query,
            max_dist,
            max_qty,
            sorted,
        };

        let mut results = R::new_with_capacity(max_qty);

        self.backtracking_query(req_ctx, |leaf| {
            leaf.nearest_n_within::<D, R>(query, max_dist, &mut results);
        });

        if sorted {
            results.into_sorted_vec()
        } else {
            results.into_vec()
        }
    }
}

struct NearestNWithinReqCtx<'a, A, O, const K: usize> {
    query: &'a [A; K],
    max_dist: O,
    max_qty: usize,
    sorted: bool,
}

impl<A, O, const K: usize> QueryContext<A, K> for NearestNWithinReqCtx<'_, A, O, K> {
    fn query(&self) -> &[A; K] {
        self.query
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kd_tree::leaf_strategies::dummy::DummyLeafStrategy;
    use crate::Eytzinger;

    #[test]
    fn test_get_leaf_idx() {
        let tree: KdTree<f32, u32, Eytzinger<3>, DummyLeafStrategy, 3, 32> = KdTree::default();

        let query = [0.0f32; 3];

        let result = tree.get_leaf_idx(&query);

        assert_eq!(result, 3);
    }
}
