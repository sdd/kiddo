use crate::kd_tree::traits::QueryContext;
use crate::kd_tree::KdTree;
use crate::traits_unified_2::{AxisUnified, Basics, DistanceMetricUnified, LeafStrategy};
use crate::StemStrategy;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::num::NonZero;

impl<A, T, SS, LS, const K: usize, const B: usize> KdTree<A, T, SS, LS, K, B>
where
    A: AxisUnified<Coord = A>,
    T: Basics + PartialOrd + PartialEq,
    LS: LeafStrategy<A, T, SS, K, B>,
    SS: StemStrategy,
{
    pub fn best_n_within<D>(
        &self,
        query: &[A; K],
        dist: <D as DistanceMetricUnified<A, K>>::Output,
        max_qty: NonZero<usize>,
    ) -> BinaryHeap<BestNeighbour<<D as DistanceMetricUnified<A, K>>::Output, T>>
    where
        D: DistanceMetricUnified<A, K>,
    {
        type DistOut<A, const K: usize, D> = <D as DistanceMetricUnified<A, K>>::Output;

        let req_ctx = BestNWithinReqCtx { query, dist };

        let max_qty = max_qty.into();

        let mut results: BinaryHeap<BestNeighbour<DistOut<A, K, D>, T>> =
            BinaryHeap::with_capacity(max_qty);

        self.backtracking_query(req_ctx, |(coords, items): &([&[A]; K], &[T]), _l| {
            // TODO: Vectorized
            for idx in 0..items.len() {
                let mut distance = <DistOut<A, K, D> as AxisUnified>::zero();

                (0..K).for_each(|dim| {
                    distance += D::dist1(coords[dim][idx], query[dim]);
                });

                if distance < req_ctx.dist {
                    let item = *unsafe { items.get_unchecked(idx) };
                    if results.len() < max_qty {
                        results.push(BestNeighbour { distance, item });
                    } else {
                        let mut top = results.peek_mut().unwrap();
                        if item < top.item {
                            top.item = item;
                            top.distance = distance;
                        }
                    }
                }
            }

            true // continue processing
        });

        results
    }
}

#[derive(Debug, Copy, Clone)]
struct BestNWithinReqCtx<'a, A, DOut, const K: usize> {
    query: &'a [A; K],
    dist: DOut,
}

impl<'a, A, DOut, const K: usize> QueryContext<A, K> for BestNWithinReqCtx<'a, A, DOut, K> {
    fn query(&self) -> &[A; K] {
        self.query
    }
}

/// Represents an entry in the results of a "best" query, with `distance` being the distance of this
/// particular item from the query point, and `item` being the stored item index that was found
/// as part of the query.
#[derive(Debug, Copy, Clone)]
pub struct BestNeighbour<DOut, T> {
    /// the distance of the found item from the query point according to the supplied distance metric
    pub distance: DOut,
    /// the stored index of an item that was found in the query
    pub item: T,
}

impl<DOut: PartialOrd + PartialEq, T: Basics + PartialEq + PartialOrd> PartialEq<Self>
    for BestNeighbour<DOut, T>
{
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance && self.item == other.item
    }
}

impl<DOut: PartialOrd + PartialEq, T: Basics + PartialEq + PartialOrd> PartialOrd<Self>
    for BestNeighbour<DOut, T>
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // First compare by distance, then by item to break ties
        match self.distance.partial_cmp(&other.distance) {
            Some(Ordering::Equal) => self.item.partial_cmp(&other.item),
            non_eq => non_eq,
        }
    }
}

impl<DOut: PartialOrd + PartialEq, T: Basics + PartialEq + PartialOrd> Eq
    for BestNeighbour<DOut, T>
{
}

impl<DOut: PartialOrd + PartialEq, T: Basics + PartialEq + PartialOrd> Ord
    for BestNeighbour<DOut, T>
{
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

impl<DOut, T: Basics> From<BestNeighbour<DOut, T>> for (DOut, T) {
    fn from(elem: BestNeighbour<DOut, T>) -> Self {
        (elem.distance, elem.item)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits_unified_2::{DummyLeafStrategy, SquaredEuclidean};
    use crate::Eytzinger;

    #[test]
    fn test_get_leaf_idx() {
        let tree: KdTree<f32, u32, Eytzinger<3>, DummyLeafStrategy, 3, 32> = KdTree::default();

        let query = [0.0f32; 3];

        let result = tree
            .best_n_within::<SquaredEuclidean>(&query, 0.5f32, NonZero::new(3).unwrap())
            .len();

        assert_eq!(result, 3);
    }
}
