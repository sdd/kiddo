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
        // <D as DistanceMetricUnified<A, K>>::Output: Ord,
    {
        let max_qty = max_qty.into();
        let mut req_ctx = BestNWithinReqCtx::<A, T, <D as DistanceMetricUnified<A, K>>::Output, K> {
            query,
            max_dist,
            max_qty,
            results: BinaryHeap::with_capacity(max_qty),
        };

        self.backtracking_query::<_, _, D>(&mut req_ctx, |leaf, req_ctx| {
            leaf.best_n_within::<D>(query, max_dist, &mut req_ctx.results);
        });

        req_ctx.results
    }
}

#[derive(Debug)]
struct BestNWithinReqCtx<'a, A, T, O, const K: usize>
where
    O: AxisUnified<Coord = O>, // + Ord,
    T: Ord,
{
    query: &'a [A; K],
    max_dist: O,
    max_qty: usize,
    results: BinaryHeap<BestNeighbour<O, T>>,
}

impl<'a, A, T, O, const K: usize> QueryContext<A, O, K> for BestNWithinReqCtx<'a, A, T, O, K>
where
    O: AxisUnified<Coord = O>, // + Ord,
    T: Ord,
{
    fn query(&self) -> &[A; K] {
        self.query
    }
    fn max_dist(&self) -> O {
        self.results
            .peek()
            .map(|n| n.distance)
            .unwrap_or(self.max_dist)
    }
}

#[cfg(test)]
mod tests {
    use std::num::{NonZero, NonZeroUsize};

    use rand::rngs::StdRng;
    use rand::Rng;
    use rand::SeedableRng;

    use crate::kd_tree::leaf_strategies::flat_vec::FlatVec;
    use crate::kd_tree::KdTree;
    use crate::traits::DistanceMetric;
    use crate::traits_unified_2::DistanceMetricUnified;
    use crate::traits_unified_2::SquaredEuclidean;
    use crate::{BestNeighbour, Eytzinger};

    const RNG_SEED: u64 = 42;

    #[test]
    fn best_n_within_flat_vec_f32() {
        let mut rng = StdRng::seed_from_u64(RNG_SEED);

        let mut points: Vec<[f32; 3]> = vec![];
        for _ in 0..65_536 {
            let x = rng.gen_range(0.0..1.0);
            let y = rng.gen_range(0.0..1.0);
            let z = rng.gen_range(0.0..1.0);
            points.push([x, y, z]);
        }

        let tree: KdTree<f32, u32, Eytzinger<3>, FlatVec<f32, u32, 3, 32>, 3, 32> =
            KdTree::new_from_slice(&points);

        assert!(!tree.is_empty());
        assert_eq!(tree.size(), 65_536);
        assert_eq!(tree.leaf_count(), 2048);
        assert_eq!(tree.max_stem_level(), 10);

        // perform a best_n_within query
        let query_point = [0.5, 0.5, 0.5];
        let radius = 0.1f32;
        let max_qty = NonZeroUsize::new(10).unwrap();
        let results = tree.best_n_within::<SquaredEuclidean<f32>>(&query_point, radius, max_qty);
        assert_eq!(results.len(), 10);
    }

    #[test]
    fn can_query_best_items_within_radius_large_scale() {
        let mut rng = StdRng::seed_from_u64(RNG_SEED);

        const TREE_SIZE: usize = 100_000;
        const NUM_QUERIES: usize = 100;
        let max_qty = NonZero::new(2).unwrap();

        let content_to_add: Vec<_> = (0..TREE_SIZE).map(|_| rng.random::<[f64; 2]>()).collect();

        let tree: KdTree<f64, u32, Eytzinger<2>, FlatVec<f64, u32, 2, 32>, 2, 32> =
            KdTree::new_from_slice(&content_to_add);

        assert_eq!(tree.size(), TREE_SIZE);

        let query_points: Vec<_> = (0..NUM_QUERIES)
            .map(|_| rng.random::<_>()) // Use the seeded rng
            .collect();

        for query_point in query_points {
            let radius = 100000f64;
            let expected = linear_search(&content_to_add, &query_point, radius, max_qty.into());

            let result: Vec<_> = tree
                .best_n_within::<SquaredEuclidean<f64>>(&query_point, radius, max_qty)
                .into_iter()
                .collect();

            assert_eq!(result, expected);
        }
    }

    fn linear_search(
        content: &[[f64; 2]],
        query: &[f64; 2],
        radius: f64,
        max_qty: usize,
    ) -> Vec<BestNeighbour<f64, u32>> {
        let mut best_items = Vec::with_capacity(max_qty);

        for (item, p) in content.iter().enumerate() {
            let distance: f64 = crate::SquaredEuclidean::dist(query, p);
            if distance <= radius {
                if best_items.len() < max_qty {
                    best_items.push(BestNeighbour {
                        distance,
                        item: item as u32,
                    });
                } else if (item as u32) < best_items.last().unwrap().item {
                    best_items.pop().unwrap();
                    best_items.push(BestNeighbour {
                        distance,
                        item: item as u32,
                    });
                }
            }
            best_items.sort_unstable();
        }
        best_items.reverse();

        best_items
    }
}
