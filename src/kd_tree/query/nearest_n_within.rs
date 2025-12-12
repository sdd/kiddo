use crate::kd_tree::result_collection::ResultCollection;
use crate::kd_tree::traits::QueryContext;
use crate::kd_tree::KdTree;
use crate::traits_unified_2::{AxisUnified, Basics, DistanceMetricUnified, LeafStrategy};
use crate::{NearestNeighbour, StemStrategy};
use sorted_vec::SortedVec;
use std::collections::BinaryHeap;
use std::num::NonZeroUsize;

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
        max_qty: NonZeroUsize,
        sorted: bool,
    ) -> Vec<NearestNeighbour<D::Output, T>>
    where
        D: DistanceMetricUnified<A, K>,
    {
        let max_qty: usize = max_qty.get();

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
        let mut req_ctx = NearestNWithinReqCtx {
            query,
            max_dist,
            max_qty,
            sorted,
            results: R::new_with_capacity(max_qty),
            _phantom: std::marker::PhantomData,
        };

        self.backtracking_query::<_, _, D>(&mut req_ctx, |leaf, req_ctx| {
            leaf.nearest_n_within::<D, R>(query, max_dist, &mut req_ctx.results);
        });

        if sorted {
            req_ctx.results.into_sorted_vec()
        } else {
            req_ctx.results.into_vec()
        }
    }
}

struct NearestNWithinReqCtx<'a, A, T, O, R, const K: usize>
where
    O: AxisUnified<Coord = O>,
{
    query: &'a [A; K],
    max_dist: O,
    max_qty: usize,
    sorted: bool,
    results: R,
    _phantom: std::marker::PhantomData<T>,
}

impl<A, T, O, R, const K: usize> QueryContext<A, O, K> for NearestNWithinReqCtx<'_, A, T, O, R, K>
where
    O: AxisUnified<Coord = O>,
    R: ResultCollection<O, T>,
{
    fn query(&self) -> &[A; K] {
        self.query
    }

    fn max_dist(&self) -> O {
        self.results.max_dist()
    }
}

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;
    use std::num::{NonZero, NonZeroUsize};

    use rand::rngs::StdRng;
    use rand::Rng;
    use rand::SeedableRng;

    use crate::kd_tree::result_collection::ResultCollection;
    use crate::kd_tree::{leaf_strategies::flat_vec::FlatVec, KdTree};
    use crate::traits::{Axis, DistanceMetric};
    use crate::traits_unified_2::SquaredEuclidean;
    use crate::{Eytzinger, NearestNeighbour};

    const RNG_SEED: u64 = 42;

    #[test]
    fn nearest_n_within_sorted_flat_vec_f32() {
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

        let query_point = [0.5, 0.5, 0.5];
        let radius = 0.1;
        let max_qty = NonZeroUsize::new(10).unwrap();

        let results =
            tree.nearest_n_within::<SquaredEuclidean<f32>>(&query_point, radius, max_qty, true);
        assert_eq!(results.len(), 10);
    }

    #[test]
    fn can_query_items_within_radius_large_scale() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(3);

        const TREE_SIZE: usize = 100_000;
        const NUM_QUERIES: usize = 100;
        const RADIUS: f32 = 0.2;

        let max_qty: NonZero<usize> = NonZero::new(3).unwrap();

        let content_to_add: Vec<[f32; 4]> =
            (0..TREE_SIZE).map(|_| rng.random::<[f32; 4]>()).collect();

        let tree: KdTree<f32, u32, Eytzinger<4>, FlatVec<f32, u32, 4, 32>, 4, 32> =
            KdTree::new_from_slice(&content_to_add);
        assert_eq!(tree.size(), TREE_SIZE);

        let query_points: Vec<[f32; 4]> = (0..NUM_QUERIES)
            .map(|_| rng.random::<[f32; 4]>()) // Use the seeded rng
            .collect();

        for query_point in query_points {
            let expected = linear_search(&content_to_add, &query_point, RADIUS)
                .into_iter()
                .take(max_qty.into())
                .collect::<Vec<_>>();

            let mut result: Vec<_> = tree
                .nearest_n_within::<SquaredEuclidean<f32>>(&query_point, RADIUS, max_qty, true)
                .into_sorted_vec()
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();

            stabilize_sort(&mut result);

            assert_eq!(result, expected);
        }
    }

    fn linear_search<A: Axis, const K: usize>(
        content: &[[A; K]],
        query_point: &[A; K],
        radius: A,
    ) -> Vec<(A, u32)> {
        let mut matching_items = vec![];

        for (idx, p) in content.iter().enumerate() {
            let dist = crate::SquaredEuclidean::dist(query_point, p);
            if dist < radius {
                matching_items.push((dist, idx as u32));
            }
        }

        stabilize_sort(&mut matching_items);

        matching_items
    }

    fn stabilize_sort<A: Axis>(matching_items: &mut [(A, u32)]) {
        matching_items.sort_unstable_by(|a, b| {
            let dist_cmp = a.0.partial_cmp(&b.0).unwrap();
            if dist_cmp == Ordering::Equal {
                a.1.cmp(&b.1)
            } else {
                dist_cmp
            }
        });
    }
}
