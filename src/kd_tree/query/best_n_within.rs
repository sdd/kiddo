use crate::dist::KdTreeDistanceMetric;
use crate::kd_tree::leaf_view::TlsLeafScratch;
use crate::kd_tree::leaf_view_chunked::best_n_within::{
    best_n_within_with_query_wide, best_n_within_with_query_wide_arena,
};
use crate::kd_tree::query_stack::StackTrait;
use crate::kd_tree::result_collection::{BinaryHeapResultCollection, ResultCollection};
#[cfg(feature = "small_n_result_collectors")]
use crate::kd_tree::result_collection::{
    SmallBinaryHeapResultCollection, SMALL_RESULT_COLLECTION_MAX_QTY,
};
use crate::kd_tree::traits::QueryContext;
use crate::kd_tree::KdTree;
use crate::stem_strategies::donnelly_2_blockmarker_simd::{
    BacktrackBlock3, BacktrackBlock4, SimdSelectBestChildBlock3,
};
use crate::traits_unified_2::{AxisUnified, Basics, LeafProjection, LeafStrategy};
use crate::{BestNeighbour, StemStrategy};
use std::collections::BinaryHeap;
use std::num::NonZero;

impl<A, T, SS, LS, const K: usize, const B: usize> KdTree<A, T, SS, LS, K, B>
where
    A: AxisUnified<Coord = A> + 'static,
    T: Basics + Ord,
    LS: LeafStrategy<A, T, SS, K, B>,
    SS: StemStrategy,
{
    #[inline(always)]
    fn process_leaf_best_n_within<D, R>(
        &self,
        leaf_idx: usize,
        query_wide: &[D::Output; K],
        max_dist: D::Output,
        results: &mut R,
    ) where
        D: KdTreeDistanceMetric<A, K>,
        D::Output: AxisUnified<Coord = D::Output> + TlsLeafScratch + 'static,
        R: ResultCollection<D::Output, BestNeighbour<D::Output, T>>,
    {
        match LS::LEAF_PROJECTION {
            LeafProjection::LeafArena => {
                let arena = self.leaves.leaf_arena(leaf_idx);
                best_n_within_with_query_wide_arena::<A, T, D, R, K>(
                    &arena, query_wide, max_dist, results,
                );
            }
            LeafProjection::LeafView => {
                let leaf = self.leaves.leaf_view(leaf_idx);
                best_n_within_with_query_wide::<A, T, D, R, K, B>(
                    &leaf, query_wide, max_dist, results,
                );
            }
        }
    }

    /// Finds the best N points within a given distance.
    ///
    /// Returns up to `max_qty` points that are within `max_dist` of the query point,
    /// prioritizing better items according to `BestNeighbour` ordering.
    pub fn best_n_within<D>(
        &self,
        query: &[A; K],
        max_dist: D::Output,
        max_qty: NonZero<usize>,
    ) -> BinaryHeap<BestNeighbour<D::Output, T>>
    where
        D: KdTreeDistanceMetric<A, K>,
        D::Output: crate::stem_strategies::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static,
    {
        let max_qty = max_qty.into();

        #[cfg(feature = "small_n_result_collectors")]
        if max_qty <= SMALL_RESULT_COLLECTION_MAX_QTY {
            return self
                .best_n_within_inner::<D, SmallBinaryHeapResultCollection<
                    BestNeighbour<D::Output, T>,
                >>(query, max_dist, max_qty)
                .into_inner();
        }

        self.best_n_within_inner::<D, BinaryHeapResultCollection<BestNeighbour<D::Output, T>>>(
            query, max_dist, max_qty,
        )
        .into_inner()
    }

    fn best_n_within_inner<D, R>(&self, query: &[A; K], max_dist: D::Output, max_qty: usize) -> R
    where
        D: KdTreeDistanceMetric<A, K>,
        D::Output: crate::stem_strategies::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        R: ResultCollection<D::Output, BestNeighbour<D::Output, T>>,
        SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static,
    {
        let mut req_ctx = BestNWithinReqCtx::<A, D::Output, R, K> {
            query,
            max_dist,
            results: R::with_max_qty(max_qty),
        };

        self.backtracking_query::<_, _, D>(&mut req_ctx, |leaf_idx, query_wide, req_ctx| {
            self.process_leaf_best_n_within::<D, _>(
                leaf_idx,
                query_wide,
                max_dist,
                &mut req_ctx.results,
            );
        });

        req_ctx.results
    }
}

#[allow(unused)]
#[derive(Debug)]
struct BestNWithinReqCtx<'a, A, O, R, const K: usize>
where
    O: AxisUnified<Coord = O>,
{
    query: &'a [A; K],
    max_dist: O,
    results: R,
}

impl<'a, A, O, R, const K: usize> QueryContext<A, O, K> for BestNWithinReqCtx<'a, A, O, R, K>
where
    O: AxisUnified<Coord = O>,
{
    fn query(&self) -> &[A; K] {
        self.query
    }
    fn max_dist(&self) -> O {
        self.max_dist
    }
}

#[cfg(test)]
mod tests {
    use std::num::{NonZero, NonZeroUsize};

    use rand::rngs::StdRng;
    use rand::Rng;
    use rand::SeedableRng;

    use crate::dist::SquaredEuclidean;
    use crate::kd_tree::leaf_strategies::{FlatVec, VecOfArenas, VecOfArrays};
    use crate::kd_tree::KdTree;
    use crate::{BestNeighbour, Eytzinger};

    const RNG_SEED: u64 = 42;

    #[test]
    fn best_n_within_flat_vec_f32() {
        let mut rng = StdRng::seed_from_u64(RNG_SEED);

        let mut points: Vec<[f32; 3]> = vec![];
        for _ in 0..65_536 {
            let x = rng.random_range(0.0..1.0);
            let y = rng.random_range(0.0..1.0);
            let z = rng.random_range(0.0..1.0);
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
    fn v6_query_best_n_within_large_f64_flat_vec() {
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

    #[test]
    fn v6_query_best_n_within_large_f64_vec_of_arrays() {
        let mut rng = StdRng::seed_from_u64(RNG_SEED);

        const TREE_SIZE: usize = 100_000;
        const NUM_QUERIES: usize = 100;
        let max_qty = NonZero::new(2).unwrap();

        let content_to_add: Vec<_> = (0..TREE_SIZE).map(|_| rng.random::<[f64; 2]>()).collect();

        let tree: KdTree<f64, u32, Eytzinger<2>, VecOfArrays<f64, u32, 2, 32>, 2, 32> =
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

    #[test]
    fn v6_query_best_n_within_large_vec_of_arrays_mutated_f64() {
        let mut rng = StdRng::seed_from_u64(RNG_SEED);

        const TREE_SIZE: usize = 100_000;
        const NUM_QUERIES: usize = 100;
        let max_qty = NonZero::new(2).unwrap();

        let content_to_add: Vec<_> = (0..TREE_SIZE).map(|_| rng.random::<[f64; 2]>()).collect();

        let mut tree: KdTree<f64, u32, Eytzinger<2>, VecOfArrays<f64, u32, 2, 32>, 2, 32> =
            KdTree::default();

        for (idx, point) in content_to_add.iter().enumerate() {
            tree.add(point, idx as u32);
        }

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

    #[test]
    fn v6_query_best_n_within_vec_of_arenas_boundary_parity_f64() {
        let mut rng = StdRng::seed_from_u64(RNG_SEED);
        let query = [0.35, 0.65];
        let radius = 10.0f64;
        let max_qty = NonZero::new(5).unwrap();

        for len in [1usize, 2, 4, 8, 32, 33, 47] {
            let points: Vec<[f64; 2]> = (0..len).map(|_| rng.random::<[f64; 2]>()).collect();

            let flat_tree: KdTree<f64, u32, Eytzinger<2>, FlatVec<f64, u32, 2, 32>, 2, 32> =
                KdTree::new_from_slice(&points);
            let arena_tree: KdTree<f64, u32, Eytzinger<2>, VecOfArenas<f64, u32, 2, 32>, 2, 32> =
                KdTree::new_from_slice(&points);

            let mut flat_results: Vec<_> = flat_tree
                .best_n_within::<SquaredEuclidean<f64>>(&query, radius, max_qty)
                .into_sorted_vec();
            let mut arena_results: Vec<_> = arena_tree
                .best_n_within::<SquaredEuclidean<f64>>(&query, radius, max_qty)
                .into_sorted_vec();

            flat_results.sort();
            arena_results.sort();

            assert_eq!(arena_results, flat_results, "len={len}");
        }
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
    #[test]
    fn best_n_within_vec_of_arenas_matches_flat_vec_f64_simd() {
        let points: Vec<[f64; 3]> = (0..40)
            .map(|idx| {
                [
                    idx as f64 / 40.0,
                    ((idx * 7) % 40) as f64 / 40.0,
                    ((idx * 13) % 40) as f64 / 40.0,
                ]
            })
            .collect();
        let query = [0.42f64, 0.53, 0.61];
        let max_qty = NonZero::new(5).unwrap();
        let max_dist = 0.2;

        let flat_tree: KdTree<f64, u32, Eytzinger<3>, FlatVec<f64, u32, 3, 32>, 3, 32> =
            KdTree::new_from_slice(&points);
        let arena_tree: KdTree<f64, u32, Eytzinger<3>, VecOfArenas<f64, u32, 3, 32>, 3, 32> =
            KdTree::new_from_slice(&points);

        let flat_result = flat_tree
            .best_n_within::<SquaredEuclidean<f64>>(&query, max_dist, max_qty)
            .into_sorted_vec();
        let arena_result = arena_tree
            .best_n_within::<SquaredEuclidean<f64>>(&query, max_dist, max_qty)
            .into_sorted_vec();

        assert_eq!(arena_result, flat_result);
    }

    fn linear_search(
        content: &[[f64; 2]],
        query: &[f64; 2],
        radius: f64,
        max_qty: usize,
    ) -> Vec<BestNeighbour<f64, u32>> {
        let mut best_items = Vec::with_capacity(max_qty);

        for (item, p) in content.iter().enumerate() {
            let distance = squared_euclidean_dist(query, p);
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

    fn squared_euclidean_dist<const K: usize>(a: &[f64; K], b: &[f64; K]) -> f64 {
        let aw = (*a).map(|coord| {
            <crate::dist::SquaredEuclidean<f64> as crate::dist::DistanceMetricCore<f64>>::widen_coord(
                coord,
            )
        });
        let bw = (*b).map(|coord| {
            <crate::dist::SquaredEuclidean<f64> as crate::dist::DistanceMetricCore<f64>>::widen_coord(
                coord,
            )
        });

        <crate::dist::SquaredEuclidean<f64> as crate::dist::DistanceMetricCore<f64>>::dist::<K>(
            &aw, &bw,
        )
    }
}
