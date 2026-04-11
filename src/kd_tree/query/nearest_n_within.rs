use crate::dist::KdTreeDistanceMetric;
use crate::kd_tree::leaf_view::TlsLeafScratch;
use crate::kd_tree::leaf_view_chunked::nearest_n_within::{
    nearest_n_within_with_query_wide, nearest_n_within_with_query_wide_arena,
};
use crate::kd_tree::query_stack::StackTrait;
use crate::kd_tree::result_collection::{
    BinaryHeapResultCollection, ResultCollection, SortedVecResultCollection,
};
use crate::kd_tree::traits::QueryContext;
use crate::kd_tree::KdTree;
use crate::stem_strategies::donnelly_2_blockmarker_simd::{
    BacktrackBlock3, BacktrackBlock4, SimdSelectBestChildBlock3,
};
use crate::traits_unified_2::{AxisUnified, Basics, LeafProjection, LeafStrategy};
use crate::{NearestNeighbour, StemStrategy};
use std::num::NonZeroUsize;

const MAX_VEC_RESULT_SIZE: usize = 20;

impl<A, T, SS, LS, const K: usize, const B: usize> KdTree<A, T, SS, LS, K, B>
where
    A: AxisUnified<Coord = A> + 'static,
    T: Basics + Ord,
    LS: LeafStrategy<A, T, SS, K, B>,
    SS: StemStrategy,
{
    #[inline(always)]
    fn process_leaf_nearest_n_within<D, R>(
        &self,
        leaf_idx: usize,
        query_wide: &[D::Output; K],
        max_dist: D::Output,
        results: &mut R,
    ) where
        D: KdTreeDistanceMetric<A, K>,
        D::Output: AxisUnified<Coord = D::Output> + TlsLeafScratch + 'static,
        R: ResultCollection<D::Output, NearestNeighbour<D::Output, T>>,
    {
        match LS::LEAF_PROJECTION {
            LeafProjection::LeafArena => {
                let arena = self.leaves.leaf_arena(leaf_idx);
                nearest_n_within_with_query_wide_arena::<A, T, D, R, K>(
                    &arena, query_wide, max_dist, results,
                );
            }
            LeafProjection::LeafView => {
                let leaf = self.leaves.leaf_view(leaf_idx);
                nearest_n_within_with_query_wide::<A, T, D, R, K, B>(
                    &leaf, query_wide, max_dist, results,
                );
            }
        }
    }

    /// Finds up to N nearest points within a given distance.
    ///
    /// Returns up to `max_qty` points that are within `max_dist` of the query point.
    /// If `sorted` is true, results are returned in order of increasing distance.
    pub fn nearest_n_within<D>(
        &self,
        query: &[A; K],
        max_dist: D::Output,
        max_qty: NonZeroUsize,
        sorted: bool,
    ) -> Vec<NearestNeighbour<D::Output, T>>
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
        let max_qty: usize = max_qty.get();

        if max_qty == usize::MAX {
            self.nearest_n_within_inner::<D, Vec<NearestNeighbour<D::Output, T>>>(
                query, max_dist, max_qty, sorted,
            )
        } else if sorted && max_qty <= MAX_VEC_RESULT_SIZE {
            self.nearest_n_within_inner::<
                D,
                SortedVecResultCollection<NearestNeighbour<D::Output, T>>,
            >(query, max_dist, max_qty, sorted)
        } else {
            self.nearest_n_within_inner::<
                D,
                BinaryHeapResultCollection<NearestNeighbour<D::Output, T>>,
            >(query, max_dist, max_qty, sorted)
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
        D: KdTreeDistanceMetric<A, K>,
        D::Output: crate::stem_strategies::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        R: ResultCollection<D::Output, NearestNeighbour<D::Output, T>>,
        SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static,
    {
        let mut req_ctx = NearestNWithinReqCtx {
            query,
            max_dist,
            results: R::with_max_qty(max_qty),
            _phantom: std::marker::PhantomData,
        };

        self.backtracking_query::<_, _, D>(&mut req_ctx, |leaf_idx, query_wide, req_ctx| {
            self.process_leaf_nearest_n_within::<D, R>(
                leaf_idx,
                query_wide,
                max_dist,
                &mut req_ctx.results,
            );
        });

        if sorted {
            req_ctx.results.into_sorted_vec()
        } else {
            req_ctx.results.into_vec()
        }
    }
}

#[allow(unused)]
struct NearestNWithinReqCtx<'a, A, T, O, R, const K: usize>
where
    O: AxisUnified<Coord = O>,
{
    query: &'a [A; K],
    max_dist: O,
    results: R,
    _phantom: std::marker::PhantomData<T>,
}

impl<A, T, O, R, const K: usize> QueryContext<A, O, K> for NearestNWithinReqCtx<'_, A, T, O, R, K>
where
    O: AxisUnified<Coord = O>,
    R: ResultCollection<O, NearestNeighbour<O, T>>,
{
    fn query(&self) -> &[A; K] {
        self.query
    }

    fn max_dist(&self) -> O {
        let results_cap = self.results.threshold_distance().unwrap_or(O::max_value());
        if results_cap < self.max_dist {
            results_cap
        } else {
            self.max_dist
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;
    use std::num::{NonZero, NonZeroUsize};

    use rand::rngs::StdRng;
    use rand::Rng;
    use rand::SeedableRng;
    use test_log::test;

    use crate::dist::SquaredEuclidean;
    use crate::kd_tree::leaf_strategies::{FlatVec, VecOfArenas, VecOfArrays};
    use crate::kd_tree::KdTree;
    use crate::traits::Axis;
    use crate::Eytzinger;

    const RNG_SEED: u64 = 42;

    #[test]
    fn nearest_n_within_sorted_flat_vec_f32() {
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

        let query_point = [0.5, 0.5, 0.5];
        let radius = 0.1;
        let max_qty = NonZeroUsize::new(10).unwrap();

        let results =
            tree.nearest_n_within::<SquaredEuclidean<f32>>(&query_point, radius, max_qty, true);
        assert_eq!(results.len(), 10);
    }

    #[test]
    fn nearest_n_within_vec_of_arenas_matches_flat_vec_f32() {
        let points: Vec<[f32; 3]> = (0..40)
            .map(|idx| {
                [
                    idx as f32 / 40.0,
                    ((idx * 7) % 40) as f32 / 40.0,
                    ((idx * 13) % 40) as f32 / 40.0,
                ]
            })
            .collect();
        let query = [0.42f32, 0.53, 0.61];
        let max_qty = NonZeroUsize::new(5).unwrap();
        let max_dist = 0.2;

        let flat_tree: KdTree<f32, u32, Eytzinger<3>, FlatVec<f32, u32, 3, 32>, 3, 32> =
            KdTree::new_from_slice(&points);
        let arena_tree: KdTree<f32, u32, Eytzinger<3>, VecOfArenas<f32, u32, 3, 32>, 3, 32> =
            KdTree::new_from_slice(&points);

        let flat_result =
            flat_tree.nearest_n_within::<SquaredEuclidean<f32>>(&query, max_dist, max_qty, true);
        let arena_result =
            arena_tree.nearest_n_within::<SquaredEuclidean<f32>>(&query, max_dist, max_qty, true);

        assert_eq!(arena_result, flat_result);
    }

    #[test]
    fn nearest_n_within_unbounded_vec_of_arenas_matches_flat_vec_f32() {
        let points: Vec<[f32; 3]> = (0..40)
            .map(|idx| {
                [
                    ((idx * 3) % 40) as f32 / 40.0,
                    ((idx * 11) % 40) as f32 / 40.0,
                    ((idx * 17) % 40) as f32 / 40.0,
                ]
            })
            .collect();
        let query = [0.35f32, 0.45, 0.55];
        let max_dist = 0.5;

        let flat_tree: KdTree<f32, u32, Eytzinger<3>, FlatVec<f32, u32, 3, 32>, 3, 32> =
            KdTree::new_from_slice(&points);
        let arena_tree: KdTree<f32, u32, Eytzinger<3>, VecOfArenas<f32, u32, 3, 32>, 3, 32> =
            KdTree::new_from_slice(&points);

        let flat_result = flat_tree.nearest_n_within::<SquaredEuclidean<f32>>(
            &query,
            max_dist,
            NonZeroUsize::MAX,
            true,
        );
        let arena_result = arena_tree.nearest_n_within::<SquaredEuclidean<f32>>(
            &query,
            max_dist,
            NonZeroUsize::MAX,
            true,
        );

        assert_eq!(arena_result, flat_result);
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
    #[test]
    fn nearest_n_within_vec_of_arenas_matches_flat_vec_f64_simd() {
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
        let max_qty = NonZeroUsize::new(5).unwrap();
        let max_dist = 0.2;

        let flat_tree: KdTree<f64, u32, Eytzinger<3>, FlatVec<f64, u32, 3, 32>, 3, 32> =
            KdTree::new_from_slice(&points);
        let arena_tree: KdTree<f64, u32, Eytzinger<3>, VecOfArenas<f64, u32, 3, 32>, 3, 32> =
            KdTree::new_from_slice(&points);

        let flat_result =
            flat_tree.nearest_n_within::<SquaredEuclidean<f64>>(&query, max_dist, max_qty, true);
        let arena_result =
            arena_tree.nearest_n_within::<SquaredEuclidean<f64>>(&query, max_dist, max_qty, true);

        assert_eq!(arena_result, flat_result);
    }

    #[test]
    fn v6_n_items_within_f32_eytzinger_large_scale() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(RNG_SEED);

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

        for query_point in query_points.iter() {
            let expected = linear_search(&content_to_add, query_point, RADIUS)
                .into_iter()
                .take(max_qty.into())
                .collect::<Vec<_>>();

            let mut result: Vec<_> = tree
                .nearest_n_within::<SquaredEuclidean<f32>>(query_point, RADIUS, max_qty, true)
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();

            stabilize_sort(&mut result);

            // println!("Query #{}", i);
            assert_eq!(result, expected);
        }
    }

    #[test]
    fn v6_nearest_n_within_f32_eytzinger_large_vec_of_arrays() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(RNG_SEED);

        const TREE_SIZE: usize = 100_000;
        const NUM_QUERIES: usize = 100;
        const RADIUS: f32 = 0.2;

        let max_qty: NonZero<usize> = NonZero::new(3).unwrap();

        let content_to_add: Vec<[f32; 4]> =
            (0..TREE_SIZE).map(|_| rng.random::<[f32; 4]>()).collect();

        let tree: KdTree<f32, u32, Eytzinger<4>, VecOfArrays<f32, u32, 4, 32>, 4, 32> =
            KdTree::new_from_slice(&content_to_add);
        assert_eq!(tree.size(), TREE_SIZE);

        let query_points: Vec<[f32; 4]> = (0..NUM_QUERIES)
            .map(|_| rng.random::<[f32; 4]>()) // Use the seeded rng
            .collect();

        for query_point in query_points.iter() {
            let expected = linear_search(&content_to_add, query_point, RADIUS)
                .into_iter()
                .take(max_qty.into())
                .collect::<Vec<_>>();

            let mut result: Vec<_> = tree
                .nearest_n_within::<SquaredEuclidean<f32>>(query_point, RADIUS, max_qty, true)
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();

            stabilize_sort(&mut result);

            // println!("Query #{}", i);
            assert_eq!(result, expected);
        }
    }

    #[test]
    fn v6_nearest_n_within_f32_eytzinger_large_vec_of_arrays_mutated_f32() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(RNG_SEED);

        const TREE_SIZE: usize = 100_000;
        const NUM_QUERIES: usize = 100;
        const RADIUS: f32 = 0.2;

        let max_qty: NonZero<usize> = NonZero::new(3).unwrap();

        let content_to_add: Vec<[f32; 4]> =
            (0..TREE_SIZE).map(|_| rng.random::<[f32; 4]>()).collect();

        let mut tree: KdTree<f32, u32, Eytzinger<4>, VecOfArrays<f32, u32, 4, 32>, 4, 32> =
            KdTree::default();

        for (idx, point) in content_to_add.iter().enumerate() {
            tree.add(point, idx as u32);
        }

        assert_eq!(tree.size(), TREE_SIZE);

        let query_points: Vec<[f32; 4]> = (0..NUM_QUERIES)
            .map(|_| rng.random::<[f32; 4]>()) // Use the seeded rng
            .collect();

        for query_point in query_points.iter() {
            let expected = linear_search(&content_to_add, query_point, RADIUS)
                .into_iter()
                .take(max_qty.into())
                .collect::<Vec<_>>();

            let mut result: Vec<_> = tree
                .nearest_n_within::<SquaredEuclidean<f32>>(query_point, RADIUS, max_qty, true)
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();

            stabilize_sort(&mut result);

            // println!("Query #{}", i);
            assert_eq!(result, expected);
        }
    }

    fn linear_search<A: Axis, const K: usize>(
        content: &[[A; K]],
        query_point: &[A; K],
        radius: A,
    ) -> Vec<(A, u32)>
    where
        crate::dist::SquaredEuclidean<A>: crate::dist::DistanceMetricCore<A, Output = A>,
    {
        let mut matching_items = vec![];

        for (idx, p) in content.iter().enumerate() {
            let dist = squared_euclidean_dist(query_point, p);
            if dist <= radius {
                matching_items.push((dist, idx as u32));
            }
        }

        stabilize_sort(&mut matching_items);

        matching_items
    }

    fn squared_euclidean_dist<A: Axis, const K: usize>(a: &[A; K], b: &[A; K]) -> A
    where
        crate::dist::SquaredEuclidean<A>: crate::dist::DistanceMetricCore<A, Output = A>,
    {
        let aw = (*a).map(|coord| {
            <crate::dist::SquaredEuclidean<A> as crate::dist::DistanceMetricCore<A>>::widen_coord(
                coord,
            )
        });
        let bw = (*b).map(|coord| {
            <crate::dist::SquaredEuclidean<A> as crate::dist::DistanceMetricCore<A>>::widen_coord(
                coord,
            )
        });

        <crate::dist::SquaredEuclidean<A> as crate::dist::DistanceMetricCore<A>>::dist::<K>(
            &aw, &bw,
        )
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
