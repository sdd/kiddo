use crate::dist::KdTreeDistanceMetric;
use crate::kd_tree::query_stack::StackTrait;
use crate::kd_tree::traits::QueryContext;
use crate::kd_tree::KdTree;
use crate::kd_tree::KdTreeQueryOps;
use crate::leaf_view::TlsLeafScratch;
use crate::leaf_view_chunked::nearest_n_within::{
    nearest_n_within_with_query_wide, nearest_n_within_with_query_wide_arena,
};
use crate::results::result_collection::VisitorResultCollection;
use crate::stem_strategy::donnelly_2_blockmarker_simd::{
    BacktrackBlock3, BacktrackBlock4, SimdSelectBestChildBlock3,
};
use crate::traits_unified_2::{AxisUnified, Basics, LeafProjection, LeafStrategy};
use crate::{NearestNeighbour, StemStrategy};
use std::num::NonZeroUsize;

impl<A, T, SS, LS, const K: usize, const B: usize> KdTree<A, T, SS, LS, K, B>
where
    A: AxisUnified<Coord = A> + 'static,
    T: Basics + PartialOrd,
    LS: LeafStrategy<A, T, SS, K, B>,
    SS: StemStrategy,
{
    #[inline(always)]
    fn process_leaf_within_unsorted_visit<D, F>(
        &self,
        leaf_idx: usize,
        query_wide: &[D::Output; K],
        max_dist: D::Output,
        visitor: &mut F,
    ) where
        D: KdTreeDistanceMetric<A, K>,
        D::Output: AxisUnified<Coord = D::Output> + TlsLeafScratch + 'static,
        F: FnMut(NearestNeighbour<D::Output, T>),
    {
        let mut results = VisitorResultCollection::new(visitor);

        match LS::LEAF_PROJECTION {
            LeafProjection::LeafArena => {
                let arena = self.leaves.leaf_arena(leaf_idx);
                nearest_n_within_with_query_wide_arena::<A, T, D, _, K>(
                    &arena,
                    query_wide,
                    max_dist,
                    &mut results,
                );
            }
            LeafProjection::LeafView => {
                let leaf = self.leaves.leaf_view(leaf_idx);
                nearest_n_within_with_query_wide::<A, T, D, _, K, B>(
                    &leaf,
                    query_wide,
                    max_dist,
                    &mut results,
                );
            }
        }
    }

    /// Visits every point within a given distance of the query point, unsorted.
    ///
    /// This is the lowest-overhead streaming range-query API. It runs the normal
    /// traversal and optimized leaf kernels, but routes each match directly to
    /// `visitor` instead of building a result collection.
    ///
    /// Prefer this over [`within_unsorted_iter`](Self::within_unsorted_iter) when callback
    /// style is acceptable and allocation/dispatch overhead matters.
    #[inline]
    pub fn within_unsorted_visit<D, F>(&self, query: &[A; K], max_dist: D::Output, mut visitor: F)
    where
        D: KdTreeDistanceMetric<A, K>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static,
        F: FnMut(NearestNeighbour<D::Output, T>),
    {
        let mut req_ctx = WithinUnsortedVisitReqCtx {
            query,
            max_dist,
            _phantom: std::marker::PhantomData,
        };

        self.backtracking_query::<_, _, D>(&mut req_ctx, |leaf_idx, query_wide, req_ctx| {
            self.process_leaf_within_unsorted_visit::<D, F>(
                leaf_idx,
                query_wide,
                req_ctx.max_dist(),
                &mut visitor,
            );
        });
    }

    /// Finds all points within a given distance of the query point.
    ///
    /// Returns all points within `max_dist` of the query point, unsorted.
    /// This is faster than `within` when order doesn't matter.
    #[inline]
    pub fn within_unsorted<D>(
        &self,
        query: &[A; K],
        max_dist: D::Output,
    ) -> Vec<NearestNeighbour<D::Output, T>>
    where
        D: KdTreeDistanceMetric<A, K>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static,
    {
        self.nearest_n_within::<D>(query, max_dist, NonZeroUsize::MAX, false)
    }

    /// Returns a streaming iterator over all points within a given distance, unsorted.
    ///
    /// This avoids materializing the full result set returned by
    /// [`within_unsorted`](Self::within_unsorted). The iterator keeps traversal state and
    /// per-leaf matches inline in the common case, spilling to heap allocation only if the
    /// tree depth or a single leaf's match count exceeds the inline capacities.
    ///
    /// For the absolute lowest overhead, use [`within_unsorted_visit`](Self::within_unsorted_visit).
    #[inline]
    pub fn within_unsorted_iter<D>(
        &self,
        query: &[A; K],
        max_dist: D::Output,
    ) -> crate::kd_tree::WithinUnsortedIter<'_, Self, A, T, SS, LS, D, K, B>
    where
        D: KdTreeDistanceMetric<A, K>,
        D::Output: crate::stem_strategy::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS> + 'static,
    {
        crate::kd_tree::WithinUnsortedIter::new(self, query, max_dist)
    }
}

struct WithinUnsortedVisitReqCtx<'a, A, O, const K: usize>
where
    O: AxisUnified<Coord = O>,
{
    query: &'a [A; K],
    max_dist: O,
    _phantom: std::marker::PhantomData<A>,
}

impl<A, O, const K: usize> QueryContext<A, O, K> for WithinUnsortedVisitReqCtx<'_, A, O, K>
where
    O: AxisUnified<Coord = O>,
{
    #[inline(always)]
    fn query(&self) -> &[A; K] {
        self.query
    }

    #[inline(always)]
    fn max_dist(&self) -> O {
        self.max_dist
    }
}

#[cfg(test)]
mod tests {
    use rand::rngs::StdRng;
    use rand::Rng;
    use rand::SeedableRng;
    use std::cmp::Ordering;

    use crate::dist::SquaredEuclidean;
    use crate::kd_tree::KdTree;
    use crate::leaf_strategy::{FlatVec, VecOfArenas, VecOfArrays};
    use crate::Eytzinger;
    use crate::traits_unified_2::AxisUnified;

    const RNG_SEED: u64 = 42;
    const TILE_BOUNDARY_CASES: [usize; 7] = [1, 2, 4, 8, 32, 33, 47];

    #[test]
    fn within_unsorted_vec_of_arenas_matches_flat_vec_across_tile_boundaries() {
        let query = [0.29f32, 0.41, 0.53];
        let radius = 0.2;

        for &len in &TILE_BOUNDARY_CASES {
            let points: Vec<[f32; 3]> = (0..len)
                .map(|idx| {
                    [
                        ((idx * 7) % 97) as f32 / 97.0,
                        ((idx * 17 + 1) % 97) as f32 / 97.0,
                        ((idx * 29 + 2) % 97) as f32 / 97.0,
                    ]
                })
                .collect();

            let flat_tree: KdTree<f32, u32, Eytzinger<3>, FlatVec<f32, u32, 3, 32>, 3, 32> =
                KdTree::new_from_slice(&points);
            let arena_tree: KdTree<f32, u32, Eytzinger<3>, VecOfArenas<f32, u32, 3, 32>, 3, 32> =
                KdTree::new_from_slice(&points);

            let mut flat: Vec<(f32, u32)> = flat_tree
                .within_unsorted::<SquaredEuclidean<f32>>(&query, radius)
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();
            let mut arena: Vec<(f32, u32)> = arena_tree
                .within_unsorted::<SquaredEuclidean<f32>>(&query, radius)
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();

            stabilize_sort(&mut flat);
            stabilize_sort(&mut arena);

            assert_eq!(arena, flat, "len={len}");
        }
    }

    #[test]
    fn within_unsorted_visit_and_iter_match_materialized_results() {
        let points: Vec<[f64; 3]> = (0..257)
            .map(|idx| {
                [
                    ((idx * 7) % 101) as f64 / 101.0,
                    ((idx * 17 + 3) % 101) as f64 / 101.0,
                    ((idx * 31 + 5) % 101) as f64 / 101.0,
                ]
            })
            .collect();
        type Tree = KdTree<f64, u32, Eytzinger<3>, VecOfArenas<f64, u32, 3, 32>, 3, 32>;
        let tree: Tree = KdTree::new_from_slice(&points);
        let query = [0.37, 0.41, 0.43];
        let radius = 0.12;

        let mut materialized: Vec<(u32, u64)> = tree
            .within_unsorted::<SquaredEuclidean<f64>>(&query, radius)
            .into_iter()
            .map(|result| (result.item, result.distance.to_bits()))
            .collect();
        let mut visited = Vec::new();
        tree.within_unsorted_visit::<SquaredEuclidean<f64>, _>(&query, radius, |result| {
            visited.push((result.item, result.distance.to_bits()));
        });
        let mut iterated: Vec<(u32, u64)> = tree
            .within_unsorted_iter::<SquaredEuclidean<f64>>(&query, radius)
            .map(|result| (result.item, result.distance.to_bits()))
            .collect();

        materialized.sort_unstable();
        visited.sort_unstable();
        iterated.sort_unstable();

        assert_eq!(visited, materialized);
        assert_eq!(iterated, materialized);
    }

    #[test]
    fn iter_visits_every_point_item_pair() {
        let points: Vec<[f64; 3]> = (0..129)
            .map(|idx| [idx as f64, (idx * 2) as f64, (idx * 3) as f64])
            .collect();
        type Tree = KdTree<f64, u32, Eytzinger<3>, VecOfArenas<f64, u32, 3, 32>, 3, 32>;
        let tree: Tree = KdTree::new_from_slice(&points);

        let visited: Vec<(u32, [f64; 3])> = tree.iter().collect();

        assert_eq!(visited.len(), points.len());
        for (item, point) in visited {
            assert_eq!(point, points[item as usize]);
        }
    }

    #[test]
    fn v6_query_within_unsorted_large_f32_flat_vec() {
        let mut rng = StdRng::seed_from_u64(RNG_SEED);

        const TREE_SIZE: usize = 100_000;
        const NUM_QUERIES: usize = 100;
        const RADIUS: f32 = 0.2;

        let content_to_add: Vec<[f32; 4]> =
            (0..TREE_SIZE).map(|_| rng.random::<[f32; 4]>()).collect();

        let tree: KdTree<f32, u32, Eytzinger<4>, FlatVec<f32, u32, 4, 32>, 4, 32> =
            KdTree::new_from_slice(&content_to_add);

        let query_points: Vec<[f32; 4]> =
            (0..NUM_QUERIES).map(|_| rng.random::<[f32; 4]>()).collect();

        for query_point in query_points {
            let expected = linear_search(&content_to_add, &query_point, RADIUS);

            let mut result: Vec<_> = tree
                .within_unsorted::<SquaredEuclidean<f32>>(&query_point, RADIUS)
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();

            stabilize_sort(&mut result);
            assert_eq!(result, expected);
        }
    }

    #[test]
    fn v6_query_within_unsorted_f32_flat_vec_no_items() {
        let mut rng = StdRng::seed_from_u64(RNG_SEED);

        const TREE_SIZE: usize = 1_000;
        const NUM_QUERIES: usize = 1;
        const RADIUS: f32 = 0.2;

        let content_to_add: Vec<[f32; 4]> =
            (0..TREE_SIZE).map(|_| rng.random::<[f32; 4]>()).collect();

        let tree: KdTree<f32, (), Eytzinger<4>, FlatVec<f32, (), 4, 32>, 4, 32> =
            KdTree::new_from_slice_no_items(&content_to_add);

        let query_points: Vec<[f32; 4]> =
            (0..NUM_QUERIES).map(|_| rng.random::<[f32; 4]>()).collect();

        for query_point in query_points {
            let expected: Vec<_> = linear_search(&content_to_add, &query_point, RADIUS)
                .into_iter()
                .map(|(distance, _)| (distance, ()))
                .collect();

            let mut result: Vec<_> = tree
                .within_unsorted::<SquaredEuclidean<f32>>(&query_point, RADIUS)
                .into_iter()
                .map(|n| (n.distance, 1))
                .collect();

            stabilize_sort(&mut result);

            let result: Vec<_> = result
                .into_iter()
                .map(|(distance, _)| (distance, ()))
                .collect();

            assert_eq!(result, expected);
        }
    }

    #[test]
    fn v6_query_within_unsorted_large_f32_vec_of_arrays() {
        let mut rng = StdRng::seed_from_u64(RNG_SEED);

        const TREE_SIZE: usize = 100_000;
        const NUM_QUERIES: usize = 100;
        const RADIUS: f32 = 0.2;

        let content_to_add: Vec<[f32; 4]> =
            (0..TREE_SIZE).map(|_| rng.random::<[f32; 4]>()).collect();

        let tree: KdTree<f32, u32, Eytzinger<4>, VecOfArrays<f32, u32, 4, 32>, 4, 32> =
            KdTree::new_from_slice(&content_to_add);

        let query_points: Vec<[f32; 4]> =
            (0..NUM_QUERIES).map(|_| rng.random::<[f32; 4]>()).collect();

        for query_point in query_points {
            let expected = linear_search(&content_to_add, &query_point, RADIUS);

            let mut result: Vec<_> = tree
                .within_unsorted::<SquaredEuclidean<f32>>(&query_point, RADIUS)
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();

            stabilize_sort(&mut result);
            assert_eq!(result, expected);
        }
    }

    #[test]
    fn v6_query_within_unsorted_large_f32_vec_of_arrays_mutated() {
        let mut rng = StdRng::seed_from_u64(RNG_SEED);

        const TREE_SIZE: usize = 100_000;
        const NUM_QUERIES: usize = 100;
        const RADIUS: f32 = 0.2;

        let content_to_add: Vec<[f32; 4]> =
            (0..TREE_SIZE).map(|_| rng.random::<[f32; 4]>()).collect();

        let mut tree: KdTree<f32, u32, Eytzinger<4>, VecOfArrays<f32, u32, 4, 32>, 4, 32> =
            KdTree::default();

        for (idx, point) in content_to_add.iter().enumerate() {
            tree.add(point, idx as u32);
        }

        let query_points: Vec<[f32; 4]> =
            (0..NUM_QUERIES).map(|_| rng.random::<[f32; 4]>()).collect();

        for query_point in query_points {
            let expected = linear_search(&content_to_add, &query_point, RADIUS);

            let mut result: Vec<_> = tree
                .within_unsorted::<SquaredEuclidean<f32>>(&query_point, RADIUS)
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();

            stabilize_sort(&mut result);
            assert_eq!(result, expected);
        }
    }

    fn linear_search<A, const K: usize>(
        content: &[[A; K]],
        query_point: &[A; K],
        radius: A,
    ) -> Vec<(A, u32)>
    where
        A: AxisUnified<Coord = A> + 'static,
        SquaredEuclidean<A>: crate::dist::DistanceMetricCore<A, Output = A>,
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

    fn squared_euclidean_dist<A, const K: usize>(a: &[A; K], b: &[A; K]) -> A
    where
        A: AxisUnified<Coord = A> + 'static,
        SquaredEuclidean<A>: crate::dist::DistanceMetricCore<A, Output = A>,
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

    fn stabilize_sort<A>(matching_items: &mut [(A, u32)])
    where
        A: AxisUnified<Coord = A> + 'static,
    {
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
