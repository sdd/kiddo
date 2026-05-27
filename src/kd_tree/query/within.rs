use std::num::NonZeroUsize;

use crate::dist::KdTreeDistanceMetric;
use crate::kd_tree::query_stack::StackTrait;
use crate::leaf_view::TlsLeafScratch;
use crate::stem_strategy::donnelly_2_blockmarker_simd::{
    BacktrackBlock3, BacktrackBlock4, SimdSelectBestChildBlock3,
};
use crate::{Axis, Content, KdTree, LeafStrategy, QueryResultItem, StemStrategy};

impl<A, T, SS, LS, const K: usize, const B: usize> KdTree<A, T, SS, LS, K, B>
where
    A: Axis<Coord = A> + 'static,
    T: Content + PartialOrd,
    LS: LeafStrategy<A, T, SS, K, B>,
    SS: StemStrategy,
{
    pub(crate) fn within_impl<D, const EXCLUSIVE: bool>(
        &self,
        query: &[A; K],
        max_dist: D::Output,
    ) -> Vec<QueryResultItem<(), T, D::Output>>
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
        self.nearest_n_within_impl::<D, EXCLUSIVE>(query, max_dist, NonZeroUsize::MAX, true)
    }
}

#[cfg(test)]
mod tests {
    use rand::rngs::StdRng;
    use rand::Rng;
    use rand::SeedableRng;
    use std::cmp::Ordering;

    use crate::dist::Manhattan;
    use crate::dist::SquaredEuclidean;
    use crate::kd_tree::KdTree;
    use crate::leaf_strategy::{FlatVec, VecOfArenas, VecOfArrays};
    use crate::Axis;
    use crate::Eytzinger;

    const RNG_SEED: u64 = 42;
    const TILE_BOUNDARY_CASES: [usize; 7] = [1, 2, 4, 8, 32, 33, 47];

    #[test]
    fn within_exclusive_boundaries_excludes_exact_threshold_matches() {
        let points = vec![[0.0f64, 0.0], [1.0, 0.0], [2.0, 0.0], [0.5, 0.0]];
        let tree: KdTree<f64, u32, Eytzinger<2>, FlatVec<f64, u32, 2, 32>, 2, 32> =
            KdTree::new_from_slice(&points).unwrap();
        let query = [0.0, 0.0];

        let inclusive: Vec<_> = tree
            .query(&query)
            .within::<SquaredEuclidean<f64>>(1.0)
            .execute()
            .into_iter()
            .map(|n| n.item)
            .collect();
        let exclusive: Vec<_> = tree
            .query(&query)
            .within::<SquaredEuclidean<f64>>(1.0)
            .exclusive_boundaries()
            .execute()
            .into_iter()
            .map(|n| n.item)
            .collect();

        assert_eq!(inclusive, vec![0, 3, 1]);
        assert_eq!(exclusive, vec![0, 3]);
    }

    #[test]
    fn within_vec_of_arenas_matches_flat_vec_across_tile_boundaries() {
        let query = [0.31f32, 0.47, 0.59];
        let radius = 0.35;

        for &len in &TILE_BOUNDARY_CASES {
            let points: Vec<[f32; 3]> = (0..len)
                .map(|idx| {
                    [
                        ((idx * 5) % 97) as f32 / 97.0,
                        ((idx * 13 + 1) % 97) as f32 / 97.0,
                        ((idx * 23 + 2) % 97) as f32 / 97.0,
                    ]
                })
                .collect();

            let flat_tree: KdTree<f32, u32, Eytzinger<3>, FlatVec<f32, u32, 3, 32>, 3, 32> =
                KdTree::new_from_slice(&points).unwrap();
            let arena_tree: KdTree<f32, u32, Eytzinger<3>, VecOfArenas<f32, u32, 3, 32>, 3, 32> =
                KdTree::new_from_slice(&points).unwrap();

            let mut flat: Vec<(f32, u32)> = flat_tree
                .query(&query)
                .within::<Manhattan<f32>>(radius)
                .execute()
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();
            let mut arena: Vec<(f32, u32)> = arena_tree
                .query(&query)
                .within::<Manhattan<f32>>(radius)
                .execute()
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();

            stabilize_sort(&mut flat);
            stabilize_sort(&mut arena);

            assert_eq!(arena, flat, "len={len}");
        }
    }

    #[test]
    fn can_query_items_within_radius_large_scale() {
        let mut rng = StdRng::seed_from_u64(RNG_SEED);

        const TREE_SIZE: usize = 100_000;
        const NUM_QUERIES: usize = 100;
        const RADIUS: f32 = 0.2;

        let content_to_add: Vec<[f32; 4]> =
            (0..TREE_SIZE).map(|_| rng.random::<[f32; 4]>()).collect();

        let tree: KdTree<f32, u32, Eytzinger<4>, FlatVec<f32, u32, 4, 32>, 4, 32> =
            KdTree::new_from_slice(&content_to_add).unwrap();

        let query_points: Vec<[f32; 4]> =
            (0..NUM_QUERIES).map(|_| rng.random::<[f32; 4]>()).collect();

        for query_point in query_points {
            let expected = linear_search(&content_to_add, &query_point, RADIUS);

            let mut result: Vec<_> = tree
                .query(&query_point)
                .within::<Manhattan<f32>>(RADIUS)
                .execute()
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();

            stabilize_sort(&mut result);

            assert_eq!(result, expected);
        }
    }

    #[test]
    fn v6_query_within_large_vec_of_arrays_f32() {
        let mut rng = StdRng::seed_from_u64(RNG_SEED);

        const TREE_SIZE: usize = 100_000;
        const NUM_QUERIES: usize = 100;
        const RADIUS: f32 = 0.2;

        let content_to_add: Vec<[f32; 4]> =
            (0..TREE_SIZE).map(|_| rng.random::<[f32; 4]>()).collect();

        let tree: KdTree<f32, u32, Eytzinger<4>, VecOfArrays<f32, u32, 4, 32>, 4, 32> =
            KdTree::new_from_slice(&content_to_add).unwrap();

        let query_points: Vec<[f32; 4]> =
            (0..NUM_QUERIES).map(|_| rng.random::<[f32; 4]>()).collect();

        for query_point in query_points {
            let expected = linear_search(&content_to_add, &query_point, RADIUS);

            let mut result: Vec<_> = tree
                .query(&query_point)
                .within::<Manhattan<f32>>(RADIUS)
                .execute()
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();

            stabilize_sort(&mut result);

            assert_eq!(result, expected);
        }
    }

    #[test]
    fn v6_query_within_vec_of_arrays_f32_no_items() {
        let mut rng = StdRng::seed_from_u64(RNG_SEED);

        const TREE_SIZE: usize = 1_000;
        const NUM_QUERIES: usize = 1;
        const RADIUS: f32 = 0.2;

        let content_to_add: Vec<[f32; 4]> =
            (0..TREE_SIZE).map(|_| rng.random::<[f32; 4]>()).collect();

        let tree: KdTree<f32, (), Eytzinger<4>, VecOfArrays<f32, (), 4, 32>, 4, 32> =
            KdTree::new_from_slice_no_items(&content_to_add).unwrap();

        let query_points: Vec<[f32; 4]> =
            (0..NUM_QUERIES).map(|_| rng.random::<[f32; 4]>()).collect();

        for query_point in query_points {
            let expected: Vec<_> = linear_search(&content_to_add, &query_point, RADIUS)
                .into_iter()
                .map(|(distance, _)| (distance, ()))
                .collect();

            let mut result: Vec<_> = tree
                .query(&query_point)
                .within::<Manhattan<f32>>(RADIUS)
                .execute()
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
    fn v6_query_within_large_vec_of_arrays_mutated_f32() {
        let mut rng = StdRng::seed_from_u64(RNG_SEED);

        const TREE_SIZE: usize = 100_000;
        const NUM_QUERIES: usize = 100;
        const RADIUS: f32 = 0.2;

        let content_to_add: Vec<[f32; 4]> =
            (0..TREE_SIZE).map(|_| rng.random::<[f32; 4]>()).collect();

        let mut tree: KdTree<f32, u32, Eytzinger<4>, VecOfArrays<f32, u32, 4, 32>, 4, 32> =
            KdTree::default();

        for (idx, point) in content_to_add.iter().enumerate() {
            tree.add(point, idx as u32).unwrap();
        }

        let query_points: Vec<[f32; 4]> =
            (0..NUM_QUERIES).map(|_| rng.random::<[f32; 4]>()).collect();

        for query_point in query_points {
            let expected = linear_search(&content_to_add, &query_point, RADIUS);

            let mut result: Vec<_> = tree
                .query(&query_point)
                .within::<Manhattan<f32>>(RADIUS)
                .execute()
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
        A: Axis<Coord = A> + 'static,
        Manhattan<A>: crate::dist::DistanceMetricCore<A, Output = A>,
    {
        let mut matching_items = vec![];

        for (idx, p) in content.iter().enumerate() {
            let dist = manhattan_dist(query_point, p);
            if dist <= radius {
                matching_items.push((dist, idx as u32));
            }
        }

        stabilize_sort(&mut matching_items);

        matching_items
    }

    fn manhattan_dist<A, const K: usize>(a: &[A; K], b: &[A; K]) -> A
    where
        A: Axis<Coord = A>,
        Manhattan<A>: crate::dist::DistanceMetricCore<A, Output = A>,
    {
        let aw = (*a)
            .map(|coord| <Manhattan<A> as crate::dist::DistanceMetricCore<A>>::widen_coord(coord));
        let bw = (*b)
            .map(|coord| <Manhattan<A> as crate::dist::DistanceMetricCore<A>>::widen_coord(coord));

        <Manhattan<A> as crate::dist::DistanceMetricCore<A>>::dist::<K>(&aw, &bw)
    }

    fn stabilize_sort<A>(matching_items: &mut [(A, u32)])
    where
        A: Axis<Coord = A>,
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
