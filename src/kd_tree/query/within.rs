use crate::kd_tree::query_stack::StackTrait;
use crate::kd_tree::KdTree;
use crate::traits_unified_2::{AxisUnified, Basics, DistanceMetricUnified, LeafStrategy};
use crate::{NearestNeighbour, StemStrategy};
use std::num::NonZeroUsize;

impl<A, T, SS, LS, const K: usize, const B: usize> KdTree<A, T, SS, LS, K, B>
where
    A: AxisUnified<Coord = A>,
    T: Basics + Ord,
    LS: LeafStrategy<A, T, SS, K, B>,
    SS: StemStrategy,
{
    /// Finds all points within a given distance of the query point.
    ///
    /// Returns all points within `max_dist` of the query point, sorted by distance.
    pub fn within<D>(
        &self,
        query: &[A; K],
        max_dist: D::Output,
    ) -> Vec<NearestNeighbour<D::Output, T>>
    where
        D: DistanceMetricUnified<A, K>,
        SS::Stack<D::Output>: StackTrait<D::Output, SS>,
    {
        self.nearest_n_within::<D>(query, max_dist, NonZeroUsize::MAX, true)
    }
}

#[cfg(test)]
mod tests {
    use rand::rngs::StdRng;
    use rand::Rng;
    use rand::SeedableRng;
    use std::cmp::Ordering;

    use crate::kd_tree::leaf_strategies::{FlatVec, VecOfArrays};
    use crate::kd_tree::KdTree;
    use crate::traits::Axis;
    use crate::traits::DistanceMetric;
    use crate::traits_unified_2::Manhattan;
    use crate::Eytzinger;

    const RNG_SEED: u64 = 42;

    #[test]
    fn can_query_items_within_radius_large_scale() {
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
                .within::<Manhattan<f32>>(&query_point, RADIUS)
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
            KdTree::new_from_slice(&content_to_add);

        let query_points: Vec<[f32; 4]> =
            (0..NUM_QUERIES).map(|_| rng.random::<[f32; 4]>()).collect();

        for query_point in query_points {
            let expected = linear_search(&content_to_add, &query_point, RADIUS);

            let mut result: Vec<_> = tree
                .within::<Manhattan<f32>>(&query_point, RADIUS)
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();

            stabilize_sort(&mut result);

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
            tree.add(point, idx as u32);
        }

        let query_points: Vec<[f32; 4]> =
            (0..NUM_QUERIES).map(|_| rng.random::<[f32; 4]>()).collect();

        for query_point in query_points {
            let expected = linear_search(&content_to_add, &query_point, RADIUS);

            let mut result: Vec<_> = tree
                .within::<Manhattan<f32>>(&query_point, RADIUS)
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
            let dist = crate::Manhattan::dist(query_point, p);
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
