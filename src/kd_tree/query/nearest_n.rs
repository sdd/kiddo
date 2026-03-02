use crate::kd_tree::query_stack::StackTrait;
use crate::kd_tree::KdTree;
use crate::stem_strategies::donnelly_2_blockmarker_simd::{BacktrackBlock3, BacktrackBlock4};
use crate::traits_unified_2::{AxisUnified, Basics, DistanceMetricUnified, LeafStrategy};
use crate::{NearestNeighbour, StemStrategy};
use std::num::NonZero;

impl<A, T, SS, LS, const K: usize, const B: usize> KdTree<A, T, SS, LS, K, B>
where
    A: AxisUnified<Coord = A>,
    T: Basics + Ord,
    LS: LeafStrategy<A, T, SS, K, B>,
    SS: StemStrategy,
{
    /// Finds the N nearest points to the query point.
    ///
    /// If `sorted` is true, results are returned in order of increasing distance.
    pub fn nearest_n<D>(
        &self,
        query: &[A; K],
        max_qty: NonZero<usize>,
        sorted: bool,
    ) -> Vec<NearestNeighbour<D::Output, T>>
    where
        D: DistanceMetricUnified<A, K>
            + crate::stem_strategies::DistanceMetricSimdBlock3<A, K, D::Output>
            + crate::stem_strategies::DistanceMetricSimdBlock4<A, K, D::Output>,
        D::Output: crate::stem_strategies::SimdPrune + BacktrackBlock3 + BacktrackBlock4,
        SS::Stack<D::Output>: StackTrait<D::Output, SS>,
    {
        self.nearest_n_within::<D>(query, D::Output::max_value(), max_qty, sorted)
    }
}

#[cfg(test)]
mod tests {
    use az::{Az, Cast};
    use rand::rngs::StdRng;
    use rand::Rng;
    use rand::SeedableRng;
    use std::array;
    use std::num::NonZero;

    use crate::kd_tree::leaf_strategies::{FlatVec, VecOfArrays};
    use crate::kd_tree::KdTree;
    use crate::traits::Axis;
    use crate::traits::DistanceMetric;
    use crate::traits_unified_2::SquaredEuclidean;
    use crate::Eytzinger;

    const RNG_SEED: u64 = 42;

    #[test]
    fn v6_query_nearest_n_large_f64_flat_vec() {
        let mut rng = StdRng::seed_from_u64(RNG_SEED);

        const TREE_SIZE: usize = 100_000;
        const NUM_QUERIES: usize = 100;

        let max_qty = NonZero::new(10).unwrap();

        let content_to_add: Vec<[f64; 4]> =
            (0..TREE_SIZE).map(|_| rng.random::<[f64; 4]>()).collect();

        let tree: KdTree<f64, u32, Eytzinger<4>, FlatVec<f64, u32, 4, 32>, 4, 32> =
            KdTree::new_from_slice(&content_to_add);

        assert_eq!(tree.size(), TREE_SIZE);

        let query_points: Vec<[f64; 4]> =
            (0..NUM_QUERIES).map(|_| rng.random::<[f64; 4]>()).collect();

        for query_point in query_points {
            let expected = linear_search(&content_to_add, max_qty.into(), &query_point);

            let result: Vec<_> = tree
                .nearest_n::<SquaredEuclidean<f64>>(&query_point, max_qty, true)
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();

            assert_eq!(result, expected);
        }
    }

    #[test]
    fn v6_query_nearest_n_large_f64_vec_of_arrays() {
        let mut rng = StdRng::seed_from_u64(RNG_SEED);

        const TREE_SIZE: usize = 100_000;
        const NUM_QUERIES: usize = 100;

        let max_qty = NonZero::new(10).unwrap();

        let content_to_add: Vec<[f64; 4]> =
            (0..TREE_SIZE).map(|_| rng.random::<[f64; 4]>()).collect();

        let tree: KdTree<f64, u32, Eytzinger<4>, VecOfArrays<f64, u32, 4, 32>, 4, 32> =
            KdTree::new_from_slice(&content_to_add);

        assert_eq!(tree.size(), TREE_SIZE);

        let query_points: Vec<[f64; 4]> =
            (0..NUM_QUERIES).map(|_| rng.random::<[f64; 4]>()).collect();

        for query_point in query_points {
            let expected = linear_search(&content_to_add, max_qty.into(), &query_point);

            let result: Vec<_> = tree
                .nearest_n::<SquaredEuclidean<f64>>(&query_point, max_qty, true)
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();

            assert_eq!(result, expected);
        }
    }

    #[test]
    fn v6_query_nearest_n_large_f64_vec_of_arrays_mutated_f64() {
        let mut rng = StdRng::seed_from_u64(RNG_SEED);

        const TREE_SIZE: usize = 100_000;
        const NUM_QUERIES: usize = 100;

        let max_qty = NonZero::new(10).unwrap();

        let content_to_add: Vec<[f64; 4]> =
            (0..TREE_SIZE).map(|_| rng.random::<[f64; 4]>()).collect();

        let mut tree: KdTree<f64, u32, Eytzinger<4>, VecOfArrays<f64, u32, 4, 32>, 4, 32> =
            KdTree::default();

        for (idx, point) in content_to_add.iter().enumerate() {
            tree.add(point, idx as u32);
        }

        assert_eq!(tree.size(), TREE_SIZE);

        let query_points: Vec<[f64; 4]> =
            (0..NUM_QUERIES).map(|_| rng.random::<[f64; 4]>()).collect();

        for query_point in query_points {
            let expected = linear_search(&content_to_add, max_qty.into(), &query_point);

            let result: Vec<_> = tree
                .nearest_n::<SquaredEuclidean<f64>>(&query_point, max_qty, true)
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();

            assert_eq!(result, expected);
        }
    }

    fn linear_search<A: Axis, R, const K: usize>(
        content: &[[A; K]],
        qty: usize,
        query_point: &[A; K],
    ) -> Vec<(A, R)>
    where
        usize: Cast<R>,
    {
        let mut results: Vec<(A, R)> = vec![];

        for (idx, p) in content.iter().enumerate() {
            let dist = crate::SquaredEuclidean::dist(query_point, p);
            if results.len() < qty {
                results.push((dist, idx.az::<R>()));
                results.sort_by(|(a_dist, _), (b_dist, _)| a_dist.partial_cmp(b_dist).unwrap());
            } else if dist < results[qty - 1].0 {
                results[qty - 1] = (dist, idx.az::<R>());
                results.sort_by(|(a_dist, _), (b_dist, _)| a_dist.partial_cmp(b_dist).unwrap());
            }
        }

        results
    }

    fn random_point_f32<const K: usize>(rng: &mut StdRng) -> [f32; K] {
        array::from_fn(|_| rng.random_range(-1.0f32..1.0f32))
    }

    fn random_point_f64<const K: usize>(rng: &mut StdRng) -> [f64; K] {
        array::from_fn(|_| rng.random_range(-1.0f64..1.0f64))
    }

    fn sort_by_distance_then_item_f32(items: &mut [(f32, usize)]) {
        items.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then_with(|| a.1.cmp(&b.1)));
    }

    fn sort_by_distance_then_item_f64(items: &mut [(f64, usize)]) {
        items.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then_with(|| a.1.cmp(&b.1)));
    }

    #[test]
    fn v6_query_nearest_n_unsorted_respects_max_qty_mutable_f32_regression() {
        const K: usize = 2;
        const B: usize = 32;
        const POINT_COUNT: usize = 1024;
        const CONTENT_SEED: u64 = 1_260_253_197;
        const QUERY_SEED: u64 = 13_787_848_794_416_797_126;

        let mut rng_content = StdRng::seed_from_u64(CONTENT_SEED);
        let points: Vec<[f32; K]> = (0..POINT_COUNT)
            .map(|_| random_point_f32::<K>(&mut rng_content))
            .collect();

        let mut tree: KdTree<f32, usize, Eytzinger<K>, VecOfArrays<f32, usize, K, B>, K, B> =
            KdTree::default();
        for (idx, point) in points.iter().enumerate() {
            tree.add(point, idx);
        }

        let mut rng_query = StdRng::seed_from_u64(QUERY_SEED);
        let query = random_point_f32::<K>(&mut rng_query);
        let max_qty = NonZero::new(20).unwrap();

        let mut unsorted: Vec<(f32, usize)> = tree
            .nearest_n::<SquaredEuclidean<f32>>(&query, max_qty, false)
            .into_iter()
            .map(|n| (n.distance, n.item))
            .collect();

        assert_eq!(unsorted.len(), max_qty.get());

        let mut sorted: Vec<(f32, usize)> = tree
            .nearest_n::<SquaredEuclidean<f32>>(&query, max_qty, true)
            .into_iter()
            .map(|n| (n.distance, n.item))
            .collect();

        sort_by_distance_then_item_f32(&mut unsorted);
        sort_by_distance_then_item_f32(&mut sorted);

        assert_eq!(unsorted, sorted);
    }

    #[test]
    fn v6_query_nearest_n_unsorted_respects_max_qty_mutable_f64_regression() {
        const K: usize = 2;
        const B: usize = 32;
        const POINT_COUNT: usize = 1024;
        const CONTENT_SEED: u64 = 1_260_253_197;
        const QUERY_SEED: u64 = 13_787_848_794_416_797_126;

        let mut rng_content = StdRng::seed_from_u64(CONTENT_SEED);
        let points: Vec<[f64; K]> = (0..POINT_COUNT)
            .map(|_| random_point_f64::<K>(&mut rng_content))
            .collect();

        let mut tree: KdTree<f64, usize, Eytzinger<K>, VecOfArrays<f64, usize, K, B>, K, B> =
            KdTree::default();
        for (idx, point) in points.iter().enumerate() {
            tree.add(point, idx);
        }

        let mut rng_query = StdRng::seed_from_u64(QUERY_SEED);
        let query = random_point_f64::<K>(&mut rng_query);
        let max_qty = NonZero::new(15).unwrap();

        let mut unsorted: Vec<(f64, usize)> = tree
            .nearest_n::<SquaredEuclidean<f64>>(&query, max_qty, false)
            .into_iter()
            .map(|n| (n.distance, n.item))
            .collect();

        assert_eq!(unsorted.len(), max_qty.get());

        let mut sorted: Vec<(f64, usize)> = tree
            .nearest_n::<SquaredEuclidean<f64>>(&query, max_qty, true)
            .into_iter()
            .map(|n| (n.distance, n.item))
            .collect();

        sort_by_distance_then_item_f64(&mut unsorted);
        sort_by_distance_then_item_f64(&mut sorted);

        assert_eq!(unsorted, sorted);
    }
}
