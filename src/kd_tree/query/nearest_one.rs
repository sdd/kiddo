use crate::kd_tree::query_stack::StackTrait;
use crate::kd_tree::traits::QueryContext;
use crate::kd_tree::KdTree;
use crate::traits_unified_2::{AxisUnified, Basics, DistanceMetricUnified, LeafStrategy};
use crate::StemStrategy;

impl<A, T, SS, LS, const K: usize, const B: usize> KdTree<A, T, SS, LS, K, B>
where
    A: AxisUnified<Coord = A>,
    T: Basics + Copy + Default + PartialOrd + PartialEq,
    LS: LeafStrategy<A, T, SS, K, B>,
    SS: StemStrategy,
{
    /// Finds the nearest point to the query point.
    ///
    /// Returns a tuple of (distance, item) for the nearest neighbor.
    pub fn nearest_one<D>(&self, query: &[A; K]) -> (D::Output, T)
    where
        D: DistanceMetricUnified<A, K>,
        SS::Stack<D::Output>: StackTrait<D::Output, SS>,
    {
        let mut req_ctx = NearestOneReqCtx {
            query,
            best_dist: D::Output::max_value(),
            best_item: T::default(),
        };

        self.backtracking_query::<_, _, D>(&mut req_ctx, |leaf, query_ctx| {
            leaf.nearest_one::<D>(query, &mut query_ctx.best_dist, &mut query_ctx.best_item);
        });

        (req_ctx.best_dist, req_ctx.best_item)
    }
}

pub(crate) struct NearestOneReqCtx<'a, A, T, O, const K: usize>
where
    O: AxisUnified<Coord = O>,
{
    query: &'a [A; K],
    best_dist: O,
    best_item: T,
}

impl<A, T, O, const K: usize> QueryContext<A, O, K> for NearestOneReqCtx<'_, A, T, O, K>
where
    O: AxisUnified<Coord = O>,
{
    fn query(&self) -> &[A; K] {
        self.query
    }

    fn max_dist(&self) -> O {
        self.best_dist
    }
}

#[cfg(test)]
mod tests {
    use rand::Rng;
    use rand::SeedableRng;
    use test_log::test;

    use crate::kd_tree::leaf_strategies::{FlatVec, VecOfArrays};
    use crate::kd_tree::KdTree;
    use crate::stem_strategies::Donnelly;
    use crate::traits::{Axis, DistanceMetric};
    use crate::traits_unified_2::SquaredEuclidean;
    use crate::{Eytzinger, NearestNeighbour};

    #[test]
    fn v6_query_nearest_one_small_f64_flat_vec_eytzinger() {
        let content_to_add: [[f64; 4]; 16] = [
            [0.9f64, 0.0f64, 0.9f64, 0.0f64],
            [0.4f64, 0.5f64, 0.4f64, 0.51f64],
            [0.12f64, 0.3f64, 0.12f64, 0.3f64],
            [0.7f64, 0.2f64, 0.7f64, 0.22f64],
            [0.13f64, 0.4f64, 0.13f64, 0.4f64],
            [0.6f64, 0.3f64, 0.6f64, 0.33f64],
            [0.2f64, 0.7f64, 0.2f64, 0.7f64],
            [0.14f64, 0.5f64, 0.14f64, 0.5f64],
            [0.3f64, 0.6f64, 0.3f64, 0.6f64],
            [0.10f64, 0.1f64, 0.10f64, 0.1f64],
            [0.16f64, 0.7f64, 0.16f64, 0.7f64],
            [0.1f64, 0.8f64, 0.1f64, 0.8f64],
            [0.15f64, 0.6f64, 0.15f64, 0.6f64],
            [0.5f64, 0.4f64, 0.5f64, 0.44f64],
            [0.8f64, 0.1f64, 0.8f64, 0.15f64],
            [0.11f64, 0.2f64, 0.11f64, 0.2f64],
        ];

        let tree: KdTree<f64, u32, Eytzinger<4>, FlatVec<f64, u32, 4, 32>, 4, 32> =
            KdTree::new_from_slice(&content_to_add);

        assert!(!tree.is_empty());
        assert_eq!(tree.size(), 16);

        let query_point = [0.78f64, 0.55f64, 0.78f64, 0.55f64];

        let expected = (0.17570000000000008, 5);

        let results = tree.nearest_one::<SquaredEuclidean<f64>>(&query_point);
        assert_eq!(results, expected);
    }

    #[test]
    fn v6_query_nearest_one_large_f32_flatvec_eytzinger() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(3);

        const TREE_SIZE: usize = 100_000;
        const NUM_QUERIES: usize = 1000;

        let content_to_add: Vec<[f32; 4]> =
            (0..TREE_SIZE).map(|_| rng.random::<[f32; 4]>()).collect();

        let tree: KdTree<f32, u32, Eytzinger<4>, FlatVec<f32, u32, 4, 32>, 4, 32> =
            KdTree::new_from_slice(&content_to_add);

        assert_eq!(tree.size(), TREE_SIZE);

        let query_points: Vec<[f32; 4]> = (0..NUM_QUERIES)
            .map(|_| rng.random::<[f32; 4]>()) // Use the seeded rng
            .collect();

        for (_i, query_point) in query_points.iter().enumerate() {
            let expected = linear_search(&content_to_add, query_point);
            let result = tree.nearest_one::<SquaredEuclidean<f32>>(query_point);

            assert_eq!(result.0, expected.distance);
            assert_eq!(result.1 as usize, expected.item);
        }
    }

    #[test]
    fn v6_query_nearest_one_large_f32_flatvec_donnelly() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(3);

        const TREE_SIZE: usize = 100_000;
        const NUM_QUERIES: usize = 1000;

        let content_to_add: Vec<[f32; 4]> =
            (0..TREE_SIZE).map(|_| rng.random::<[f32; 4]>()).collect();

        let tree: KdTree<f32, u32, Donnelly<4, 64, 4, 4>, FlatVec<f32, u32, 4, 32>, 4, 32> =
            KdTree::new_from_slice(&content_to_add);

        assert_eq!(tree.size(), TREE_SIZE);

        let query_points: Vec<[f32; 4]> = (0..NUM_QUERIES)
            .map(|_| rng.random::<[f32; 4]>()) // Use the seeded rng
            .collect();

        for (_i, query_point) in query_points.iter().enumerate() {
            let expected = linear_search(&content_to_add, query_point);
            let result = tree.nearest_one::<SquaredEuclidean<f32>>(query_point);

            assert_eq!(result.0, expected.distance);
            assert_eq!(result.1 as usize, expected.item);
        }
    }

    #[test]
    fn v6_query_nearest_one_large_f32_vec_of_arrays_eytzinger() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(3);

        const TREE_SIZE: usize = 100_000;
        const NUM_QUERIES: usize = 1000;

        let content_to_add: Vec<[f32; 4]> =
            (0..TREE_SIZE).map(|_| rng.random::<[f32; 4]>()).collect();

        let tree: KdTree<f32, u32, Eytzinger<4>, VecOfArrays<f32, u32, 4, 32>, 4, 32> =
            KdTree::new_from_slice(&content_to_add);

        assert_eq!(tree.size(), TREE_SIZE);

        let query_points: Vec<[f32; 4]> = (0..NUM_QUERIES)
            .map(|_| rng.random::<[f32; 4]>()) // Use the seeded rng
            .collect();

        // println!("Tree: {}", &tree);

        for (_i, query_point) in query_points.iter().enumerate() {
            let expected = linear_search(&content_to_add, query_point);
            let result = tree.nearest_one::<SquaredEuclidean<f32>>(query_point);

            assert_eq!(result.0, expected.distance);
            assert_eq!(result.1 as usize, expected.item);
        }
    }

    #[test]
    fn v6_query_nearest_one_large_f32_vec_of_arrays_mutated_eytzinger() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(3);

        const TREE_SIZE: usize = 100_000;
        const NUM_QUERIES: usize = 1000;

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

        for (i, query_point) in query_points.iter().enumerate() {
            let expected = linear_search(&content_to_add, query_point);
            let result = tree.nearest_one::<SquaredEuclidean<f32>>(query_point);

            assert_eq!(
                result.0, expected.distance,
                "Incorrect distance, query index: {i}"
            );
            assert_eq!(
                result.1 as usize, expected.item,
                "Incorrect item, query index: {i}"
            );
        }
    }

    #[test]
    fn v6_query_nearest_one_large_f32_vec_of_arrays_mutated_donnelly() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(3);

        const TREE_SIZE: usize = 100_000;
        const NUM_QUERIES: usize = 1000;

        let content_to_add: Vec<[f32; 4]> =
            (0..TREE_SIZE).map(|_| rng.random::<[f32; 4]>()).collect();

        let mut tree: KdTree<f32, u32, Donnelly<4, 64, 4, 4>, VecOfArrays<f32, u32, 4, 32>, 4, 32> =
            KdTree::default();

        for (idx, point) in content_to_add.iter().enumerate() {
            tree.add(point, idx as u32);
        }

        assert_eq!(tree.size(), TREE_SIZE);

        let query_points: Vec<[f32; 4]> = (0..NUM_QUERIES)
            .map(|_| rng.random::<[f32; 4]>()) // Use the seeded rng
            .collect();

        for (_i, query_point) in query_points.iter().enumerate() {
            let expected = linear_search(&content_to_add, query_point);
            let result = tree.nearest_one::<SquaredEuclidean<f32>>(query_point);

            assert_eq!(result.0, expected.distance);
            assert_eq!(result.1 as usize, expected.item);
        }
    }

    #[test]
    fn v6_query_nearest_one_large_f32_vec_of_arrays_donnelly() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(3);

        const TREE_SIZE: usize = 100_000;
        const NUM_QUERIES: usize = 1000;

        let content_to_add: Vec<[f32; 4]> =
            (0..TREE_SIZE).map(|_| rng.random::<[f32; 4]>()).collect();

        let tree: KdTree<f32, u32, Donnelly<4, 64, 4, 4>, VecOfArrays<f32, u32, 4, 32>, 4, 32> =
            KdTree::new_from_slice(&content_to_add);

        assert_eq!(tree.size(), TREE_SIZE);

        let query_points: Vec<[f32; 4]> = (0..NUM_QUERIES)
            .map(|_| rng.random::<[f32; 4]>()) // Use the seeded rng
            .collect();

        for (_i, query_point) in query_points.iter().enumerate() {
            let expected = linear_search(&content_to_add, query_point);
            let result = tree.nearest_one::<SquaredEuclidean<f32>>(query_point);

            assert_eq!(result.0, expected.distance);
            assert_eq!(result.1 as usize, expected.item);
        }
    }

    fn linear_search<A: Axis, const K: usize>(
        content: &[[A; K]],
        query_point: &[A; K],
    ) -> NearestNeighbour<A, usize> {
        let mut best_dist: A = A::infinity();
        let mut best_item: usize = usize::MAX;

        for (idx, p) in content.iter().enumerate() {
            let dist = crate::SquaredEuclidean::dist(query_point, p);
            if dist < best_dist {
                best_item = idx;
                best_dist = dist;
            }
        }

        NearestNeighbour {
            distance: best_dist,
            item: best_item,
        }
    }

    #[test]
    #[cfg(feature = "simd")]
    #[cfg(target_arch = "x86_64")]
    fn v6_query_nearest_one_donnelly_marker_simd_f64() {
        use crate::stem_strategies::{Block3, DonnellyMarkerPf, DonnellyMarkerSimd};

        // Test DonnellyMarkerSimd with f64 data using exact nearest_one query
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);

        // Use 8192 points which with bucket size 32 gives 256 leaves
        // 256 leaves = 2^8, so tree depth = 8
        // With Block3, depth 8 is not divisible by 3, so tree will be padded to depth 9
        let points: Vec<[f64; 3]> = (0..2_048) // 8_192)
            .map(|_| {
                [
                    rng.random::<f64>(),
                    rng.random::<f64>(),
                    rng.random::<f64>(),
                ]
            })
            .collect();

        let tree: KdTree<
            f64,
            u32,
            DonnellyMarkerSimd<Block3, 64, 8, 3>,
            FlatVec<f64, u32, 3, 32>,
            3,
            32,
        > = KdTree::new_from_slice(&points);

        let tree_non_simd: KdTree<
            f64,
            u32,
            DonnellyMarkerPf<Block3, 64, 8, 3>,
            FlatVec<f64, u32, 3, 32>,
            3,
            32,
        > = KdTree::new_from_slice(&points);

        assert!(!tree.is_empty());
        assert_eq!(tree.size(), 2_048);
        assert_eq!(tree.leaf_count(), 64);

        // Verify max_stem_level is padded to multiple of block size (3)
        // 256 leaves = depth 8, padded to 9
        assert_eq!((tree.max_stem_level() + 1) % 3, 0);
        assert_eq!(tree.max_stem_level(), 5);

        // println!("NON-SIMD: {}", tree_non_simd);
        // println!("SIMD: {}", tree);

        // Test multiple query points to ensure backtracking queries work correctly
        let query_points: Vec<[f64; 3]> = (0..50)
            .map(|_| {
                [
                    rng.random::<f64>(),
                    rng.random::<f64>(),
                    rng.random::<f64>(),
                ]
            })
            .collect();

        for (i, query_point) in query_points.iter().enumerate() {
            // tracing::debug!("Query point: #{i} ({query_point:?})");

            let expected = linear_search(&points, query_point);
            // println!("\n========== QUERY #{i} ==========");
            // println!("Query point: {:?}", query_point);
            // println!("Expected: item={}, dist²={}", expected.item, expected.distance);

            let _result = tree_non_simd.nearest_one::<SquaredEuclidean<f64>>(query_point);
            // println!("NON-SIMD: item={}, dist²={}", result.1, result.0);

            let result = tree.nearest_one::<SquaredEuclidean<f64>>(query_point);
            // println!("SIMD: item={}, dist²={}", result.1, result.0);

            assert_eq!(
                result.0, expected.distance,
                "Distance mismatch for query #{} ({:?})",
                i, query_point
            );
            assert_eq!(
                result.1 as usize, expected.item,
                "Item mismatch for query #{} ({:?})",
                i, query_point
            );
        }
    }

    #[test]
    #[cfg(feature = "simd")]
    #[cfg(target_arch = "x86_64")]
    fn v6_query_nearest_one_donnelly_marker_simd_block3_f32() {
        use crate::stem_strategies::{Block3, DonnellyMarkerSimd};

        // Test DonnellyMarkerSimd with f32 data using exact nearest_one query
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);

        // Use smaller dataset for faster test (16384 points = 512 leaves = 2^9, depth = 9)
        // Block4 doesn't divide evenly into 9, will be padded to 12
        let points: Vec<[f32; 4]> = (0..16_384)
            .map(|_| {
                [
                    rng.random::<f32>(),
                    rng.random::<f32>(),
                    rng.random::<f32>(),
                    rng.random::<f32>(),
                ]
            })
            .collect();

        let tree: KdTree<
            f32,
            u32,
            DonnellyMarkerSimd<Block3, 64, 4, 4>,
            FlatVec<f32, u32, 4, 32>,
            4,
            32,
        > = KdTree::new_from_slice(&points);

        assert!(!tree.is_empty());
        assert_eq!(tree.size(), 16_384);
        assert_eq!(tree.leaf_count(), 512);

        // Verify max_stem_level is padded to multiple of block size (3)
        assert_eq!((tree.max_stem_level() + 1) % 3, 0);

        // Test multiple query points to ensure backtracking queries work correctly
        let query_points: Vec<[f32; 4]> = (0..50)
            .map(|_| {
                [
                    rng.random::<f32>(),
                    rng.random::<f32>(),
                    rng.random::<f32>(),
                    rng.random::<f32>(),
                ]
            })
            .collect();

        for query_point in query_points.iter() {
            let expected = linear_search(&points, query_point);
            let result = tree.nearest_one::<SquaredEuclidean<f32>>(query_point);

            assert_eq!(
                result.0, expected.distance,
                "Distance mismatch for query {:?}",
                query_point
            );
            assert_eq!(
                result.1 as usize, expected.item,
                "Item mismatch for query {:?}",
                query_point
            );
        }
    }

    #[test]
    #[cfg(feature = "simd")]
    #[cfg(target_arch = "x86_64")]
    fn v6_query_nearest_one_donnelly_marker_simd_block4_f32() {
        use crate::stem_strategies::{Block4, DonnellyMarkerSimd};

        // Test DonnellyMarkerSimd with f32 data using exact nearest_one query
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);

        // Use smaller dataset for faster test (16384 points = 512 leaves = 2^9, depth = 9)
        // Block4 doesn't divide evenly into 9, will be padded to 12
        let points: Vec<[f32; 4]> = (0..16_384)
            .map(|_| {
                [
                    rng.random::<f32>(),
                    rng.random::<f32>(),
                    rng.random::<f32>(),
                    rng.random::<f32>(),
                ]
            })
            .collect();

        let tree: KdTree<
            f32,
            u32,
            DonnellyMarkerSimd<Block4, 64, 4, 4>,
            FlatVec<f32, u32, 4, 32>,
            4,
            32,
        > = KdTree::new_from_slice(&points);

        assert!(!tree.is_empty());
        assert_eq!(tree.size(), 16_384);
        assert_eq!(tree.leaf_count(), 512);

        // Verify max_stem_level is padded to multiple of block size (4)
        assert_eq!((tree.max_stem_level() + 1) % 4, 0);

        // Test multiple query points to ensure backtracking queries work correctly
        let query_points: Vec<[f32; 4]> = (0..50)
            .map(|_| {
                [
                    rng.random::<f32>(),
                    rng.random::<f32>(),
                    rng.random::<f32>(),
                    rng.random::<f32>(),
                ]
            })
            .collect();

        for query_point in query_points.iter() {
            let expected = linear_search(&points, query_point);
            let result = tree.nearest_one::<SquaredEuclidean<f32>>(query_point);

            assert_eq!(
                result.0, expected.distance,
                "Distance mismatch for query {:?}",
                query_point
            );
            assert_eq!(
                result.1 as usize, expected.item,
                "Item mismatch for query {:?}",
                query_point
            );
        }
    }
}
