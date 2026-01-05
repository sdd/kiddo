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
    /// Finds an approximate nearest point to the query point.
    ///
    /// This is faster than `nearest_one` but may not return the true nearest neighbour.
    /// It searches only the leaf that the query point falls into.
    pub fn approx_nearest_one<D>(&self, query: &[A; K]) -> (D::Output, T)
    where
        D: DistanceMetricUnified<A, K>,
    {
        let req_ctx = ApproxNearestOneReqCtx::<A, D::Output, K> {
            query,
            _phantom: std::marker::PhantomData,
        };

        let mut best_dist = D::Output::max_value();
        let mut best_item = T::default();

        self.straight_query(req_ctx, |leaf| {
            leaf.nearest_one::<D>(query, &mut best_dist, &mut best_item);
        });

        (best_dist, best_item)
    }
}

struct ApproxNearestOneReqCtx<'a, A, O, const K: usize> {
    query: &'a [A; K],
    _phantom: std::marker::PhantomData<O>,
}

impl<A, O, const K: usize> QueryContext<A, O, K> for ApproxNearestOneReqCtx<'_, A, O, K> {
    fn query(&self) -> &[A; K] {
        self.query
    }

    fn max_dist(&self) -> O {
        panic!("approx_nearest_one should not be called with max_dist")
    }
}

#[cfg(test)]
mod tests {
    use rand::rngs::StdRng;
    use rand::Rng;
    use rand::SeedableRng;

    use crate::kd_tree::leaf_strategies::{FlatVec, VecOfArrays};
    use crate::kd_tree::KdTree;
    use crate::stem_strategies::{Block3, Block4, Donnelly, DonnellyMarkerPf};

    #[cfg(feature = "simd")]
    use crate::stem_strategies::DonnellyMarkerSimd;
    use crate::traits_unified_2::SquaredEuclidean;
    use crate::Eytzinger;

    const RNG_SEED: u64 = 42;

    #[test]
    fn v6_approx_nearest_one_flat_vec_f32() {
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

        let results = tree.approx_nearest_one::<SquaredEuclidean<f32>>(&query_point);

        assert_eq!(results, (0.0014114721, 19074));
    }

    #[test]
    fn v6_approx_nearest_one_vec_of_arrays_f32() {
        let mut rng = StdRng::seed_from_u64(RNG_SEED);

        let mut points: Vec<[f32; 3]> = vec![];
        for _ in 0..65_536 {
            let x = rng.random_range(0.0..1.0);
            let y = rng.random_range(0.0..1.0);
            let z = rng.random_range(0.0..1.0);
            points.push([x, y, z]);
        }

        let tree: KdTree<f32, u32, Eytzinger<3>, VecOfArrays<f32, u32, 3, 32>, 3, 32> =
            KdTree::new_from_slice(&points);

        assert!(!tree.is_empty());
        assert_eq!(tree.size(), 65_536);
        assert_eq!(tree.leaf_count(), 2048);
        assert_eq!(tree.max_stem_level(), 10);

        let query_point = [0.5, 0.5, 0.5];

        let results = tree.approx_nearest_one::<SquaredEuclidean<f32>>(&query_point);

        assert_eq!(results, (0.0014114721, 19074));
    }

    #[test]
    fn v6_approx_nearest_one_vec_of_arrays_mutated_one_split_f32() {
        let mut rng = StdRng::seed_from_u64(RNG_SEED);

        // Add 33 points to trigger exactly one split (bucket size is 32)
        let mut points: Vec<[f32; 3]> = vec![];
        for _ in 0..33 {
            let x = rng.random_range(0.0..1.0);
            let y = rng.random_range(0.0..1.0);
            let z = rng.random_range(0.0..1.0);
            points.push([x, y, z]);
        }

        let mut tree: KdTree<f32, u32, Eytzinger<3>, VecOfArrays<f32, u32, 3, 32>, 3, 32> =
            KdTree::default();

        for (idx, point) in points.iter().enumerate() {
            tree.add(point, idx as u32)
        }

        // Print tree state for debugging
        println!("\nTree state after adding 33 points:");
        println!("{}", tree);

        assert!(!tree.is_empty());
        assert_eq!(tree.size(), 33);
        assert_eq!(tree.leaf_count(), 2); // Should have split into 2 leaves

        // Check that the root stem value was updated from max_value to a reasonable pivot
        let root_stem_value = tree.stems[1]; // Eytzinger root is at index 1
        assert!(
            root_stem_value < f32::MAX,
            "Root stem should be updated from max_value"
        );
        assert!(
            root_stem_value >= 0.0 && root_stem_value <= 1.0,
            "Root stem should be a reasonable value in [0, 1]"
        );

        // Just verify the tree queries successfully - we'll verify correctness later
        let query_point = [0.5, 0.5, 0.5];
        let _results = tree.approx_nearest_one::<SquaredEuclidean<f32>>(&query_point);
    }

    #[test]
    fn v6_approx_nearest_one_vec_of_arrays_mutated_two_splits_f32() {
        let mut rng = StdRng::seed_from_u64(RNG_SEED);

        // Add 65 points to trigger exactly two splits (bucket size is 32)
        // First split at 33 points: 1 leaf -> 2 leaves
        // Second split at 65 points: 2 leaves -> 3 leaves
        let mut points: Vec<[f32; 3]> = vec![];
        for _ in 0..65 {
            let x = rng.random_range(0.0..1.0);
            let y = rng.random_range(0.0..1.0);
            let z = rng.random_range(0.0..1.0);
            points.push([x, y, z]);
        }

        let mut tree: KdTree<f32, u32, Eytzinger<3>, VecOfArrays<f32, u32, 3, 32>, 3, 32> =
            KdTree::default();

        for (idx, point) in points.iter().enumerate() {
            // println!("Tree at item {}: {}", idx, tree);
            tree.add(point, idx as u32)
        }

        // Print tree state for debugging
        // println!("\nTree state after adding 65 points:");
        println!("{}", tree);

        assert!(!tree.is_empty());
        assert_eq!(tree.size(), 65);
        assert_eq!(tree.leaf_count(), 3); // Should have split into 3 leaves

        // Check that the root stem value was updated from max_value to a reasonable pivot
        let root_stem_value = tree.stems[1]; // Eytzinger root is at index 1
        assert!(
            root_stem_value < f32::MAX,
            "Root stem should be updated from max_value"
        );
        assert!(
            root_stem_value >= 0.0 && root_stem_value <= 1.0,
            "Root stem should be a reasonable value in [0, 1]"
        );

        // Check that at least one child stem was also initialized
        let left_child_stem = tree.stems[2]; // Eytzinger left child at index 2
        let right_child_stem = tree.stems[3]; // Eytzinger right child at index 3

        // At least one child should have been split (not max_value any more)
        let has_initialized_child = (left_child_stem < f32::MAX) || (right_child_stem < f32::MAX);
        assert!(
            has_initialized_child,
            "At least one child stem should be initialized after second split"
        );

        // Just verify the tree queries successfully - we'll verify correctness later
        let query_point = [0.5, 0.5, 0.5];
        let _results = tree.approx_nearest_one::<SquaredEuclidean<f32>>(&query_point);
    }

    #[test]
    fn v6_approx_nearest_one_vec_of_arrays_mutated_f32() {
        let mut rng = StdRng::seed_from_u64(RNG_SEED);

        let mut points: Vec<[f32; 3]> = vec![];
        for _ in 0..65_536 {
            let x = rng.random_range(0.0..1.0);
            let y = rng.random_range(0.0..1.0);
            let z = rng.random_range(0.0..1.0);
            points.push([x, y, z]);
        }

        let mut tree: KdTree<f32, u32, Eytzinger<3>, VecOfArrays<f32, u32, 3, 32>, 3, 32> =
            KdTree::default();

        for (idx, point) in points.iter().enumerate() {
            // println!("Tree at item {}: {}", idx, tree);
            tree.add(point, idx as u32);
        }

        assert!(!tree.is_empty());
        assert_eq!(tree.size(), 65_536);
        // assert_eq!(tree.leaf_count(), 2048);
        // assert_eq!(tree.max_stem_level(), 10);

        let query_point = [0.5, 0.5, 0.5];

        let results = tree.approx_nearest_one::<SquaredEuclidean<f32>>(&query_point);

        assert_eq!(results, (0.0003201659, 21996));
    }

    #[test]
    fn v6_approx_nearest_one_flat_vec_f32_donnelly_marker() {
        let mut rng = StdRng::seed_from_u64(RNG_SEED);

        let mut points: Vec<[f32; 4]> = vec![];
        for _ in 0..131_072 {
            let x = rng.random_range(0.0..1.0);
            let y = rng.random_range(0.0..1.0);
            let z = rng.random_range(0.0..1.0);
            let w = rng.random_range(0.0..1.0);
            points.push([x, y, z, w]);
        }

        // Use DonnellyMarkerPf with Block4 (4 levels per block, matching 16 f32s per 64-byte cache line)
        let tree: KdTree<
            f32,
            u32,
            DonnellyMarkerPf<Block4, 64, 4, 4>,
            FlatVec<f32, u32, 4, 32>,
            4,
            32,
        > = KdTree::new_from_slice(&points);

        assert!(!tree.is_empty());
        assert_eq!(tree.size(), 131_072);

        let query_point = [0.5, 0.5, 0.5, 0.5];

        let results = tree.approx_nearest_one::<SquaredEuclidean<f32>>(&query_point);

        assert!(results.0 < 0.1, "Distance should be reasonably small");
        assert!(results.1 < 131_072, "Item index should be valid");
    }

    #[test]
    fn v6_approx_nearest_one_donnelly_marker_matches_eytzinger() {
        // Verify that DonnellyMarkerPf produces the same results as Eytzinger
        // for the same input data
        let mut rng = StdRng::seed_from_u64(RNG_SEED);

        let points: Vec<[f32; 4]> = (0..8_192)
            .map(|_| {
                [
                    rng.random_range(0.0..1.0),
                    rng.random_range(0.0..1.0),
                    rng.random_range(0.0..1.0),
                    rng.random_range(0.0..1.0),
                ]
            })
            .collect();

        let tree_eytz: KdTree<f32, u32, Eytzinger<4>, FlatVec<f32, u32, 4, 32>, 4, 32> =
            KdTree::new_from_slice(&points);

        let tree_donnelly: KdTree<
            f32,
            u32,
            DonnellyMarkerPf<Block4, 64, 4, 4>,
            FlatVec<f32, u32, 4, 32>,
            4,
            32,
        > = KdTree::new_from_slice(&points);

        // Test multiple query points
        let query_points: Vec<[f32; 4]> = (0..100)
            .map(|_| {
                [
                    rng.random_range(0.0..1.0),
                    rng.random_range(0.0..1.0),
                    rng.random_range(0.0..1.0),
                    rng.random_range(0.0..1.0),
                ]
            })
            .collect();

        for (idx, query_point) in query_points.iter().enumerate() {
            let result_eytz = tree_eytz.approx_nearest_one::<SquaredEuclidean<f32>>(query_point);
            let result_donnelly =
                tree_donnelly.approx_nearest_one::<SquaredEuclidean<f32>>(query_point);

            // The results should be identical since both use the same construction
            // and the approximate query just returns the best match in the target leaf
            assert_eq!(
                result_eytz, result_donnelly,
                "DonnellyMarkerPf should produce same results as Eytzinger for query #{} ({:?})",
                idx, query_point
            );
        }
    }

    #[test]
    fn v6_approx_nearest_one_donnelly_matches_eytzinger() {
        let mut rng = StdRng::seed_from_u64(RNG_SEED);

        let points: Vec<[f32; 4]> = (0..8_192)
            .map(|_| {
                [
                    rng.random_range(0.0..1.0),
                    rng.random_range(0.0..1.0),
                    rng.random_range(0.0..1.0),
                    rng.random_range(0.0..1.0),
                ]
            })
            .collect();

        let tree_eytz: KdTree<f32, u32, Eytzinger<4>, FlatVec<f32, u32, 4, 32>, 4, 32> =
            KdTree::new_from_slice(&points);

        let tree_donnelly: KdTree<
            f32,
            u32,
            Donnelly<4, 64, 4, 4>,
            FlatVec<f32, u32, 4, 32>,
            4,
            32,
        > = KdTree::new_from_slice(&points);

        // Test multiple query points
        let query_points: Vec<[f32; 4]> = (0..100)
            .map(|_| {
                [
                    rng.random_range(0.0..1.0),
                    rng.random_range(0.0..1.0),
                    rng.random_range(0.0..1.0),
                    rng.random_range(0.0..1.0),
                ]
            })
            .collect();

        for (idx, query_point) in query_points.iter().enumerate() {
            let result_eytz = tree_eytz.approx_nearest_one::<SquaredEuclidean<f32>>(query_point);
            let result_donnelly =
                tree_donnelly.approx_nearest_one::<SquaredEuclidean<f32>>(query_point);

            assert_eq!(
                result_eytz, result_donnelly,
                "Donnelly should produce same results as Eytzinger for query #{} ({:?})",
                idx, query_point
            );
        }
    }

    #[test]
    fn v6_approx_nearest_one_donnelly_marker_matches_donnelly() {
        // Verify that DonnellyMarkerPf produces the same results as Donnelly
        // for the same input data
        let mut rng = StdRng::seed_from_u64(RNG_SEED);

        let points: Vec<[f32; 4]> = (0..8_192)
            .map(|_| {
                [
                    rng.random_range(0.0..1.0),
                    rng.random_range(0.0..1.0),
                    rng.random_range(0.0..1.0),
                    rng.random_range(0.0..1.0),
                ]
            })
            .collect();

        let tree_donnelly: KdTree<
            f32,
            u32,
            Donnelly<4, 64, 4, 4>,
            FlatVec<f32, u32, 4, 32>,
            4,
            32,
        > = KdTree::new_from_slice(&points);

        let tree_donnelly_marker: KdTree<
            f32,
            u32,
            DonnellyMarkerPf<Block4, 64, 4, 4>,
            FlatVec<f32, u32, 4, 32>,
            4,
            32,
        > = KdTree::new_from_slice(&points);

        // Test multiple query points
        let query_points: Vec<[f32; 4]> = (0..100)
            .map(|_| {
                [
                    rng.random_range(0.0..1.0),
                    rng.random_range(0.0..1.0),
                    rng.random_range(0.0..1.0),
                    rng.random_range(0.0..1.0),
                ]
            })
            .collect();

        for (idx, query_point) in query_points.iter().enumerate() {
            let result_eytz =
                tree_donnelly.approx_nearest_one::<SquaredEuclidean<f32>>(query_point);
            let result_donnelly =
                tree_donnelly_marker.approx_nearest_one::<SquaredEuclidean<f32>>(query_point);

            // The results should be identical since both use the same construction
            // and the approximate query just returns the best match in the target leaf
            assert_eq!(
                result_eytz, result_donnelly,
                "DonnellyMarkerPf should produce same results as Donnelly for query #{} ({:?})",
                idx, query_point
            );
        }
    }

    #[test]
    #[cfg(feature = "simd")]
    #[cfg(target_arch = "x86_64")]
    fn v6_approx_nearest_one_donnelly_marker_simd_f64() {
        // Test DonnellyMarkerSimd with f64 data
        let mut rng = StdRng::seed_from_u64(RNG_SEED);

        // Use 8192 points which with bucket size 32 gives 256 leaves
        // 256 leaves = 2^8, so tree depth = 8
        // With Block3, depth 8 is not divisible by 3, so tree will be padded to depth 9
        let points: Vec<[f64; 3]> = (0..8_192)
            .map(|_| {
                [
                    rng.random_range(0.0..1.0),
                    rng.random_range(0.0..1.0),
                    rng.random_range(0.0..1.0),
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

        assert!(!tree.is_empty());
        assert_eq!(tree.size(), 8_192);
        assert_eq!(tree.leaf_count(), 256);

        // Verify max_stem_level is padded to multiple of block size (3)
        // 256 leaves = depth 8, padded to 9
        assert_eq!((tree.max_stem_level() + 1) % 3, 0);
        assert_eq!(tree.max_stem_level(), 8);

        // Test multiple query points to ensure queries work correctly
        let query_points: Vec<[f64; 3]> = (0..100)
            .map(|_| {
                [
                    rng.random_range(0.0..1.0),
                    rng.random_range(0.0..1.0),
                    rng.random_range(0.0..1.0),
                ]
            })
            .collect();

        for query_point in query_points.iter() {
            let result = tree.approx_nearest_one::<SquaredEuclidean<f64>>(query_point);

            // Verify result is valid
            assert!(result.0 >= 0.0, "Distance should be non-negative");
            assert!(result.1 < 8_192, "Item index should be valid");

            // Verify the returned item is actually close to the query
            // (approximate query returns best in target leaf, so should be reasonably close)
            assert!(
                result.0 < 1.0,
                "Distance should be reasonable for unit cube"
            );
        }
    }

    #[test]
    #[cfg(feature = "simd")]
    #[cfg(target_arch = "x86_64")]
    fn v6_approx_nearest_one_donnelly_marker_simd_f32() {
        // Test DonnellyMarkerSimd with f32 data
        let mut rng = StdRng::seed_from_u64(RNG_SEED);

        let points: Vec<[f32; 4]> = (0..2_097_152)
            .map(|_| {
                [
                    rng.random_range(0.0..1.0),
                    rng.random_range(0.0..1.0),
                    rng.random_range(0.0..1.0),
                    rng.random_range(0.0..1.0),
                ]
            })
            .collect();

        // 2_097_152 points with bucket size 32 = 65_536 leaves = 2^16, depth = 16
        // Block4 divides evenly into 16
        let tree: KdTree<
            f32,
            u32,
            DonnellyMarkerSimd<Block4, 64, 4, 4>,
            FlatVec<f32, u32, 4, 32>,
            4,
            32,
        > = KdTree::new_from_slice(&points);

        assert!(!tree.is_empty());
        assert_eq!(tree.size(), 2_097_152);
        assert_eq!(tree.leaf_count(), 65_536);
        // Verify max_stem_level is padded to multiple of block size
        assert_eq!((tree.max_stem_level() + 1) % 4, 0);

        let query_point = [0.5, 0.5, 0.5, 0.5];
        let results = tree.approx_nearest_one::<SquaredEuclidean<f32>>(&query_point);

        assert!(results.0 < 0.2, "Distance should be reasonably small");
        assert!(results.1 < 2_097_152, "Item index should be valid");
    }
}
