use crate::dist::KdTreeDistanceMetric;
use crate::kd_tree::leaf_view::TlsLeafScratch;
use crate::kd_tree::leaf_view_chunked::nearest_one::nearest_one_with_query_wide;
use crate::kd_tree::query_stack::StackTrait;
use crate::kd_tree::traits::QueryContext;
use crate::kd_tree::KdTree;
use crate::stem_strategies::donnelly_2_blockmarker_simd::{
    BacktrackBlock3, BacktrackBlock4, SimdSelectBestChildBlock3,
};
use crate::traits_unified_2::{AxisUnified, Basics, LeafProjection, LeafStrategy};
use crate::StemStrategy;

impl<A, T, SS, LS, const K: usize, const B: usize> KdTree<A, T, SS, LS, K, B>
where
    A: AxisUnified<Coord = A> + 'static,
    T: Basics + Copy + Default + PartialOrd + PartialEq,
    LS: LeafStrategy<A, T, SS, K, B>,
    SS: StemStrategy,
{
    #[inline(always)]
    fn process_leaf_nearest_one<D>(
        &self,
        leaf_idx: usize,
        query_wide: &[D::Output; K],
        best_dist: &mut D::Output,
        best_item: &mut T,
    ) where
        D: KdTreeDistanceMetric<A, K>,
        D::Output: AxisUnified<Coord = D::Output> + 'static,
    {
        match LS::LEAF_PROJECTION {
            LeafProjection::LeafArena => {
                let arena = self.leaves.leaf_arena(leaf_idx);
                crate::kd_tree::leaf_view_chunked::nearest_one::nearest_one_with_query_wide_arena::<
                    A,
                    T,
                    D,
                    K,
                >(&arena, query_wide, best_dist, best_item);
            }
            LeafProjection::LeafView => {
                let leaf = self.leaves.leaf_view(leaf_idx);
                nearest_one_with_query_wide::<A, T, D, K, B>(
                    &leaf, query_wide, best_dist, best_item,
                );
            }
        }
    }

    /// Finds the nearest point to the query point.
    ///
    /// Returns a tuple of (distance, item) for the nearest neighbor.
    #[inline(always)]
    pub fn nearest_one<D>(&self, query: &[A; K]) -> (D::Output, T)
    where
        D: KdTreeDistanceMetric<A, K>,
        D::Output: crate::stem_strategies::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS> + Default + 'static,
        SS: 'static,
    {
        if self.stem_leaf_resolution.uses_arithmetic() {
            return self.nearest_one_arithmetic::<D>(query);
        }

        self.nearest_one_mapped::<D>(query)
    }

    #[inline(always)]
    fn nearest_one_mapped<D>(&self, query: &[A; K]) -> (D::Output, T)
    where
        D: KdTreeDistanceMetric<A, K>,
        D::Output: crate::stem_strategies::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS> + Default + 'static,
    {
        let mut req_ctx = NearestOneReqCtx {
            query,
            best_dist: D::Output::max_value(),
            best_item: T::default(),
        };

        self.backtracking_query::<_, _, D>(&mut req_ctx, |leaf_idx, query_wide, query_ctx| {
            self.process_leaf_nearest_one::<D>(
                leaf_idx,
                query_wide,
                &mut query_ctx.best_dist,
                &mut query_ctx.best_item,
            );
        });

        (req_ctx.best_dist, req_ctx.best_item)
    }

    #[inline(always)]
    fn nearest_one_arithmetic<D>(&self, query: &[A; K]) -> (D::Output, T)
    where
        D: KdTreeDistanceMetric<A, K>,
        D::Output: crate::stem_strategies::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS> + Default + 'static,
        SS: 'static,
    {
        if SS::BLOCK_SIZE != 1 {
            return self.nearest_one_mapped::<D>(query);
        }

        let mut req_ctx = NearestOneReqCtx {
            query,
            best_dist: D::Output::max_value(),
            best_item: T::default(),
        };

        self.arithmetic_query::<_, _, D>(&mut req_ctx, |leaf_idx, query_wide, query_ctx| {
            self.process_leaf_nearest_one::<D>(
                leaf_idx,
                query_wide,
                &mut query_ctx.best_dist,
                &mut query_ctx.best_item,
            );
        });

        (req_ctx.best_dist, req_ctx.best_item)
    }

    #[inline(always)]
    fn nearest_one_arithmetic_with_stack<D>(
        &self,
        query: &[A; K],
        stack: &mut SS::Stack<D::Output>,
    ) -> (D::Output, T)
    where
        D: KdTreeDistanceMetric<A, K>,
        D::Output: crate::stem_strategies::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS>,
    {
        let mut req_ctx = NearestOneReqCtx {
            query,
            best_dist: D::Output::max_value(),
            best_item: T::default(),
        };

        self.arithmetic_query_with_stack::<_, _, D>(
            &mut req_ctx,
            stack,
            |leaf_idx, query_wide, query_ctx| {
                self.process_leaf_nearest_one::<D>(
                    leaf_idx,
                    query_wide,
                    &mut query_ctx.best_dist,
                    &mut query_ctx.best_item,
                );
            },
        );

        (req_ctx.best_dist, req_ctx.best_item)
    }

    #[cfg_attr(not(feature = "cargo_asm"), allow(dead_code))]
    #[inline(always)]
    pub(crate) fn nearest_one_with_stack<D>(
        &self,
        query: &[A; K],
        stack: &mut SS::Stack<D::Output>,
    ) -> (D::Output, T)
    where
        D: KdTreeDistanceMetric<A, K>,
        D::Output: crate::stem_strategies::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS>,
    {
        if self.stem_leaf_resolution.uses_arithmetic() {
            return self.nearest_one_arithmetic_with_stack::<D>(query, stack);
        }

        self.nearest_one_mapped_with_stack::<D>(query, stack)
    }

    #[inline(always)]
    fn nearest_one_mapped_with_stack<D>(
        &self,
        query: &[A; K],
        stack: &mut SS::Stack<D::Output>,
    ) -> (D::Output, T)
    where
        D: KdTreeDistanceMetric<A, K>,
        D::Output: crate::stem_strategies::SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS>,
    {
        let mut req_ctx = NearestOneReqCtx {
            query,
            best_dist: D::Output::max_value(),
            best_item: T::default(),
        };

        self.backtracking_query_with_stack::<_, _, D>(
            &mut req_ctx,
            stack,
            |leaf_idx, query_wide, query_ctx| {
                self.process_leaf_nearest_one::<D>(
                    leaf_idx,
                    query_wide,
                    &mut query_ctx.best_dist,
                    &mut query_ctx.best_item,
                );
            },
        );

        (req_ctx.best_dist, req_ctx.best_item)
    }
}

#[allow(missing_docs)]
#[cfg(feature = "cargo_asm")]
pub mod cargo_asm {
    use crate::dist::SquaredEuclidean;
    use crate::kd_tree::leaf_strategies::FlatVec;
    use crate::kd_tree::query_stack::QueryStack;
    use crate::kd_tree::KdTree;
    use crate::Eytzinger;

    const K: usize = 3;
    const BUCKET_SIZE: usize = 64;

    type KdT =
        KdTree<f64, usize, Eytzinger<K>, FlatVec<f64, usize, K, BUCKET_SIZE>, K, BUCKET_SIZE>;

    /// Hook for cargo-asm to render the v6 nearest-one call path.
    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn v6_nearest_one_eytzinger_cargo_asm_hook(
        tree: &KdT,
        query: [f64; 3],
        stack: &mut QueryStack<f64, Eytzinger<3>>,
    ) -> (f64, usize) {
        tree.nearest_one_with_stack::<SquaredEuclidean<f64>>(&query, stack)
    }

    /// Hook for cargo-asm to render the arithmetic Eytzinger core directly.
    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn v6_nearest_one_eytzinger_arithmetic_core_cargo_asm_hook(
        tree: &KdT,
        query: [f64; 3],
        stack: &mut QueryStack<f64, Eytzinger<3>>,
    ) -> (f64, usize) {
        tree.nearest_one_arithmetic_with_stack::<SquaredEuclidean<f64>>(&query, stack)
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

    // TOOO: investigate into whether this can be removed
    #[inline]
    fn prune_on_equal_max_dist(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use assert_float_eq::assert_float_relative_eq;
    use rand::Rng;
    use rand::SeedableRng;
    use test_log::test;

    use crate::dist::SquaredEuclidean;
    use crate::kd_tree::leaf_strategies::{FlatVec, VecOfArenas, VecOfArrays};
    use crate::kd_tree::KdTree;
    use crate::stem_strategies::Donnelly;
    use crate::traits::Axis;
    use crate::{Eytzinger, NearestNeighbour};

    const REL_EPS_F32: f32 = 1.0e-6;
    const REL_EPS_F64: f64 = 1.0e-12;

    fn assert_nearest_f32(actual: (f32, u32), expected: &NearestNeighbour<f32, usize>) {
        assert_float_relative_eq!(actual.0, expected.distance, REL_EPS_F32);
        assert_eq!(actual.1 as usize, expected.item);
    }

    fn assert_nearest_f64(actual: (f64, u32), expected: &NearestNeighbour<f64, usize>) {
        assert_float_relative_eq!(actual.0, expected.distance, REL_EPS_F64);
        assert_eq!(actual.1 as usize, expected.item);
    }

    #[test]
    fn nearest_one_vec_of_arenas_small_f64() {
        let points = vec![
            [0.0f64, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [0.5, 0.5, 0.6],
        ];

        let tree: KdTree<f64, u32, Eytzinger<3>, VecOfArenas<f64, u32, 3, 32>, 3, 32> =
            KdTree::new_from_slice(&points);

        let result = tree.nearest_one::<SquaredEuclidean<f64>>(&[0.45, 0.55, 0.65]);

        assert_float_relative_eq!(result.0, 0.0075, REL_EPS_F64);
        assert_eq!(result.1, 3);
    }

    #[test]
    fn nearest_one_vec_of_arenas_matches_flat_vec_f32() {
        let points = vec![
            [0.1f32, 0.2, 0.3],
            [0.9, 0.8, 0.7],
            [0.41, 0.52, 0.63],
            [0.4, 0.5, 0.6],
            [0.7, 0.1, 0.2],
        ];
        let query = [0.39f32, 0.51, 0.61];

        let flat_tree: KdTree<f32, u32, Eytzinger<3>, FlatVec<f32, u32, 3, 32>, 3, 32> =
            KdTree::new_from_slice(&points);
        let arena_tree: KdTree<f32, u32, Eytzinger<3>, VecOfArenas<f32, u32, 3, 32>, 3, 32> =
            KdTree::new_from_slice(&points);

        let flat_result = flat_tree.nearest_one::<SquaredEuclidean<f32>>(&query);
        let arena_result = arena_tree.nearest_one::<SquaredEuclidean<f32>>(&query);

        assert_float_relative_eq!(arena_result.0, flat_result.0, REL_EPS_F32);
        assert_eq!(arena_result.1, flat_result.1);
    }

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
        assert_float_relative_eq!(results.0, expected.0, REL_EPS_F64);
        assert_eq!(results.1, expected.1);
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

        for query_point in query_points.iter() {
            let expected = linear_search(&content_to_add, query_point);
            let result = tree.nearest_one::<SquaredEuclidean<f32>>(query_point);

            assert_nearest_f32(result, &expected);
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

        for query_point in query_points.iter() {
            let expected = linear_search(&content_to_add, query_point);
            let result = tree.nearest_one::<SquaredEuclidean<f32>>(query_point);

            assert_nearest_f32(result, &expected);
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

        for query_point in query_points.iter() {
            let expected = linear_search(&content_to_add, query_point);
            let result = tree.nearest_one::<SquaredEuclidean<f32>>(query_point);

            assert_nearest_f32(result, &expected);
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

            assert_float_relative_eq!(result.0, expected.distance, REL_EPS_F32);
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

        for query_point in query_points.iter() {
            let expected = linear_search(&content_to_add, query_point);
            let result = tree.nearest_one::<SquaredEuclidean<f32>>(query_point);

            assert_nearest_f32(result, &expected);
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

        for query_point in query_points.iter() {
            let expected = linear_search(&content_to_add, query_point);
            let result = tree.nearest_one::<SquaredEuclidean<f32>>(query_point);

            assert_nearest_f32(result, &expected);
        }
    }

    fn linear_search<A: Axis, const K: usize>(
        content: &[[A; K]],
        query_point: &[A; K],
    ) -> NearestNeighbour<A, usize>
    where
        crate::dist::SquaredEuclidean<A>: crate::dist::DistanceMetricCore<A, Output = A>,
    {
        let mut best_dist: A = A::infinity();
        let mut best_item: usize = usize::MAX;

        for (idx, p) in content.iter().enumerate() {
            let dist = squared_euclidean_dist(query_point, p);
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

        for query_point in query_points.iter() {
            // tracing::debug!("Query point: #{i} ({query_point:?})");

            let expected = linear_search(&points, query_point);
            // println!("\n========== QUERY #{i} ==========");
            // println!("Query point: {:?}", query_point);
            // println!("Expected: item={}, dist²={}", expected.item, expected.distance);

            let _result = tree_non_simd.nearest_one::<SquaredEuclidean<f64>>(query_point);
            // println!("NON-SIMD: item={}, dist²={}", result.1, result.0);

            let result = tree.nearest_one::<SquaredEuclidean<f64>>(query_point);
            // println!("SIMD: item={}, dist²={}", result.1, result.0);

            assert_nearest_f64(result, &expected);
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

            assert_float_relative_eq!(result.0, expected.distance, REL_EPS_F32);
            assert_eq!(
                result.1 as usize, expected.item,
                "Item mismatch for query {:?}",
                query_point
            );
        }
    }
}
