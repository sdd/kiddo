use crate::kd_tree::leaf_view::TlsLeafScratch;
use crate::kd_tree::query_orchestrator::with_tls_query_stack;
use crate::kd_tree::query_stack::{
    scalar_ctx_from_parts, scalar_ctx_into_parts, QueryStack, QueryStackContext, StackTrait,
};
use crate::kd_tree::traits::QueryContext;
use crate::kd_tree::KdTree;
use crate::stem_strategies::donnelly_2_blockmarker_simd::{BacktrackBlock3, BacktrackBlock4};
use crate::traits_unified_2::{AxisUnified, Basics, DistanceMetricUnified, LeafStrategy};
use crate::{Eytzinger, StemStrategy};
use std::any::TypeId;
use std::cmp::Ordering;
use std::ptr::NonNull;

impl<A, T, SS, LS, const K: usize, const B: usize> KdTree<A, T, SS, LS, K, B>
where
    A: AxisUnified<Coord = A> + 'static,
    T: Basics + Copy + Default + PartialOrd + PartialEq,
    LS: LeafStrategy<A, T, SS, K, B>,
    SS: StemStrategy,
{
    /// Finds the nearest point to the query point.
    ///
    /// Returns a tuple of (distance, item) for the nearest neighbor.
    #[inline(always)]
    pub fn nearest_one<D>(&self, query: &[A; K]) -> (D::Output, T)
    where
        D: DistanceMetricUnified<A, K>
            + crate::stem_strategies::DistanceMetricSimdBlock3<A, K, D::Output>
            + crate::stem_strategies::DistanceMetricSimdBlock4<A, K, D::Output>,
        D::Output: crate::stem_strategies::SimdPrune
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
        D: DistanceMetricUnified<A, K>
            + crate::stem_strategies::DistanceMetricSimdBlock3<A, K, D::Output>
            + crate::stem_strategies::DistanceMetricSimdBlock4<A, K, D::Output>,
        D::Output: crate::stem_strategies::SimdPrune
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

        self.backtracking_query::<_, _, D>(&mut req_ctx, |leaf, query_wide, query_ctx| {
            leaf.nearest_one_with_query_wide::<D>(
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
        D: DistanceMetricUnified<A, K>
            + crate::stem_strategies::DistanceMetricSimdBlock3<A, K, D::Output>
            + crate::stem_strategies::DistanceMetricSimdBlock4<A, K, D::Output>,
        D::Output: crate::stem_strategies::SimdPrune
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

        with_tls_query_stack::<SS::Stack<D::Output>, _>(|stack| {
            stack.clear();
            self.nearest_one_arithmetic_with_stack::<D>(query, stack)
        })
    }

    #[inline(always)]
    fn nearest_one_arithmetic_with_stack<D>(
        &self,
        query: &[A; K],
        stack: &mut SS::Stack<D::Output>,
    ) -> (D::Output, T)
    where
        D: DistanceMetricUnified<A, K>
            + crate::stem_strategies::DistanceMetricSimdBlock3<A, K, D::Output>
            + crate::stem_strategies::DistanceMetricSimdBlock4<A, K, D::Output>,
        D::Output: crate::stem_strategies::SimdPrune
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS>,
        SS: 'static,
    {
        if TypeId::of::<SS>() == TypeId::of::<Eytzinger<K>>() {
            let stack = unsafe {
                &mut *(stack as *mut SS::Stack<D::Output>
                    as *mut QueryStack<D::Output, Eytzinger<K>>)
            };
            return self.nearest_one_arithmetic_eytzinger_with_query_stack::<D>(query, stack);
        }

        let stack =
            unsafe { &mut *(stack as *mut SS::Stack<D::Output> as *mut QueryStack<D::Output, SS>) };
        self.nearest_one_arithmetic_scalar_with_query_stack::<D>(query, stack)
    }

    #[cfg_attr(not(feature = "cargo_asm"), allow(dead_code))]
    #[inline(always)]
    pub(crate) fn nearest_one_with_stack<D>(
        &self,
        query: &[A; K],
        stack: &mut SS::Stack<D::Output>,
    ) -> (D::Output, T)
    where
        D: DistanceMetricUnified<A, K>
            + crate::stem_strategies::DistanceMetricSimdBlock3<A, K, D::Output>
            + crate::stem_strategies::DistanceMetricSimdBlock4<A, K, D::Output>,
        D::Output: crate::stem_strategies::SimdPrune
            + BacktrackBlock3
            + BacktrackBlock4
            + TlsLeafScratch
            + 'static,
        SS::Stack<D::Output>: StackTrait<D::Output, SS>,
        SS: 'static,
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
        D: DistanceMetricUnified<A, K>
            + crate::stem_strategies::DistanceMetricSimdBlock3<A, K, D::Output>
            + crate::stem_strategies::DistanceMetricSimdBlock4<A, K, D::Output>,
        D::Output: crate::stem_strategies::SimdPrune
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
            |leaf, query_wide, query_ctx| {
                leaf.nearest_one_with_query_wide::<D>(
                    query_wide,
                    &mut query_ctx.best_dist,
                    &mut query_ctx.best_item,
                );
            },
        );

        (req_ctx.best_dist, req_ctx.best_item)
    }

    #[inline(always)]
    fn nearest_one_arithmetic_scalar_with_query_stack<D>(
        &self,
        query: &[A; K],
        stack: &mut QueryStack<D::Output, SS>,
    ) -> (D::Output, T)
    where
        D: DistanceMetricUnified<A, K>
            + crate::stem_strategies::DistanceMetricSimdBlock3<A, K, D::Output>
            + crate::stem_strategies::DistanceMetricSimdBlock4<A, K, D::Output>,
        D::Output: BacktrackBlock3 + BacktrackBlock4 + TlsLeafScratch,
    {
        if self.is_empty() {
            return (D::Output::max_value(), T::default());
        }

        let stems_ptr =
            NonNull::new(self.stems.as_ptr() as *mut u8).unwrap_or_else(NonNull::dangling);
        let mut query_wide: [D::Output; K] = [D::Output::zero(); K];
        for dim in 0..K {
            query_wide[dim] = D::widen_coord(query[dim]);
        }

        let mut best_dist = D::Output::max_value();
        let mut best_item = T::default();
        let mut off = [D::Output::zero(); K];

        let mut stem_strat = SS::new(stems_ptr);
        stack.clear();
        stack.push(scalar_ctx_from_parts::<D::Output, SS>(
            stem_strat.deferred_state(),
            D::Output::zero(),
            D::Output::zero(),
        ));

        while let Some(stack_ctx) = stack.pop() {
            let (stem_state, old_off, rd) = scalar_ctx_into_parts::<D::Output, SS>(stack_ctx);
            stem_strat.rehydrate_deferred_state(stem_state);
            let mut dim = stem_strat.dim();

            if D::Output::cmp(rd, best_dist) != Ordering::Less {
                continue;
            }

            unsafe { *off.get_unchecked_mut(dim) = old_off };

            loop {
                if stem_strat.level() > self.max_stem_level {
                    break;
                }

                let pivot = unsafe { *self.stems.get_unchecked(stem_strat.stem_idx()) };
                if pivot < A::max_value() {
                    let query_elem = unsafe { *query.get_unchecked(dim) };
                    let is_right_child = query_elem >= pivot;
                    let far_ctx = stem_strat.branch_relative(is_right_child);

                    let pivot_wide: D::Output = D::widen_coord(pivot);
                    let query_elem_wide = unsafe { *query_wide.get_unchecked(dim) };
                    let new_off = D::Output::saturating_dist(query_elem_wide, pivot_wide);
                    let old_off = unsafe { *off.get_unchecked(dim) };

                    let new_dist1 = D::dist1(new_off, D::Output::zero());
                    let old_dist1 = D::dist1(old_off, D::Output::zero());
                    let rd_far = D::Output::saturating_add(rd - old_dist1, new_dist1);

                    if D::Output::cmp(rd_far, best_dist) != Ordering::Greater {
                        stack.push(scalar_ctx_from_parts::<D::Output, SS>(
                            far_ctx.deferred_state(),
                            new_off,
                            rd_far,
                        ));
                    }
                } else {
                    stem_strat.traverse(false);
                }

                dim = stem_strat.dim();
            }

            let leaf_idx = stem_strat.leaf_idx();
            debug_assert!(
                leaf_idx < self.leaf_count(),
                "arithmetic nearest_one resolved invalid leaf_idx={} leaf_count={}",
                leaf_idx,
                self.leaf_count()
            );

            let leaf_view = self.leaves.leaf_view(leaf_idx);
            leaf_view.nearest_one_with_query_wide::<D>(&query_wide, &mut best_dist, &mut best_item);
        }

        (best_dist, best_item)
    }
    #[inline(always)]
    fn nearest_one_arithmetic_eytzinger_with_query_stack<D>(
        &self,
        query: &[A; K],
        stack: &mut QueryStack<D::Output, Eytzinger<K>>,
    ) -> (D::Output, T)
    where
        D: DistanceMetricUnified<A, K>
            + crate::stem_strategies::DistanceMetricSimdBlock3<A, K, D::Output>
            + crate::stem_strategies::DistanceMetricSimdBlock4<A, K, D::Output>,
        D::Output: BacktrackBlock3 + BacktrackBlock4 + TlsLeafScratch,
    {
        if self.is_empty() {
            return (D::Output::max_value(), T::default());
        }

        let stems_ptr =
            NonNull::new(self.stems.as_ptr() as *mut u8).unwrap_or_else(NonNull::dangling);
        let mut query_wide: [D::Output; K] = [D::Output::zero(); K];
        for dim in 0..K {
            query_wide[dim] = D::widen_coord(query[dim]);
        }

        let mut best_dist = D::Output::max_value();
        let mut best_item = T::default();
        let mut off = [D::Output::zero(); K];

        let mut stem_strat = Eytzinger::<K>::new(stems_ptr);
        stack.clear();
        stack.push(QueryStackContext::new(stem_strat.deferred_state()));

        while let Some(stack_ctx) = stack.pop() {
            let (stem_state, old_off, rd) = stack_ctx.into_parts();
            stem_strat.rehydrate_deferred_state(stem_state);
            let mut dim = stem_strat.dim();

            if D::Output::cmp(rd, best_dist) != Ordering::Less {
                continue;
            }

            unsafe { *off.get_unchecked_mut(dim) = old_off };

            loop {
                if stem_strat.level() > self.max_stem_level {
                    break;
                }

                let pivot = unsafe { *self.stems.get_unchecked(stem_strat.stem_idx()) };
                if pivot < A::max_value() {
                    let query_elem = unsafe { *query.get_unchecked(dim) };
                    let is_right_child = query_elem >= pivot;
                    let far_ctx = stem_strat.branch_relative(is_right_child);

                    let pivot_wide: D::Output = D::widen_coord(pivot);
                    let query_elem_wide = unsafe { *query_wide.get_unchecked(dim) };
                    let new_off = D::Output::saturating_dist(query_elem_wide, pivot_wide);
                    let old_off = unsafe { *off.get_unchecked(dim) };

                    let new_dist1 = D::dist1(new_off, D::Output::zero());
                    let old_dist1 = D::dist1(old_off, D::Output::zero());
                    let rd_far = D::Output::saturating_add(rd - old_dist1, new_dist1);

                    if D::Output::cmp(rd_far, best_dist) != Ordering::Greater {
                        stack.push(QueryStackContext {
                            stem_state: far_ctx.deferred_state(),
                            old_off: new_off,
                            rd: rd_far,
                        });
                    }
                } else {
                    stem_strat.traverse(false);
                }

                dim = stem_strat.dim();
            }

            let leaf_idx = stem_strat.leaf_idx();
            debug_assert!(
                leaf_idx < self.leaf_count(),
                "arithmetic Eytzinger nearest_one resolved invalid leaf_idx={} leaf_count={}",
                leaf_idx,
                self.leaf_count()
            );

            let leaf_view = self.leaves.leaf_view(leaf_idx);
            leaf_view.nearest_one_with_query_wide::<D>(&query_wide, &mut best_dist, &mut best_item);
        }

        (best_dist, best_item)
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

        for query_point in query_points.iter() {
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

        for query_point in query_points.iter() {
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

        for query_point in query_points.iter() {
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

        for query_point in query_points.iter() {
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

        for query_point in query_points.iter() {
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
