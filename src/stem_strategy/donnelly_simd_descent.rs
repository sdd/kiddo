//! Hybrid Donnelly stem strategy.
//!
//! Uses block-at-once block-marker SIMD descent, but exact-search pruning and
//! backtracking are handled with scalar sibling materialization on the normal
//! scalar query stack.

use std::marker::PhantomData;
use std::ptr::NonNull;

use aligned_vec::AVec;

use crate::kd_tree::query_stack::{QueryStack, ScalarStackContext};
use crate::stem_strategy::donnelly_2_blockmarker_simd::{
    child_interval_bounds_block3, compare_block3, interval_distance_1d,
};
use crate::stem_strategy::donnelly_core::DonnellyCore;
use crate::stem_strategy::Block3;
use crate::{Axis, StemStrategy};

/// Scalar stack context for block-at-once scalar backtracking.
///
/// `restore_dim` tracks which dimension's interval offset must be restored when
/// resuming a deferred subtree. For block traversal this can differ from the
/// resumed stem strategy's current dimension because the block advances the
/// split dimension once for the entire block.
#[derive(Clone, Copy, Debug)]
pub struct DonnellySimdDescentStackContext<A, S> {
    stem_state: S,
    restore_dim: usize,
    old_off: A,
    rd: A,
}

impl<A, S> DonnellySimdDescentStackContext<A, S> {
    #[inline(always)]
    pub(crate) fn new(stem_state: S, restore_dim: usize, old_off: A, rd: A) -> Self {
        Self {
            stem_state,
            restore_dim,
            old_off,
            rd,
        }
    }
}

impl<A, S> ScalarStackContext<A, S> for DonnellySimdDescentStackContext<A, S> {
    #[inline(always)]
    fn from_parts(stem_state: S, old_off: A, rd: A) -> Self {
        Self {
            stem_state,
            restore_dim: usize::MAX,
            old_off,
            rd,
        }
    }

    #[inline(always)]
    fn into_parts(self) -> (S, A, A) {
        (self.stem_state, self.old_off, self.rd)
    }

    #[inline(always)]
    fn from_parts_with_restore_dim(stem_state: S, restore_dim: usize, old_off: A, rd: A) -> Self {
        Self::new(stem_state, restore_dim, old_off, rd)
    }

    #[inline(always)]
    fn into_parts_with_restore_dim(self) -> (S, Option<usize>, A, A) {
        let restore_dim = (self.restore_dim != usize::MAX).then_some(self.restore_dim);
        (self.stem_state, restore_dim, self.old_off, self.rd)
    }
}

/// Donnelly block-marker descent with scalar pruning/backtracking.
#[derive(Copy, Clone, Debug)]
pub struct DonnellySimdDescent<const CL: u32, const VB: u32, const K: usize> {
    core: DonnellyCore<CL, VB, K>,
    _marker: PhantomData<Block3>,
}

impl<const CL: u32, const VB: u32, const K: usize> DonnellySimdDescent<CL, VB, K> {
    #[inline(always)]
    fn can_take_full_block(&self, stems_len: usize, max_stem_level: i32) -> bool {
        let block_width = 1usize << Self::BLOCK_SIZE;
        let block_base_idx = self.stem_idx();
        let minor_tri_height = (CL / VB).ilog2();

        self.level() + Self::BLOCK_SIZE as i32 - 1 <= max_stem_level
            && block_base_idx + block_width <= stems_len
            && Self::BLOCK_SIZE as u32 == minor_tri_height
            && self.core.minor_level() == 0
    }

    #[inline(always)]
    fn block_child(&self, child_idx: u8) -> Self {
        let mut child = *self;
        child
            .core
            .traverse_block(child_idx, Self::BLOCK_SIZE as u32);
        child
    }

    #[inline(always)]
    fn child_new_off<A, O, D, const K2: usize>(
        stems: &[A],
        block_base_idx: usize,
        child_idx: usize,
        query_wide_val: O,
    ) -> O
    where
        A: Axis<Coord = A>,
        O: Axis<Coord = O>,
        D: crate::dist::DistanceMetricCore<A, Output = O>,
    {
        let (lower_offset, upper_offset) = child_interval_bounds_block3(child_idx);

        let lower = if lower_offset == 255 {
            A::min_value()
        } else {
            unsafe { *stems.get_unchecked(block_base_idx + lower_offset as usize) }
        };

        let upper = if upper_offset == 255 {
            A::max_value()
        } else {
            unsafe { *stems.get_unchecked(block_base_idx + upper_offset as usize) }
        };

        let lower_wide = D::widen_coord(lower);
        let upper_wide = D::widen_coord(upper);
        interval_distance_1d(query_wide_val, lower_wide, upper_wide)
    }
}

impl<const CL: u32, const VB: u32, const K: usize> StemStrategy for DonnellySimdDescent<CL, VB, K> {
    const ROOT_IDX: usize = 0;
    const BLOCK_SIZE: usize = 3;

    type DeferredState = Self;
    type StackContext<A> = DonnellySimdDescentStackContext<A, Self::DeferredState>;
    type Stack<A> = QueryStack<A, Self>;

    #[inline]
    fn new(stems_ptr: NonNull<u8>) -> Self {
        debug_assert!(CL >= VB);

        Self {
            core: DonnellyCore::new(stems_ptr),
            _marker: PhantomData,
        }
    }

    #[inline(always)]
    fn stem_idx(&self) -> usize {
        self.core.stem_idx()
    }

    #[inline(always)]
    fn deferred_state(&self) -> Self::DeferredState {
        *self
    }

    #[inline(always)]
    fn rehydrate_deferred_state(&mut self, state: Self::DeferredState) {
        *self = state;
    }

    #[inline(always)]
    fn leaf_idx(&self) -> usize {
        self.core.leaf_idx()
    }

    #[inline(always)]
    fn dim(&self) -> usize {
        self.core.level() as usize / Self::BLOCK_SIZE % K
    }

    #[inline(always)]
    fn construction_dim(&self) -> usize {
        self.core.level() as usize / Self::BLOCK_SIZE % K
    }

    #[inline(always)]
    fn level(&self) -> i32 {
        self.core.level()
    }

    #[inline(always)]
    fn traverse(&mut self, is_right: bool) {
        self.core.traverse(is_right)
    }

    #[inline(always)]
    fn branch(&mut self) -> Self {
        Self {
            core: self.core.branch(),
            _marker: PhantomData,
        }
    }

    #[inline(always)]
    fn child_indices(&self) -> (usize, usize) {
        self.core.child_indices()
    }

    fn get_stem_node_count_from_leaf_node_count(leaf_node_count: usize) -> usize {
        if leaf_node_count < 2 {
            0
        } else {
            leaf_node_count.next_power_of_two() - 1
        }
    }

    fn stem_node_padding_factor() -> usize {
        50
    }

    fn trim_unneeded_stems<A: Axis<Coord = A>>(stems: &mut AVec<A>, max_stem_level: usize) {
        let stems_ptr = NonNull::new(stems.as_ptr() as *mut u8).unwrap();
        if !stems.is_empty() {
            let mut so = Self::new(stems_ptr);
            loop {
                let val = &stems[so.stem_idx()];
                let is_right_child = !A::is_max_value(*val);
                so.traverse(is_right_child);
                if so.level() as usize == max_stem_level {
                    break;
                }
            }

            #[cfg(debug_assertions)]
            {
                for i in (so.stem_idx() + 1)..stems.len() {
                    debug_assert!(A::is_max_value(stems[i]), "stems[{i}] = {}", stems[i]);
                }
            }

            stems.truncate(so.stem_idx() + 1);
        }
    }

    fn get_leaf_idx<A: Axis, const K2: usize>(
        stems: &[A],
        query: &[A; K2],
        max_stem_level: i32,
    ) -> usize
    where
        Self: Sized,
    {
        let stems_ptr = NonNull::new(stems.as_ptr() as *mut u8).unwrap();
        let mut strat = Self::new(stems_ptr);

        while strat.level() <= max_stem_level {
            let dim = strat.dim();
            let query_val = unsafe { *query.get_unchecked(dim) };

            if strat.can_take_full_block(stems.len(), max_stem_level) {
                let child_idx = compare_block3(stems, query_val, strat.stem_idx());
                strat
                    .core
                    .traverse_block(child_idx, Self::BLOCK_SIZE as u32);
            } else {
                let stem_idx = strat.stem_idx();
                let is_right = if stem_idx < stems.len() {
                    query_val >= unsafe { *stems.get_unchecked(stem_idx) }
                } else {
                    false
                };
                strat.traverse(is_right);
            }
        }

        strat.leaf_idx()
    }

    #[inline(always)]
    fn backtracking_traverse_step<A, O, D, const K2: usize>(
        &mut self,
        stems: &[A],
        query: &[A; K2],
        query_wide: &[O; K2],
        off: &mut [O; K2],
        dim: &mut usize,
        rd: O,
        max_stem_level: i32,
        best_dist: O,
        stack: &mut Self::Stack<O>,
    ) -> bool
    where
        Self: Sized,
        A: Axis<Coord = A>,
        O: Axis<Coord = O> + crate::stem_strategy::donnelly_2_blockmarker_simd::BacktrackBlock3,
        D: crate::dist::DistanceMetricCore<A, Output = O>
            + crate::stem_strategy::donnelly_2_blockmarker_simd::backtrack_traits::DistanceMetricSimdBlock3<
                A,
                K2,
                O,
            >,
    {
        if self.level() > max_stem_level {
            return false;
        }

        let dim_val = *dim;
        let query_val = unsafe { *query.get_unchecked(dim_val) };
        let query_wide_val = unsafe { *query_wide.get_unchecked(dim_val) };

        if !self.can_take_full_block(stems.len(), max_stem_level) {
            let stem_idx = self.stem_idx();
            let pivot = if stem_idx < stems.len() {
                unsafe { *stems.get_unchecked(stem_idx) }
            } else {
                A::max_value()
            };

            if pivot < A::max_value() {
                let is_right_child = query_val >= pivot;
                let far_ctx = self.branch_relative(is_right_child);
                let pivot_wide: O = D::widen_coord(pivot);
                let new_off = O::saturating_dist(query_wide_val, pivot_wide);
                let rd_far = D::rect_dist_after_update(rd, off, *dim, new_off);

                if O::cmp(rd_far, best_dist) != std::cmp::Ordering::Greater {
                    stack.push(Self::StackContext::<O>::from_parts_with_restore_dim(
                        far_ctx.deferred_state(),
                        dim_val,
                        new_off,
                        rd_far,
                    ));
                }
            } else {
                self.traverse(false);
            }

            *dim = self.dim();
            return true;
        }

        let block_base_idx = self.stem_idx();
        let child_idx = compare_block3(stems, query_val, block_base_idx) as usize;
        let base = *self;

        let mut candidate_idx = [0u8; 7];
        let mut candidate_new_off = [O::zero(); 7];
        let mut candidate_rd = [O::zero(); 7];
        let mut candidate_count = 0usize;

        for sibling_idx in 0..8 {
            if sibling_idx == child_idx {
                continue;
            }

            let new_off = Self::child_new_off::<A, O, D, K2>(
                stems,
                block_base_idx,
                sibling_idx,
                query_wide_val,
            );
            let rd_far = D::rect_dist_after_update(rd, off, *dim, new_off);

            if O::cmp(rd_far, best_dist) == std::cmp::Ordering::Greater {
                continue;
            }

            let mut insert_at = candidate_count;
            for i in 0..candidate_count {
                if O::cmp(rd_far, candidate_rd[i]) == std::cmp::Ordering::Less {
                    insert_at = i;
                    break;
                }
            }

            for i in (insert_at..candidate_count).rev() {
                candidate_idx[i + 1] = candidate_idx[i];
                candidate_new_off[i + 1] = candidate_new_off[i];
                candidate_rd[i + 1] = candidate_rd[i];
            }

            candidate_idx[insert_at] = sibling_idx as u8;
            candidate_new_off[insert_at] = new_off;
            candidate_rd[insert_at] = rd_far;
            candidate_count += 1;
        }

        for i in (0..candidate_count).rev() {
            let sibling = base.block_child(candidate_idx[i]);
            stack.push(Self::StackContext::<O>::from_parts_with_restore_dim(
                sibling.deferred_state(),
                dim_val,
                candidate_new_off[i],
                candidate_rd[i],
            ));
        }

        let child_new_off =
            Self::child_new_off::<A, O, D, K2>(stems, block_base_idx, child_idx, query_wide_val);
        unsafe { *off.get_unchecked_mut(dim_val) = child_new_off };

        self.core
            .traverse_block(child_idx as u8, Self::BLOCK_SIZE as u32);
        *dim = self.dim();
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dist::SquaredEuclidean;
    use crate::kd_tree::query_stack::ScalarStackContext;
    use aligned_vec::avec;

    #[test]
    fn test_donnelly_simd_descent_basics() {
        let stems = vec![10.0f64; 100];
        let stems_ptr = NonNull::new(stems.as_ptr() as *mut u8).unwrap();
        let mut strat = DonnellySimdDescent::<64, 8, 3>::new(stems_ptr);

        assert_eq!(strat.stem_idx(), 0);
        assert_eq!(strat.level(), 0);
        assert_eq!(strat.dim(), 0);

        strat.traverse(true);
        assert!(strat.stem_idx() > 0);
        assert_eq!(strat.level(), 1);
    }

    fn build_test_block3_pivots_f64() -> [f64; 8] {
        [0.2, 0.4, 0.6, 0.1, 0.3, 0.5, 0.7, f64::INFINITY]
    }

    #[test]
    fn test_stack_context_round_trips_restore_dim() {
        type Strat = DonnellySimdDescent<64, 8, 3>;

        let stems = [f64::INFINITY; 8];
        let stems_ptr = NonNull::new(stems.as_ptr() as *mut u8).unwrap();
        let strat = Strat::new(stems_ptr);

        let ctx = DonnellySimdDescentStackContext::new(strat, 2, 0.25f64, 1.5);
        let (state, restore_dim, old_off, rd) =
            <DonnellySimdDescentStackContext<f64, Strat> as ScalarStackContext<f64, Strat>>::into_parts_with_restore_dim(ctx);

        assert_eq!(state.stem_idx(), strat.stem_idx());
        assert_eq!(restore_dim, Some(2));
        assert_eq!(old_off, 0.25);
        assert_eq!(rd, 1.5);

        let default_ctx = <DonnellySimdDescentStackContext<f64, Strat> as ScalarStackContext<
            f64,
            Strat,
        >>::from_parts(strat, 0.125, 0.75);
        let (_, restore_dim, old_off, rd) =
            <DonnellySimdDescentStackContext<f64, Strat> as ScalarStackContext<f64, Strat>>::into_parts_with_restore_dim(default_ctx);

        assert_eq!(restore_dim, None);
        assert_eq!(old_off, 0.125);
        assert_eq!(rd, 0.75);
    }

    #[test]
    fn test_can_take_full_block_only_at_minor_triangle_boundary() {
        type Strat = DonnellySimdDescent<64, 8, 3>;

        let stems = build_test_block3_pivots_f64();
        let stems_ptr = NonNull::new(stems.as_ptr() as *mut u8).unwrap();
        let mut strat = Strat::new(stems_ptr);

        assert!(strat.can_take_full_block(stems.len(), 2));
        assert!(!strat.can_take_full_block(7, 2));
        assert!(!strat.can_take_full_block(stems.len(), 1));

        strat.traverse(false);
        assert!(!strat.can_take_full_block(stems.len(), 3));
    }

    #[test]
    fn test_block_child_matches_manual_traverse_block() {
        type Strat = DonnellySimdDescent<64, 8, 3>;

        let stems = build_test_block3_pivots_f64();
        let stems_ptr = NonNull::new(stems.as_ptr() as *mut u8).unwrap();
        let strat = Strat::new(stems_ptr);

        let child = strat.block_child(5);

        let mut manual = strat;
        manual.core.traverse_block(5, Strat::BLOCK_SIZE as u32);

        assert_eq!(child.stem_idx(), manual.stem_idx());
        assert_eq!(child.level(), manual.level());
        assert_eq!(child.leaf_idx(), manual.leaf_idx());
        assert_eq!(child.dim(), manual.dim());
    }

    #[test]
    fn test_child_new_off_uses_child_interval_bounds() {
        type Strat = DonnellySimdDescent<64, 8, 3>;

        let stems = build_test_block3_pivots_f64();
        let query = 0.45;

        let left_off =
            Strat::child_new_off::<f64, f64, SquaredEuclidean<f64>, 3>(&stems, 0, 0, query);
        let middle_off =
            Strat::child_new_off::<f64, f64, SquaredEuclidean<f64>, 3>(&stems, 0, 4, query);
        let right_off =
            Strat::child_new_off::<f64, f64, SquaredEuclidean<f64>, 3>(&stems, 0, 7, query);

        assert!((left_off - 0.35).abs() < 1e-12);
        assert_eq!(middle_off, 0.0);
        assert!((right_off - 0.25).abs() < 1e-12);
    }

    #[test]
    fn test_leaf_metadata_helpers_match_current_layout() {
        type Strat = DonnellySimdDescent<64, 8, 3>;

        assert_eq!(Strat::get_stem_node_count_from_leaf_node_count(0), 0);
        assert_eq!(Strat::get_stem_node_count_from_leaf_node_count(1), 0);
        assert_eq!(Strat::get_stem_node_count_from_leaf_node_count(2), 1);
        assert_eq!(Strat::get_stem_node_count_from_leaf_node_count(7), 7);
        assert_eq!(Strat::get_stem_node_count_from_leaf_node_count(9), 15);
        assert_eq!(Strat::stem_node_padding_factor(), 50);
    }

    #[test]
    fn test_trim_unneeded_stems_truncates_to_last_reachable_left_path_node() {
        type Strat = DonnellySimdDescent<64, 8, 3>;

        let mut stems = avec![f64::INFINITY; 100];
        Strat::trim_unneeded_stems(&mut stems, 2);

        let ptr = NonNull::dangling();
        let mut manual = Strat::new(ptr);
        loop {
            manual.traverse(false);
            if manual.level() == 2 {
                break;
            }
        }

        assert_eq!(stems.len(), manual.stem_idx() + 1);
        assert!(stems.iter().all(|&x| x.is_infinite()));
    }

    #[test]
    fn test_get_leaf_idx_matches_manual_full_block_traversal() {
        type Strat = DonnellySimdDescent<64, 8, 3>;

        let stems = build_test_block3_pivots_f64();
        let query = [0.45, 0.0, 0.0];
        let leaf_idx = Strat::get_leaf_idx::<f64, 3>(&stems, &query, 2);

        let stems_ptr = NonNull::new(stems.as_ptr() as *mut u8).unwrap();
        let mut manual = Strat::new(stems_ptr);
        let child_idx = compare_block3(&stems, query[0], 0);
        manual.core.traverse_block(child_idx, 3);

        assert_eq!(leaf_idx, manual.leaf_idx());
    }

    #[test]
    fn test_get_leaf_idx_matches_manual_scalar_fallback() {
        type Strat = DonnellySimdDescent<64, 8, 3>;

        let stems = build_test_block3_pivots_f64();
        let query = [0.45, 0.0, 0.0];
        let leaf_idx = Strat::get_leaf_idx::<f64, 3>(&stems, &query, 1);

        let stems_ptr = NonNull::new(stems.as_ptr() as *mut u8).unwrap();
        let mut manual = Strat::new(stems_ptr);
        manual.traverse(true);
        manual.traverse(false);

        assert_eq!(leaf_idx, manual.leaf_idx());
    }

    #[test]
    fn test_backtracking_step_full_block_updates_state_and_pushes_sorted_siblings() {
        type Strat = DonnellySimdDescent<64, 8, 3>;

        let stems = build_test_block3_pivots_f64();
        let stems_ptr = NonNull::new(stems.as_ptr() as *mut u8).unwrap();
        let mut strat = Strat::new(stems_ptr);
        let query = [0.45, 0.0, 0.0];
        let query_wide = query;
        let mut off = [0.0; 3];
        let mut dim = 0usize;
        let mut stack = <Strat as StemStrategy>::Stack::<f64>::default();

        let child_idx = compare_block3(&stems, query[0], 0) as usize;
        let mut expected = Vec::new();
        for sibling_idx in 0..8 {
            if sibling_idx == child_idx {
                continue;
            }

            let new_off = Strat::child_new_off::<f64, f64, SquaredEuclidean<f64>, 3>(
                &stems,
                0,
                sibling_idx,
                query_wide[0],
            );
            let rd_far = new_off * new_off;
            if rd_far <= 0.2 {
                expected.push((sibling_idx as u8, new_off, rd_far));
            }
        }
        expected.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

        let stepped = strat.backtracking_traverse_step::<f64, f64, SquaredEuclidean<f64>, 3>(
            &stems,
            &query,
            &query_wide,
            &mut off,
            &mut dim,
            0.0,
            2,
            0.2,
            &mut stack,
        );

        assert!(stepped);
        assert_eq!(off[0], 0.0);
        assert_eq!(dim, strat.dim());

        let mut popped = Vec::new();
        while let Some(ctx) = stack.pop() {
            let (state, restore_dim, old_off, rd) =
                <DonnellySimdDescentStackContext<f64, Strat> as ScalarStackContext<f64, Strat>>::into_parts_with_restore_dim(ctx);
            popped.push((state.stem_idx(), restore_dim, old_off, rd));
        }

        assert_eq!(popped.len(), expected.len());
        for ((stem_idx, restore_dim, old_off, rd), (sibling_idx, expected_off, expected_rd)) in
            popped.into_iter().zip(expected.into_iter())
        {
            let expected_state = Strat::new(stems_ptr).block_child(sibling_idx);
            assert_eq!(stem_idx, expected_state.stem_idx());
            assert_eq!(restore_dim, Some(0));
            assert_eq!(old_off, expected_off);
            assert_eq!(rd, expected_rd);
        }
    }

    #[test]
    fn test_backtracking_step_scalar_fallback_pushes_far_context() {
        type Strat = DonnellySimdDescent<64, 8, 3>;

        let stems = build_test_block3_pivots_f64();
        let stems_ptr = NonNull::new(stems.as_ptr() as *mut u8).unwrap();
        let mut strat = Strat::new(stems_ptr);
        let query = [0.45, 0.0, 0.0];
        let query_wide = query;
        let mut off = [0.0; 3];
        let mut dim = 0usize;
        let mut stack = <Strat as StemStrategy>::Stack::<f64>::default();

        let stepped = strat.backtracking_traverse_step::<f64, f64, SquaredEuclidean<f64>, 3>(
            &stems,
            &query,
            &query_wide,
            &mut off,
            &mut dim,
            0.0,
            0,
            1.0,
            &mut stack,
        );

        assert!(stepped);
        assert_eq!(strat.level(), 1);
        assert_eq!(strat.dim(), 0);

        let popped = stack.pop().expect("expected deferred far context");
        let (state, restore_dim, old_off, rd) =
            <DonnellySimdDescentStackContext<f64, Strat> as ScalarStackContext<f64, Strat>>::into_parts_with_restore_dim(popped);

        let mut expected_far = Strat::new(stems_ptr);
        expected_far.traverse(false);

        assert_eq!(state.stem_idx(), expected_far.stem_idx());
        assert_eq!(restore_dim, Some(0));
        assert!((old_off - 0.25).abs() < 1e-12);
        assert!((rd - 0.0625).abs() < 1e-12);
        assert!(stack.pop().is_none());
    }

    #[test]
    fn test_backtracking_step_returns_false_past_max_level() {
        type Strat = DonnellySimdDescent<64, 8, 3>;

        let stems = build_test_block3_pivots_f64();
        let stems_ptr = NonNull::new(stems.as_ptr() as *mut u8).unwrap();
        let mut strat = Strat::new(stems_ptr);
        strat.core.traverse_block(4, 3);

        let query = [0.45, 0.0, 0.0];
        let query_wide = query;
        let mut off = [0.0; 3];
        let mut dim = strat.dim();
        let mut stack = <Strat as StemStrategy>::Stack::<f64>::default();

        let stepped = strat.backtracking_traverse_step::<f64, f64, SquaredEuclidean<f64>, 3>(
            &stems,
            &query,
            &query_wide,
            &mut off,
            &mut dim,
            0.0,
            2,
            1.0,
            &mut stack,
        );

        assert!(!stepped);
        assert!(stack.pop().is_none());
    }
}
