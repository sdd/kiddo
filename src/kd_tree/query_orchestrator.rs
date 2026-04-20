use crate::dist::KdTreeDistanceMetric;
use crate::kd_tree::query_stack::{ScalarStackContext, StackTrait};
use crate::kd_tree::traits::QueryContext;
use crate::kd_tree::{KdTreeAccessor, StemLeafResolution};
use crate::stem_strategies::{
    donnelly_2_blockmarker_simd::{BacktrackBlock3, BacktrackBlock4},
    DistanceMetricSimdBlock3, DistanceMetricSimdBlock4, SimdPrune, SimdSelectBestChildBlock3,
};
use crate::traits_unified_2::{AxisUnified, Basics, LeafStrategy};
use crate::StemStrategy;
use std::any::{Any, TypeId};
use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::ptr::NonNull;

mod simd;

thread_local! {
    static QUERY_STACKS: UnsafeCell<HashMap<TypeId, Box<dyn Any>>> =
        UnsafeCell::new(HashMap::new());
}

#[inline]
pub(crate) fn with_tls_query_stack<S, R>(f: impl FnOnce(&mut S) -> R) -> R
where
    S: Default + 'static,
{
    QUERY_STACKS.with(|stacks| {
        let stacks = unsafe { &mut *stacks.get() };
        let entry = stacks
            .entry(TypeId::of::<S>())
            .or_insert_with(|| Box::new(S::default()));
        let stack = entry
            .downcast_mut::<S>()
            .expect("query stack type mismatch");
        f(stack)
    })
}

#[derive(Clone, Copy, Debug)]
struct Block3PendingSelection<O> {
    child_idx: u8,
    remaining_mask: u8,
    child_rd: O,
    child_off: O,
}

#[inline(always)]
fn select_block3_pending_child<O>(
    rd_values: &[O; 8],
    new_off_values: &[O; 8],
    candidate_mask: u8,
) -> Option<Block3PendingSelection<O>>
where
    O: AxisUnified<Coord = O> + SimdSelectBestChildBlock3,
{
    if candidate_mask == 0 {
        return None;
    }

    let child_idx =
        O::simd_select_best_child_block3(rd_values, candidate_mask).unwrap_or_else(|| unsafe {
            debug_assert!(false, "candidate_mask != 0");
            core::hint::unreachable_unchecked()
        });

    Some(Block3PendingSelection {
        child_idx,
        remaining_mask: candidate_mask & !(1u8 << child_idx),
        child_rd: unsafe { *rd_values.get_unchecked(child_idx as usize) },
        child_off: unsafe { *new_off_values.get_unchecked(child_idx as usize) },
    })
}

#[inline(always)]
fn rebuild_interval_offs<O, const K: usize>(
    query_wide: &[O; K],
    lower: &[O; K],
    upper: &[O; K],
) -> [O; K]
where
    O: AxisUnified<Coord = O>,
{
    let mut off = [O::zero(); K];
    for dim in 0..K {
        off[dim] = crate::stem_strategies::donnelly_2_blockmarker_simd::interval_distance_1d(
            query_wide[dim],
            lower[dim],
            upper[dim],
        );
    }
    off
}

#[cfg(feature = "test_utils")]
#[inline]
fn force_mapped_simd_block_step() -> bool {
    matches!(
        std::env::var("KIDDO_FORCE_MAPPED_SIMD_BLOCK_STEP")
            .ok()
            .as_deref(),
        Some("1") | Some("true") | Some("TRUE") | Some("yes") | Some("YES")
    )
}

#[cfg(not(feature = "test_utils"))]
#[inline]
fn force_mapped_simd_block_step() -> bool {
    false
}

#[cfg(feature = "cargo_asm")]
pub mod cargo_asm {
    use super::select_block3_pending_child;

    /// Hook for cargo-asm to render the exact Block3 pending prune/select kernel directly.
    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn donnelly_block3_pending_select_f64_cargo_asm_hook(
        rd_values: &[f64; 8],
        new_off_values: &[f64; 8],
        candidate_mask: u8,
    ) -> Option<(u8, u8, f64, f64)> {
        select_block3_pending_child(rd_values, new_off_values, candidate_mask).map(|selection| {
            (
                selection.child_idx,
                selection.remaining_mask,
                selection.child_rd,
                selection.child_off,
            )
        })
    }
}

impl<Tree, A, T, SS, LS, const K: usize, const B: usize> KdTreeQueryOps<A, T, SS, LS, K, B> for Tree
where
    Tree: KdTreeAccessor<A, T, SS, LS, K, B>,
    A: AxisUnified<Coord = A>,
    T: Basics + Copy + Default + PartialOrd + PartialEq,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B>,
{
}

#[allow(missing_docs, private_bounds)]
pub trait KdTreeQueryOps<A, T, SS, LS, const K: usize, const B: usize>:
    KdTreeAccessor<A, T, SS, LS, K, B> + Sized
where
    A: AxisUnified<Coord = A>,
    T: Basics + Copy + Default + PartialOrd + PartialEq,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B>,
{
    #[inline]
    fn get_leaf_idx(&self, query: &[A; K]) -> usize {
        if self.stem_leaf_resolution().uses_arithmetic() {
            self.get_leaf_idx_unmapped(query)
        } else {
            self.get_leaf_idx_mapped(query)
        }
    }

    /// Non-backtracking query
    #[inline]
    fn straight_query<QC, O>(&self, query_ctx: QC, mut process_leaf: impl FnMut(usize))
    where
        QC: QueryContext<A, O, K>,
    {
        let leaf_idx = self.get_leaf_idx(query_ctx.query());

        tracing::trace!(%leaf_idx, "processing leaf");
        process_leaf(leaf_idx);
    }

    /// Get the leaf index for a query (unmapped leaves)
    #[inline]
    fn get_leaf_idx_unmapped(&self, query: &[A; K]) -> usize {
        SS::get_leaf_idx(self.stems(), query, self.max_stem_level())
    }

    /// Get the leaf index for a query (mapped leaves)
    #[inline(always)]
    fn get_leaf_idx_mapped(&self, query: &[A; K]) -> usize {
        let stems_ptr = NonNull::new(self.stems().as_ptr() as *mut u8).unwrap();
        let mut stem_strat: SS = SS::new(stems_ptr);

        while stem_strat.level() <= self.max_stem_level() {
            if let Some(leaf_idx) = self.resolve_terminal_stem(stem_strat.stem_idx()) {
                return leaf_idx;
            }

            let pivot = unsafe { self.stems().get_unchecked(stem_strat.stem_idx()) };
            let is_right_child: bool = *unsafe { query.get_unchecked(stem_strat.dim()) } >= *pivot;
            stem_strat.traverse(is_right_child);
        }

        self.stem_leaf_resolution()
            .resolve_terminal_stem_idx(stem_strat.stem_idx(), stem_strat.leaf_idx())
    }

    // TODO: don't like this structure
    /// Check if a stem points directly to a leaf
    #[inline(always)]
    fn resolve_terminal_stem(&self, stem_idx: usize) -> Option<usize> {
        if self.stem_leaf_resolution().is_terminal_stem_idx(stem_idx) {
            Some(
                self.stem_leaf_resolution()
                    .resolve_terminal_stem_idx(stem_idx, 0),
            )
        } else {
            None
        }
    }

    #[allow(unused)]
    #[inline(always)]
    fn subtree_may_contain_leaf(
        &self,
        stem_idx: usize,
        level: i32,
        leaf_idx_prefix: usize,
    ) -> bool {
        let _ = (level, leaf_idx_prefix);
        stem_idx < self.stems().len() || self.stem_leaf_resolution().is_terminal_stem_idx(stem_idx)
    }

    /// Backtracking query
    #[inline(always)]
    fn backtracking_query<QC, O, D>(
        &self,
        query_ctx: &mut QC,
        process_leaf: impl FnMut(usize, &[O; K], &mut QC),
    ) where
        QC: QueryContext<A, O, K>,
        O: AxisUnified<Coord = O>
            + SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4,
        D: KdTreeDistanceMetric<A, K, Output = O>
            + DistanceMetricSimdBlock3<A, K, O>
            + DistanceMetricSimdBlock4<A, K, O>,
        SS::Stack<O>: StackTrait<O, SS> + Default + 'static,
    {
        if self.stem_leaf_resolution().uses_arithmetic() && SS::BLOCK_SIZE == 1 {
            self.arithmetic_query::<QC, O, D>(query_ctx, process_leaf);
            return;
        }

        with_tls_query_stack::<SS::Stack<O>, _>(|stack| {
            stack.clear();
            self.backtracking_query_with_stack::<QC, O, D>(query_ctx, stack, process_leaf);
        });
    }

    /// Backtracking query with explicit stack
    #[inline(always)]
    fn backtracking_query_with_stack<QC, O, D>(
        &self,
        query_ctx: &mut QC,
        stack: &mut SS::Stack<O>,
        process_leaf: impl FnMut(usize, &[O; K], &mut QC),
    ) where
        QC: QueryContext<A, O, K>,
        O: AxisUnified<Coord = O>
            + SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4,
        D: KdTreeDistanceMetric<A, K, Output = O>
            + DistanceMetricSimdBlock3<A, K, O>
            + DistanceMetricSimdBlock4<A, K, O>,
        SS::Stack<O>: StackTrait<O, SS>,
    {
        if self.stem_leaf_resolution().uses_arithmetic() && SS::BLOCK_SIZE == 1 {
            self.arithmetic_query_with_stack::<QC, O, D>(query_ctx, stack, process_leaf);
            return;
        }

        SS::backtracking_query_with_stack::<Self, A, T, O, D, QC, LS, K, B>(
            self,
            query_ctx,
            stack,
            process_leaf,
        );
    }

    /// Implementation of backtracking query with scalar stack.
    /// Called by default StemStrategy::backtracking_query_with_stack implementation.
    /// SIMD strategies override the trait method with custom implementations.
    #[inline(always)]
    fn backtracking_query_with_stack_impl<QC, O, D>(
        &self,
        query_ctx: &mut QC,
        stack: &mut SS::Stack<O>,
        mut process_leaf: impl FnMut(usize, &[O; K], &mut QC),
    ) where
        QC: QueryContext<A, O, K>,
        O: AxisUnified<Coord = O> + BacktrackBlock3 + BacktrackBlock4,
        D: KdTreeDistanceMetric<A, K, Output = O>
            + DistanceMetricSimdBlock3<A, K, O>
            + DistanceMetricSimdBlock4<A, K, O>,
        SS::Stack<O>: StackTrait<O, SS>,
        SS::StackContext<O>: ScalarStackContext<O, SS::DeferredState>,
    {
        let stems_ptr = NonNull::new(self.stems().as_ptr() as *mut u8).unwrap();
        let mut stem_strat: SS = SS::new(stems_ptr);

        let query: [A; K] = *query_ctx.query();
        let mut query_wide: [O; K] = [O::zero(); K];
        for dim in 0..K {
            query_wide[dim] = D::widen_coord(query[dim]);
        }

        let mut off = [O::zero(); K];
        stack.push(SS::StackContext::<O>::from_parts(
            stem_strat.deferred_state(),
            O::zero(),
            O::zero(),
        ));
        #[cfg(feature = "result_collection_stats")]
        crate::result_collection_stats::record_query_stack_push();

        while let Some(stack_ctx) = stack.pop() {
            #[cfg(feature = "result_collection_stats")]
            crate::result_collection_stats::record_query_stack_pop();
            #[cfg(feature = "test_utils")]
            crate::test_utils::exact_query_stats::record_scalar_stack_pop();

            let (stem_state, restore_dim, old_off, rd) =
                SS::StackContext::<O>::into_parts_with_restore_dim(stack_ctx);
            stem_strat.rehydrate_deferred_state(stem_state);
            let mut dim = stem_strat.dim();
            let restore_dim = restore_dim.unwrap_or(dim);
            tracing::trace!(%dim, %old_off, %rd, ?off, "Popped stack context");

            let max_dist = query_ctx.max_dist();
            let rd_vs_max = O::cmp(rd, max_dist);

            // TOOO: investigate into whether prune_on_equal_max_dist can be removed
            let should_prune = rd_vs_max == std::cmp::Ordering::Greater
                || (query_ctx.prune_on_equal_max_dist() && rd_vs_max == std::cmp::Ordering::Equal);
            if should_prune {
                #[cfg(feature = "result_collection_stats")]
                crate::result_collection_stats::record_query_prune();
                tracing::trace!(%rd, %max_dist, "SCALAR Prune check: PRUNE");
                continue;
            }
            tracing::trace!(%rd, %max_dist, "SCALAR Prune check: VISIT");

            tracing::trace!(
                "Restoring off[{}]. was {}, now {}",
                restore_dim,
                unsafe { *off.get_unchecked(restore_dim) },
                old_off
            );
            unsafe { *off.get_unchecked_mut(restore_dim) = old_off };

            let best_dist = query_ctx.max_dist();
            if let Some(leaf_idx) = self.traverse_to_leaf::<O, D>(
                &query,
                &query_wide,
                &mut stem_strat,
                &mut off,
                &mut dim,
                rd,
                best_dist,
                stack,
            ) {
                tracing::trace!(%leaf_idx, "processing leaf");
                process_leaf(leaf_idx, &query_wide, query_ctx);
            }
        }
    }

    /// Arithmetic-resolution backtracking query.
    #[inline(always)]
    fn arithmetic_query<QC, O, D>(
        &self,
        query_ctx: &mut QC,
        process_leaf: impl FnMut(usize, &[O; K], &mut QC),
    ) where
        QC: QueryContext<A, O, K>,
        O: AxisUnified<Coord = O> + BacktrackBlock3 + BacktrackBlock4,
        D: KdTreeDistanceMetric<A, K, Output = O>
            + DistanceMetricSimdBlock3<A, K, O>
            + DistanceMetricSimdBlock4<A, K, O>,
        SS::Stack<O>: StackTrait<O, SS> + Default + 'static,
    {
        with_tls_query_stack::<SS::Stack<O>, _>(|stack| {
            self.arithmetic_query_with_stack::<QC, O, D>(query_ctx, stack, process_leaf);
        });
    }

    /// Arithmetic-resolution backtracking query with explicit stack.
    #[inline(always)]
    fn arithmetic_query_with_stack<QC, O, D>(
        &self,
        query_ctx: &mut QC,
        stack: &mut SS::Stack<O>,
        process_leaf: impl FnMut(usize, &[O; K], &mut QC),
    ) where
        QC: QueryContext<A, O, K>,
        O: AxisUnified<Coord = O> + BacktrackBlock3 + BacktrackBlock4,
        D: KdTreeDistanceMetric<A, K, Output = O>
            + DistanceMetricSimdBlock3<A, K, O>
            + DistanceMetricSimdBlock4<A, K, O>,
        SS::Stack<O>: StackTrait<O, SS>,
        SS::StackContext<O>: ScalarStackContext<O, SS::DeferredState>,
    {
        SS::arithmetic_query_with_stack::<Self, A, T, O, D, QC, LS, K, B>(
            self,
            query_ctx,
            stack,
            process_leaf,
        );
    }

    /// Implementation of arithmetic-resolution query with scalar stack.
    /// Called by default StemStrategy::arithmetic_query_with_stack implementation.
    #[inline(always)]
    fn arithmetic_query_with_stack_impl<QC, O, D>(
        &self,
        query_ctx: &mut QC,
        stack: &mut SS::Stack<O>,
        mut process_leaf: impl FnMut(usize, &[O; K], &mut QC),
    ) where
        QC: QueryContext<A, O, K>,
        O: AxisUnified<Coord = O> + BacktrackBlock3 + BacktrackBlock4,
        D: KdTreeDistanceMetric<A, K, Output = O>
            + DistanceMetricSimdBlock3<A, K, O>
            + DistanceMetricSimdBlock4<A, K, O>,
        SS::Stack<O>: StackTrait<O, SS>,
    {
        if self.size() == 0 {
            return;
        }

        let stems_ptr = NonNull::new(self.stems().as_ptr() as *mut u8).unwrap();
        let mut stem_strat: SS = SS::new(stems_ptr);

        let query: [A; K] = *query_ctx.query();
        let mut query_wide: [O; K] = [O::zero(); K];
        for dim in 0..K {
            query_wide[dim] = D::widen_coord(query[dim]);
        }

        let mut off = [O::zero(); K];
        stack.clear();
        stack.push(SS::StackContext::<O>::from_parts(
            stem_strat.deferred_state(),
            O::zero(),
            O::zero(),
        ));
        #[cfg(feature = "result_collection_stats")]
        crate::result_collection_stats::record_query_stack_push();

        while let Some(stack_ctx) = stack.pop() {
            #[cfg(feature = "result_collection_stats")]
            crate::result_collection_stats::record_query_stack_pop();
            let (stem_state, restore_dim, old_off, rd) =
                SS::StackContext::<O>::into_parts_with_restore_dim(stack_ctx);
            stem_strat.rehydrate_deferred_state(stem_state);
            let mut dim = stem_strat.dim();
            let restore_dim = restore_dim.unwrap_or(dim);

            let rd_vs_max = O::cmp(rd, query_ctx.max_dist());
            let should_prune = rd_vs_max == std::cmp::Ordering::Greater
                || (query_ctx.prune_on_equal_max_dist() && rd_vs_max == std::cmp::Ordering::Equal);
            if should_prune {
                #[cfg(feature = "result_collection_stats")]
                crate::result_collection_stats::record_query_prune();
                continue;
            }

            unsafe { *off.get_unchecked_mut(restore_dim) = old_off };
            let best_dist = query_ctx.max_dist();

            loop {
                if stem_strat.level() > self.max_stem_level() {
                    break;
                }

                let pivot = unsafe { *self.stems().get_unchecked(stem_strat.stem_idx()) };
                if pivot < A::max_value() {
                    let query_elem = unsafe { *query.get_unchecked(dim) };
                    let is_right_child = query_elem >= pivot;
                    let far_ctx = stem_strat.branch_relative(is_right_child);

                    let pivot_wide = D::widen_coord(pivot);
                    let query_elem_wide = unsafe { *query_wide.get_unchecked(dim) };
                    let new_off = O::saturating_dist(query_elem_wide, pivot_wide);
                    let old_off = unsafe { *off.get_unchecked(dim) };

                    let new_dist1 = D::dist1(new_off, O::zero());
                    let old_dist1 = D::dist1(old_off, O::zero());
                    let rd_far = O::saturating_add(rd - old_dist1, new_dist1);

                    if O::cmp(rd_far, best_dist) != std::cmp::Ordering::Greater {
                        stack.push(SS::StackContext::<O>::from_parts_with_restore_dim(
                            far_ctx.deferred_state(),
                            dim,
                            new_off,
                            rd_far,
                        ));
                        #[cfg(feature = "result_collection_stats")]
                        crate::result_collection_stats::record_query_stack_push();
                    }
                } else {
                    stem_strat.traverse(false);
                }

                dim = stem_strat.dim();
            }

            let leaf_idx = stem_strat.leaf_idx();
            debug_assert!(
                leaf_idx < self.leaf_count(),
                "arithmetic query resolved invalid leaf_idx={} leaf_count={}",
                leaf_idx,
                self.leaf_count()
            );

            process_leaf(leaf_idx, &query_wide, query_ctx);
        }
    }

    /// traverse to leaf
    #[inline(always)]
    fn traverse_to_leaf<O, D>(
        &self,
        query: &[A; K],
        query_wide: &[O; K],
        stem_strat: &mut SS,
        off: &mut [O; K],
        dim: &mut usize,
        rd: O,
        best_dist: O,
        stack: &mut SS::Stack<O>,
    ) -> Option<usize>
    where
        O: AxisUnified<Coord = O> + BacktrackBlock3 + BacktrackBlock4,
        D: KdTreeDistanceMetric<A, K, Output = O>
            + DistanceMetricSimdBlock3<A, K, O>
            + DistanceMetricSimdBlock4<A, K, O>,
        SS::Stack<O>: StackTrait<O, SS>,
        SS::StackContext<O>: ScalarStackContext<O, SS::DeferredState>,
    {
        loop {
            // Check if current stem points directly to a leaf
            // For Immutable trees, this should optimise away since resolve_terminal_stem_idx
            // will always return None
            if let Some(leaf_idx) = self.resolve_terminal_stem(stem_strat.stem_idx()) {
                return Some(leaf_idx);
            }

            // Delegate to stem strategy for traversal step
            // Default impl does level-by-level, block-based strategies do block-at-once
            let should_continue = stem_strat.backtracking_traverse_step::<A, O, D, K>(
                self.stems(),
                query,
                query_wide,
                off,
                dim,
                rd,
                self.max_stem_level(),
                best_dist,
                stack,
            );

            if !should_continue {
                break;
            }
        }

        // if !self.subtree_may_contain_leaf(
        //     stem_strat.stem_idx(),
        //     stem_strat.level(),
        //     stem_strat.leaf_idx(),
        // ) {
        //     tracing::warn!(
        //         stem_idx = stem_strat.stem_idx(),
        //         level = stem_strat.level(),
        //         leaf_idx_prefix = stem_strat.leaf_idx(),
        //         "traverse_to_leaf reached structurally invalid subtree; skipping leaf"
        //     );
        //     return None;
        // }

        Some(
            self.stem_leaf_resolution()
                .resolve_terminal_stem_idx(stem_strat.stem_idx(), stem_strat.leaf_idx()),
        )
    }

    /// Implementation of backtracking query with SIMD stack.
    /// Called by DonnellyMarkerSimd's backtracking_query_with_stack override.
    #[inline(always)]
    fn backtracking_query_with_block3_simd_stack_impl<QC, O, D>(
        &self,
        query_ctx: &mut QC,
        stack: &mut SS::Stack<O>,
        mut process_leaf: impl FnMut(usize, &[O; K], &mut QC),
    ) where
        QC: QueryContext<A, O, K>,
        O: AxisUnified<Coord = O>
            + SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4,
        D: KdTreeDistanceMetric<A, K, Output = O>
            + DistanceMetricSimdBlock3<A, K, O>
            + DistanceMetricSimdBlock4<A, K, O>,
        SS: StemStrategy
            + crate::stem_strategies::donnelly_2_blockmarker_simd::DeferredBlockTraversal,
        SS::StackContext<O>: crate::kd_tree::query_stack_simd::Block3ExactStackContext<O, SS, K>
            + crate::kd_tree::query_stack_simd::SimdIntervalStackContext<O, SS>,
    {
        use crate::kd_tree::query_stack_simd::{
            Block3ExactStackContext, Block3ExactStackContextState,
        };

        let stems_ptr = NonNull::new(self.stems().as_ptr() as *mut u8).unwrap();
        let stem_strat: SS = SS::new(stems_ptr);

        let query: [A; K] = *query_ctx.query();
        let mut query_wide: [O; K] = [O::zero(); K];
        for dim in 0..K {
            query_wide[dim] = D::widen_coord(query[dim]);
        }

        let mut off = [O::zero(); K];
        let mut lower = [O::min_value(); K];
        let mut upper = [O::max_value(); K];

        stack.push(
            <SS::StackContext<O> as Block3ExactStackContext<O, SS, K>>::new_single(stem_strat),
        );
        #[cfg(feature = "result_collection_stats")]
        crate::result_collection_stats::record_query_stack_push();

        while let Some(ctx) = stack.pop() {
            #[cfg(feature = "result_collection_stats")]
            crate::result_collection_stats::record_query_stack_pop();
            match <SS::StackContext<O> as Block3ExactStackContext<O, SS, K>>::into_block3_exact_state(ctx) {
                Block3ExactStackContextState::Single {
                    stem_strat: mut ss,
                    dim: dim_val,
                    lower_bound,
                    upper_bound,
                    old_off,
                    rd,
                } => {
                    #[cfg(feature = "test_utils")]
                    crate::test_utils::exact_query_stats::record_simd_single_pop();

                    let restore_dim = dim_val;
                    let mut dim = ss.dim();
                    tracing::trace!(
                        %restore_dim,
                        resumed_dim = %dim,
                        %old_off,
                        %rd,
                        ?off,
                        "Popped single context"
                    );

                    let max_dist = query_ctx.max_dist();
                    let rd_vs_max = O::cmp(rd, max_dist);
                    let should_prune = rd_vs_max == std::cmp::Ordering::Greater
                        || (query_ctx.prune_on_equal_max_dist()
                            && rd_vs_max == std::cmp::Ordering::Equal);
                    if should_prune {
                        #[cfg(feature = "result_collection_stats")]
                        crate::result_collection_stats::record_query_prune();
                        tracing::trace!(%rd, %max_dist, "Prune check: PRUNE");
                        continue;
                    }

                    unsafe {
                        *off.get_unchecked_mut(restore_dim) = old_off;
                        *lower.get_unchecked_mut(restore_dim) = lower_bound;
                        *upper.get_unchecked_mut(restore_dim) = upper_bound;
                    }

                    let best_dist = query_ctx.max_dist();
                    if let Some(leaf_idx) = self.traverse_to_leaf_simd::<O, D>(
                        &query,
                        &query_wide,
                        &mut ss,
                        &mut lower,
                        &mut upper,
                        &mut off,
                        &mut dim,
                        rd,
                        best_dist,
                        stack,
                    ) {
                        tracing::trace!(%leaf_idx, "processing leaf");
                        process_leaf(leaf_idx, &query_wide, query_ctx);
                    }
                }
                Block3ExactStackContextState::Block3Pending {
                    base,
                    pending_mask,
                    rd,
                    lower: restored_lower,
                    upper: restored_upper,
                } => {
                    #[cfg(feature = "test_utils")]
                    crate::test_utils::exact_query_stats::record_block3_pending_pop(pending_mask);

                    let dim_val = base.dim();
                    lower = restored_lower;
                    upper = restored_upper;
                    off = rebuild_interval_offs(&query_wide, &lower, &upper);
                    let old_off = off[dim_val];
                    let lower_bound = lower[dim_val];
                    let upper_bound = upper[dim_val];

                    tracing::trace!(%dim_val, %old_off, %rd, %pending_mask, "Popped Block3 pending context");

                    #[cfg(feature = "test_utils")]
                    let trace_enabled =
                        crate::test_utils::exact_query_trace::enabled()
                            && std::any::type_name::<O>() == "f64";

                    #[cfg(not(feature = "test_utils"))]
                    let trace_enabled = false;

                    let best_dist = query_ctx.max_dist();
                    let query_wide_val = query_wide[dim_val];
                    let (
                        candidate_mask,
                        selection,
                        trace_pending_arrays,
                    ) = if trace_enabled {
                        let mut rd_values = [O::zero(); 8];
                        let mut new_off_values = [O::zero(); 8];
                        let mut lower_bounds = [O::zero(); 8];
                        let mut upper_bounds = [O::zero(); 8];
                        let candidate_mask = base
                            .fill_block3_pending_values::<A, O, D, K>(
                                self.stems(),
                                query_wide_val,
                                lower_bound,
                                upper_bound,
                                old_off,
                                rd,
                                best_dist,
                                &mut new_off_values,
                                &mut rd_values,
                                &mut lower_bounds,
                                &mut upper_bounds,
                            )
                            & pending_mask;
                        let selection = select_block3_pending_child(
                            &rd_values,
                            &new_off_values,
                            candidate_mask,
                        )
                        .map(|selection| {
                            (
                                selection.child_idx,
                                selection.remaining_mask,
                                selection.child_rd,
                                selection.child_off,
                                lower_bounds[selection.child_idx as usize],
                                upper_bounds[selection.child_idx as usize],
                            )
                        });
                        (
                            candidate_mask,
                            selection,
                            Some((rd_values, new_off_values, lower_bounds, upper_bounds)),
                        )
                    } else {
                        let candidate_mask = base
                            .backtrack_block3_pending_mask::<A, O, D, K>(
                                self.stems(),
                                query_wide_val,
                                lower_bound,
                                upper_bound,
                                old_off,
                                rd,
                                best_dist,
                            )
                            & pending_mask;

                        let selection = if candidate_mask == 0 {
                            None
                        } else if candidate_mask.is_power_of_two() {
                            let child_idx = candidate_mask.trailing_zeros() as u8;
                            let (child_rd, child_off, child_lower_bound, child_upper_bound) = base
                                .selected_block3_pending_child_state::<A, O, D>(
                                    self.stems(),
                                    child_idx,
                                    query_wide_val,
                                    lower_bound,
                                    upper_bound,
                                    old_off,
                                    rd,
                                );
                            Some((
                                child_idx,
                                candidate_mask & !(1u8 << child_idx),
                                child_rd,
                                child_off,
                                child_lower_bound,
                                child_upper_bound,
                            ))
                        } else {
                            let mut rd_values = [O::zero(); 8];
                            let mut new_off_values = [O::zero(); 8];
                            let mut lower_bounds = [O::zero(); 8];
                            let mut upper_bounds = [O::zero(); 8];
                            let candidate_mask = base
                                .fill_block3_pending_values::<A, O, D, K>(
                                    self.stems(),
                                    query_wide_val,
                                    lower_bound,
                                    upper_bound,
                                    old_off,
                                    rd,
                                    best_dist,
                                    &mut new_off_values,
                                    &mut rd_values,
                                    &mut lower_bounds,
                                    &mut upper_bounds,
                                )
                                & candidate_mask;
                            let selection = select_block3_pending_child(
                                &rd_values,
                                &new_off_values,
                                candidate_mask,
                            )
                            .map(|selection| {
                                let child_idx = selection.child_idx as usize;
                                (
                                    selection.child_idx,
                                    selection.remaining_mask,
                                    selection.child_rd,
                                    selection.child_off,
                                    unsafe { *lower_bounds.get_unchecked(child_idx) },
                                    unsafe { *upper_bounds.get_unchecked(child_idx) },
                                )
                            });
                            selection
                        };
                        (candidate_mask, selection, None)
                    };

                    #[cfg(not(feature = "test_utils"))]
                    let _ = (candidate_mask, &trace_pending_arrays);

                    #[cfg(feature = "test_utils")]
                    crate::test_utils::exact_query_stats::record_block3_candidate_mask(
                        candidate_mask,
                    );

                    let Some((
                        selected_child_idx,
                        remaining_mask,
                        selected_child_rd,
                        selected_child_off,
                        selected_lower_bound,
                        selected_upper_bound,
                    )) = selection
                    else {
                        #[cfg(feature = "result_collection_stats")]
                        crate::result_collection_stats::record_query_prune();
                        tracing::trace!("All Block3 children pruned");
                        continue;
                    };

                    #[cfg(feature = "test_utils")]
                    {
                        if let Some((rd_values, new_off_values, lower_bounds, upper_bounds)) =
                            trace_pending_arrays
                        {
                            let old_off_f = unsafe { *(&old_off as *const O as *const f64) };
                            let rd_f = unsafe { *(&rd as *const O as *const f64) };
                            let lower_bound_f =
                                unsafe { *(&lower_bound as *const O as *const f64) };
                            let upper_bound_f =
                                unsafe { *(&upper_bound as *const O as *const f64) };
                            let child_off_f =
                                unsafe { *(&selected_child_off as *const O as *const f64) };
                            let child_rd_f =
                                unsafe { *(&selected_child_rd as *const O as *const f64) };
                            let new_off_values_f = unsafe {
                                *(&new_off_values as *const [O; 8] as *const [f64; 8])
                            };
                            let rd_values_f =
                                unsafe { *(&rd_values as *const [O; 8] as *const [f64; 8]) };
                            let lower_bounds_f = unsafe {
                                *(&lower_bounds as *const [O; 8] as *const [f64; 8])
                            };
                            let upper_bounds_f = unsafe {
                                *(&upper_bounds as *const [O; 8] as *const [f64; 8])
                            };
                            crate::test_utils::exact_query_trace::push(
                                crate::test_utils::exact_query_trace::ExactQueryTraceEvent::Block3PendingSelection {
                                    stem_idx: base.stem_idx(),
                                    level: base.level(),
                                    dim: dim_val,
                                    pending_mask,
                                    candidate_mask,
                                    selected_child_idx,
                                    child_off: child_off_f,
                                    child_rd: child_rd_f,
                                    parent_lower_bound: lower_bound_f,
                                    parent_upper_bound: upper_bound_f,
                                    old_off: old_off_f,
                                    rd: rd_f,
                                    new_off_values: new_off_values_f,
                                    rd_values: rd_values_f,
                                    lower_bounds: lower_bounds_f,
                                    upper_bounds: upper_bounds_f,
                                },
                            );
                        }
                    }

                    if remaining_mask != 0 {
                        stack.push(
                            <SS::StackContext<O> as Block3ExactStackContext<O, SS, K>>::new_block3_pending_from_state(
                                base,
                                remaining_mask,
                                rd,
                                &lower,
                                &upper,
                            ),
                        );
                        #[cfg(feature = "result_collection_stats")]
                        crate::result_collection_stats::record_query_stack_push();
                    }

                    let mut ss = base.block_child(selected_child_idx);
                    let mut dim = ss.dim();

                    unsafe {
                        *off.get_unchecked_mut(dim_val) = selected_child_off;
                        *lower.get_unchecked_mut(dim_val) = selected_lower_bound;
                        *upper.get_unchecked_mut(dim_val) = selected_upper_bound;
                    }

                    if let Some(leaf_idx) = self.traverse_to_leaf_simd::<O, D>(
                        &query,
                        &query_wide,
                        &mut ss,
                        &mut lower,
                        &mut upper,
                        &mut off,
                        &mut dim,
                        selected_child_rd,
                        best_dist,
                        stack,
                    ) {
                        tracing::trace!(%leaf_idx, "processing leaf");
                        process_leaf(leaf_idx, &query_wide, query_ctx);
                    }
                }
            }
        }
    }

    /// Implementation of backtracking query with SIMD stack.
    /// Called by DonnellyMarkerSimd's backtracking_query_with_stack override.
    #[inline(always)]
    fn backtracking_query_with_simd_stack_impl<QC, O, D>(
        &self,
        query_ctx: &mut QC,
        stack: &mut SS::Stack<O>,
        mut process_leaf: impl FnMut(usize, &[O; K], &mut QC),
    ) where
        QC: QueryContext<A, O, K>,
        O: AxisUnified<Coord = O>
            + SimdPrune
            + SimdSelectBestChildBlock3
            + BacktrackBlock3
            + BacktrackBlock4,
        D: KdTreeDistanceMetric<A, K, Output = O>
            + DistanceMetricSimdBlock3<A, K, O>
            + DistanceMetricSimdBlock4<A, K, O>,
        SS: StemStrategy<
                StackContext<O> = crate::kd_tree::query_stack_simd::SimdQueryStackContext<O, SS>,
            > + crate::stem_strategies::donnelly_2_blockmarker_simd::DeferredBlockTraversal,
    {
        use crate::kd_tree::query_stack_simd::SimdQueryStackContext;

        let stems_ptr = NonNull::new(self.stems().as_ptr() as *mut u8).unwrap();
        let stem_strat: SS = SS::new(stems_ptr);

        let query: [A; K] = *query_ctx.query();
        let mut query_wide: [O; K] = [O::zero(); K];
        for dim in 0..K {
            query_wide[dim] = D::widen_coord(query[dim]);
        }

        let mut off = [O::zero(); K];
        let mut lower = [O::min_value(); K];
        let mut upper = [O::max_value(); K];

        stack.push(SimdQueryStackContext::new_single(stem_strat));
        #[cfg(feature = "result_collection_stats")]
        crate::result_collection_stats::record_query_stack_push();

        // Backtracking loop
        while let Some(ctx) = stack.pop() {
            #[cfg(feature = "result_collection_stats")]
            crate::result_collection_stats::record_query_stack_pop();
            match ctx {
                SimdQueryStackContext::Single {
                    stem_strat: mut ss,
                    dim: dim_val,
                    lower_bound,
                    upper_bound,
                    old_off,
                    rd,
                } => {
                    // Single entry - standard scalar processing
                    let restore_dim = dim_val;
                    let mut dim = ss.dim();
                    tracing::trace!(
                        %restore_dim,
                        resumed_dim = %dim,
                        %old_off,
                        %rd,
                        ?off,
                        "Popped single context"
                    );

                    let max_dist = query_ctx.max_dist();
                    let rd_vs_max = O::cmp(rd, max_dist);
                    // TOOO: investigate into whether prune_on_equal_max_dist can be removed
                    let should_prune = rd_vs_max == std::cmp::Ordering::Greater
                        || (query_ctx.prune_on_equal_max_dist()
                            && rd_vs_max == std::cmp::Ordering::Equal);
                    if should_prune {
                        #[cfg(feature = "result_collection_stats")]
                        crate::result_collection_stats::record_query_prune();
                        tracing::trace!(%rd, %max_dist, "Prune check: PRUNE");
                        continue;
                    }
                    tracing::trace!(%rd, %max_dist, "SIMD Prune check: VISIT");

                    tracing::trace!("Restoring interval state for dim {}", restore_dim);
                    unsafe {
                        *off.get_unchecked_mut(restore_dim) = old_off;
                        *lower.get_unchecked_mut(restore_dim) = lower_bound;
                        *upper.get_unchecked_mut(restore_dim) = upper_bound;
                    }

                    let best_dist = query_ctx.max_dist();
                    if let Some(leaf_idx) = self.traverse_to_leaf_simd::<O, D>(
                        &query,
                        &query_wide,
                        &mut ss,
                        &mut lower,
                        &mut upper,
                        &mut off,
                        &mut dim,
                        rd,
                        best_dist,
                        stack,
                    ) {
                        tracing::trace!(%leaf_idx, "processing leaf");
                        process_leaf(leaf_idx, &query_wide, query_ctx);
                    }
                }
                SimdQueryStackContext::Block3Pending {
                    base,
                    rd_values,
                    new_off_values,
                    lower_bounds,
                    upper_bounds,
                    pending_mask,
                    dim: dim_val,
                    old_off,
                    lower_bound,
                    upper_bound,
                } => {
                    tracing::trace!(
                        %dim_val,
                        %old_off,
                        ?rd_values,
                        %pending_mask,
                        "Popped Block3 pending context"
                    );
                    unsafe {
                        *off.get_unchecked_mut(dim_val) = old_off;
                        *lower.get_unchecked_mut(dim_val) = lower_bound;
                        *upper.get_unchecked_mut(dim_val) = upper_bound;
                    }

                    let Some(selection) = select_block3_pending_child(
                        &rd_values,
                        &new_off_values,
                        simd::simd_prune_block(&rd_values, query_ctx.max_dist(), pending_mask),
                    ) else {
                        #[cfg(feature = "result_collection_stats")]
                        crate::result_collection_stats::record_query_prune();
                        tracing::trace!("All Block3 children pruned");
                        continue;
                    };

                    if selection.remaining_mask != 0 {
                        stack.push(SimdQueryStackContext::new_block3_pending(
                            base,
                            rd_values,
                            new_off_values,
                            lower_bounds,
                            upper_bounds,
                            selection.remaining_mask,
                            dim_val,
                            old_off,
                            lower_bound,
                            upper_bound,
                        ));
                        #[cfg(feature = "result_collection_stats")]
                        crate::result_collection_stats::record_query_stack_push();
                    }

                    let mut ss = base.block_child(selection.child_idx);
                    let mut dim = ss.dim();

                    unsafe {
                        *off.get_unchecked_mut(dim_val) = selection.child_off;
                        *lower.get_unchecked_mut(dim_val) =
                            lower_bounds[selection.child_idx as usize];
                        *upper.get_unchecked_mut(dim_val) =
                            upper_bounds[selection.child_idx as usize];
                    }

                    let best_dist = query_ctx.max_dist();
                    if let Some(leaf_idx) = self.traverse_to_leaf_simd::<O, D>(
                        &query,
                        &query_wide,
                        &mut ss,
                        &mut lower,
                        &mut upper,
                        &mut off,
                        &mut dim,
                        selection.child_rd,
                        best_dist,
                        stack,
                    ) {
                        tracing::trace!(%leaf_idx, "processing leaf");
                        process_leaf(leaf_idx, &query_wide, query_ctx);
                    }
                }
                SimdQueryStackContext::Block {
                    siblings,
                    rd_values,
                    new_off_values,
                    lower_bounds,
                    upper_bounds,
                    sibling_mask,
                    dim: dim_val,
                    old_off,
                    lower_bound,
                    upper_bound,
                } => {
                    tracing::trace!(%dim_val, %old_off, ?rd_values, %sibling_mask, "Popped block context");
                    // Restore the parent split-dimension offset captured for this block context.
                    unsafe {
                        *off.get_unchecked_mut(dim_val) = old_off;
                        *lower.get_unchecked_mut(dim_val) = lower_bound;
                        *upper.get_unchecked_mut(dim_val) = upper_bound;
                    }

                    // SIMD pruning: check which siblings pass the backtrack test
                    let max_dist = query_ctx.max_dist();
                    let surviving_mask =
                        simd::simd_prune_block::<O>(&rd_values, max_dist, sibling_mask);

                    if surviving_mask == 0 {
                        #[cfg(feature = "result_collection_stats")]
                        crate::result_collection_stats::record_query_prune();
                        tracing::trace!("All siblings pruned");
                        continue;
                    }

                    tracing::trace!(
                        surviving_mask = format!("{:08b}", surviving_mask),
                        "Some siblings survive"
                    );

                    // Save the current off state before processing siblings
                    let saved_off = off;
                    let saved_lower = lower;
                    let saved_upper = upper;

                    // Process each surviving sibling
                    for sibling_idx in 0..8 {
                        if surviving_mask & (1 << sibling_idx) != 0 {
                            let mut ss = siblings[sibling_idx];
                            let rd = rd_values[sibling_idx];
                            let new_off = new_off_values[sibling_idx];
                            // if !self.subtree_may_contain_leaf(
                            //     ss.stem_idx(),
                            //     ss.level(),
                            //     ss.leaf_idx(),
                            // ) {
                            //     tracing::warn!(
                            //         sibling_idx,
                            //         stem_idx = ss.stem_idx(),
                            //         leaf_idx_prefix = ss.leaf_idx(),
                            //         level = ss.level(),
                            //         "SIMD block sibling points to structurally invalid leaf-prefix subtree; skipping"
                            //     );
                            //     continue;
                            // }
                            let mut dim = ss.dim();

                            // Restore off array to saved state, then update the split dimension
                            // Use the per-sibling new_off value (e.g., interval distance)
                            off = saved_off;
                            lower = saved_lower;
                            upper = saved_upper;
                            tracing::trace!(
                                "Restoring off[{}]. was {}, now {} (interval dist for sibling {}). Parent dim was {}, sibling dim is {}",
                                dim_val,
                                unsafe { *off.get_unchecked(dim_val) },
                                new_off,
                                sibling_idx,
                                dim_val,
                                dim
                            );
                            // The interval distance was computed for the parent block's split dim.
                            unsafe {
                                *off.get_unchecked_mut(dim_val) = new_off;
                                *lower.get_unchecked_mut(dim_val) = lower_bounds[sibling_idx];
                                *upper.get_unchecked_mut(dim_val) = upper_bounds[sibling_idx];
                            }

                            let best_dist = query_ctx.max_dist();
                            if let Some(leaf_idx) = self.traverse_to_leaf_simd::<O, D>(
                                &query,
                                &query_wide,
                                &mut ss,
                                &mut lower,
                                &mut upper,
                                &mut off,
                                &mut dim,
                                rd,
                                best_dist,
                                stack,
                            ) {
                                tracing::trace!(%leaf_idx, "processing leaf");
                                process_leaf(leaf_idx, &query_wide, query_ctx);
                            }
                        }
                    }
                }
                SimdQueryStackContext::DeferredBlock {
                    base,
                    child_base,
                    rd_values,
                    new_off_values,
                    lower_bounds,
                    upper_bounds,
                    sibling_mask,
                    dim: dim_val,
                    old_off,
                    lower_bound,
                    upper_bound,
                } => {
                    tracing::trace!(
                        %dim_val,
                        %old_off,
                        ?rd_values,
                        %sibling_mask,
                        %child_base,
                        "Popped deferred block context"
                    );
                    unsafe {
                        *off.get_unchecked_mut(dim_val) = old_off;
                        *lower.get_unchecked_mut(dim_val) = lower_bound;
                        *upper.get_unchecked_mut(dim_val) = upper_bound;
                    }

                    let max_dist = query_ctx.max_dist();
                    let surviving_mask =
                        simd::simd_prune_block::<O>(&rd_values, max_dist, sibling_mask);

                    if surviving_mask == 0 {
                        #[cfg(feature = "result_collection_stats")]
                        crate::result_collection_stats::record_query_prune();
                        tracing::trace!("All deferred block siblings pruned");
                        continue;
                    }

                    let saved_off = off;
                    let saved_lower = lower;
                    let saved_upper = upper;

                    for sibling_idx in 0..8 {
                        if surviving_mask & (1 << sibling_idx) != 0 {
                            let mut ss = base.block_child(child_base + sibling_idx as u8);
                            let rd = rd_values[sibling_idx];
                            let new_off = new_off_values[sibling_idx];
                            let mut dim = ss.dim();

                            off = saved_off;
                            lower = saved_lower;
                            upper = saved_upper;
                            unsafe {
                                *off.get_unchecked_mut(dim_val) = new_off;
                                *lower.get_unchecked_mut(dim_val) = lower_bounds[sibling_idx];
                                *upper.get_unchecked_mut(dim_val) = upper_bounds[sibling_idx];
                            }

                            let best_dist = query_ctx.max_dist();
                            if let Some(leaf_idx) = self.traverse_to_leaf_simd::<O, D>(
                                &query,
                                &query_wide,
                                &mut ss,
                                &mut lower,
                                &mut upper,
                                &mut off,
                                &mut dim,
                                rd,
                                best_dist,
                                stack,
                            ) {
                                tracing::trace!(%leaf_idx, "processing leaf");
                                process_leaf(leaf_idx, &query_wide, query_ctx);
                            }
                        }
                    }
                }
            }
        }
    }

    /// traverse to leaf with SIMD stack
    #[inline(always)]
    fn traverse_to_leaf_simd<O, D>(
        &self,
        query: &[A; K],
        query_wide: &[O; K],
        stem_strat: &mut SS,
        lower: &mut [O; K],
        upper: &mut [O; K],
        off: &mut [O; K],
        dim: &mut usize,
        rd: O,
        best_dist: O,
        stack: &mut SS::Stack<O>,
    ) -> Option<usize>
    where
        O: AxisUnified<Coord = O> + SimdSelectBestChildBlock3 + BacktrackBlock3 + BacktrackBlock4,
        D: KdTreeDistanceMetric<A, K, Output = O>
            + DistanceMetricSimdBlock3<A, K, O>
            + DistanceMetricSimdBlock4<A, K, O>,
        SS: StemStrategy
            + crate::stem_strategies::donnelly_2_blockmarker_simd::DeferredBlockTraversal,
        SS::StackContext<O>: crate::kd_tree::query_stack_simd::SimdIntervalStackContext<O, SS>,
    {
        let use_scalar_step = !self.stem_leaf_resolution().uses_arithmetic()
            && SS::BLOCK_SIZE != 3
            && !force_mapped_simd_block_step();

        loop {
            let stem_idx = stem_strat.stem_idx();
            // Check if current stem points directly to a leaf
            if let Some(leaf_idx) = self.resolve_terminal_stem(stem_idx) {
                return Some(leaf_idx);
            }
            let stem_oob = stem_idx >= self.stems().len();

            let should_continue = if use_scalar_step {
                if stem_strat.level() > self.max_stem_level() {
                    false
                } else {
                    let dim_val = *dim;
                    let pivot = if stem_oob {
                        A::max_value()
                    } else {
                        unsafe { *self.stems().get_unchecked(stem_idx) }
                    };

                    let old_off = unsafe { *off.get_unchecked(dim_val) };

                    if pivot < A::max_value() {
                        let query_elem = unsafe { *query.get_unchecked(dim_val) };
                        let is_right_child = query_elem >= pivot;
                        let far_ctx = stem_strat.branch_relative(is_right_child);

                        let pivot_wide: O = D::widen_coord(pivot);
                        let query_elem_wide = unsafe { *query_wide.get_unchecked(dim_val) };
                        let new_off = O::saturating_dist(query_elem_wide, pivot_wide);

                        let new_dist1 = D::dist1(new_off, O::zero());
                        let old_dist1 = D::dist1(old_off, O::zero());
                        let rd_far = O::saturating_add(rd - old_dist1, new_dist1);

                        let old_lower = unsafe { *lower.get_unchecked(dim_val) };
                        let old_upper = unsafe { *upper.get_unchecked(dim_val) };
                        let near_lower;
                        let near_upper;
                        let far_lower;
                        let far_upper;

                        if is_right_child {
                            near_lower = O::max(old_lower, pivot_wide);
                            near_upper = old_upper;
                            far_lower = old_lower;
                            far_upper = if O::cmp(old_upper, pivot_wide) == std::cmp::Ordering::Less
                            {
                                old_upper
                            } else {
                                pivot_wide
                            };
                        } else {
                            near_lower = old_lower;
                            near_upper =
                                if O::cmp(old_upper, pivot_wide) == std::cmp::Ordering::Less {
                                    old_upper
                                } else {
                                    pivot_wide
                                };
                            far_lower = O::max(old_lower, pivot_wide);
                            far_upper = old_upper;
                        }

                        let near_off =
                            crate::stem_strategies::donnelly_2_blockmarker_simd::interval_distance_1d(
                                query_elem_wide,
                                near_lower,
                                near_upper,
                            );

                        if O::cmp(rd_far, best_dist) != std::cmp::Ordering::Greater {
                            stack.push(<SS::StackContext<O> as crate::kd_tree::query_stack_simd::SimdIntervalStackContext<O, SS>>::new_single_with_bounds(
                                far_ctx,
                                dim_val,
                                far_lower,
                                far_upper,
                                new_off,
                                rd_far,
                            ));
                        }

                        unsafe {
                            *lower.get_unchecked_mut(dim_val) = near_lower;
                            *upper.get_unchecked_mut(dim_val) = near_upper;
                            *off.get_unchecked_mut(dim_val) = near_off;
                        }
                    } else {
                        // +Inf pivots can still encode structural branches in left-aligned trees.
                        // Scalar traversal treats these as structural padding and just descends
                        // left without enqueuing a synthetic sibling branch.
                        stem_strat.traverse(false);
                    }

                    *dim = stem_strat.dim();
                    true
                }
            } else {
                stem_strat.backtracking_traverse_step_with_bounds::<A, O, D, K>(
                    self.stems(),
                    query,
                    query_wide,
                    lower,
                    upper,
                    off,
                    dim,
                    rd,
                    self.max_stem_level(),
                    best_dist,
                    stack,
                )
            };

            tracing::trace!(
                stem_idx = %stem_strat.stem_idx(),
                level = %stem_strat.level(),
                dim = %stem_strat.dim(),
                "Descended one block"
            );

            if !should_continue {
                break;
            }
        }

        // if !self.subtree_may_contain_leaf(
        //     stem_strat.stem_idx(),
        //     stem_strat.level(),
        //     stem_strat.leaf_idx(),
        // ) {
        //     tracing::warn!(
        //         stem_idx = stem_strat.stem_idx(),
        //         level = stem_strat.level(),
        //         leaf_idx_prefix = stem_strat.leaf_idx(),
        //         "traverse_to_leaf_simd reached structurally invalid subtree; skipping leaf"
        //     );
        //     return None;
        // }

        Some(
            self.stem_leaf_resolution()
                .resolve_terminal_stem_idx(stem_strat.stem_idx(), stem_strat.leaf_idx()),
        )
    }
}
