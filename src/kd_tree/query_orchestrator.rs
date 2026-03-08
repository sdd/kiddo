use crate::kd_tree::leaf_view::LeafView;
use crate::kd_tree::query_stack::{
    scalar_ctx_from_parts, scalar_ctx_into_parts, QueryStack, StackTrait,
};
use crate::kd_tree::traits::QueryContext;
use crate::kd_tree::KdTree;
use crate::stem_strategies::{
    donnelly_2_blockmarker_simd::{BacktrackBlock3, BacktrackBlock4},
    DistanceMetricSimdBlock3, DistanceMetricSimdBlock4, SimdPrune,
};
use crate::traits_unified_2::{
    AxisUnified, Basics, DistanceMetricUnified, LeafStrategy, Mutability,
};
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

impl<A, T, SS, LS, const K: usize, const B: usize> KdTree<A, T, SS, LS, K, B>
where
    A: AxisUnified<Coord = A>,
    T: Basics + Copy + Default + PartialOrd + PartialEq,
    SS: StemStrategy,
    LS: LeafStrategy<A, T, SS, K, B>,
{
    #[inline]
    pub(crate) fn get_leaf_idx(&self, query: &[A; K]) -> usize {
        LS::Mutability::get_leaf_idx(self, query)
    }

    /// Non-backtracking query
    #[inline]
    pub(crate) fn straight_query<QC, O>(
        &self,
        query_ctx: QC,
        mut process_leaf: impl FnMut(&LeafView<A, T, K, B>),
    ) where
        QC: QueryContext<A, O, K>,
    {
        let leaf_idx = LS::Mutability::get_leaf_idx(self, query_ctx.query());

        tracing::trace!(%leaf_idx, "processing leaf");
        let leaf_view = self.leaves.leaf_view(leaf_idx);
        process_leaf(&leaf_view);
    }

    /// Get the leaf index for a query (unmapped leaves)
    #[inline]
    pub(crate) fn get_leaf_idx_unmapped(&self, query: &[A; K]) -> usize {
        SS::get_leaf_idx(&self.stems, query, self.max_stem_level)
    }

    /// Get the leaf index for a query (mapped leaves)
    #[inline(always)]
    pub(crate) fn get_leaf_idx_mapped(&self, query: &[A; K]) -> usize {
        let stems_ptr = NonNull::new(self.stems.as_ptr() as *mut u8).unwrap();
        let mut stem_strat: SS = SS::new(stems_ptr);

        while stem_strat.level() <= self.max_stem_level {
            if let Some(leaf_idx) = self.resolve_terminal_stem(stem_strat.stem_idx()) {
                return leaf_idx;
            }

            let pivot = unsafe { self.stems.get_unchecked(stem_strat.stem_idx()) };
            let is_right_child: bool = *unsafe { query.get_unchecked(stem_strat.dim()) } >= *pivot;
            stem_strat.traverse(is_right_child);
        }

        self.stem_leaf_resolution
            .resolve_terminal_stem_idx(stem_strat.stem_idx(), stem_strat.leaf_idx())
    }

    // TODO: don't like this structure
    /// Check if a stem points directly to a leaf
    #[inline(always)]
    pub(crate) fn resolve_terminal_stem(&self, stem_idx: usize) -> Option<usize> {
        if self.stem_leaf_resolution.is_terminal_stem_idx(stem_idx) {
            Some(
                self.stem_leaf_resolution
                    .resolve_terminal_stem_idx(stem_idx, 0),
            )
        } else {
            None
        }
    }

    #[inline(always)]
    fn subtree_may_contain_leaf(
        &self,
        stem_idx: usize,
        level: i32,
        leaf_idx_prefix: usize,
    ) -> bool {
        match &self.stem_leaf_resolution {
            crate::kd_tree::StemLeafResolution::Arithmetic {
                stems_depth,
                leaf_count,
            }
            | crate::kd_tree::StemLeafResolution::Pristine {
                stems_depth,
                leaf_count,
            } => {
                if *leaf_count == 0 {
                    return false;
                }

                let consumed_levels = level.max(0) as usize;
                if consumed_levels >= *stems_depth {
                    return leaf_idx_prefix < *leaf_count;
                }

                let remaining = (*stems_depth - consumed_levels) as u32;
                let min_leaf_idx = leaf_idx_prefix.checked_shl(remaining).unwrap_or(usize::MAX);
                min_leaf_idx < *leaf_count
            }
            crate::kd_tree::StemLeafResolution::Mapped { .. } => {
                stem_idx < self.stems.len()
                    || self.stem_leaf_resolution.is_terminal_stem_idx(stem_idx)
            }
        }
    }

    /// Backtracking query
    #[inline(always)]
    pub(crate) fn backtracking_query<QC, O, D>(
        &self,
        query_ctx: &mut QC,
        process_leaf: impl FnMut(&LeafView<A, T, K, B>, &[O; K], &mut QC),
    ) where
        QC: QueryContext<A, O, K>,
        O: AxisUnified<Coord = O> + SimdPrune + BacktrackBlock3 + BacktrackBlock4,
        D: DistanceMetricUnified<A, K, Output = O>
            + DistanceMetricSimdBlock3<A, K, O>
            + DistanceMetricSimdBlock4<A, K, O>,
        SS::Stack<O>: StackTrait<O, SS> + Default + 'static,
    {
        with_tls_query_stack::<SS::Stack<O>, _>(|stack| {
            stack.clear();
            self.backtracking_query_with_stack::<QC, O, D>(query_ctx, stack, process_leaf);
        });
    }

    /// Backtracking query with explicit stack
    #[inline(always)]
    pub(crate) fn backtracking_query_with_stack<QC, O, D>(
        &self,
        query_ctx: &mut QC,
        stack: &mut SS::Stack<O>,
        process_leaf: impl FnMut(&LeafView<A, T, K, B>, &[O; K], &mut QC),
    ) where
        QC: QueryContext<A, O, K>,
        O: AxisUnified<Coord = O> + SimdPrune + BacktrackBlock3 + BacktrackBlock4,
        D: DistanceMetricUnified<A, K, Output = O>
            + DistanceMetricSimdBlock3<A, K, O>
            + DistanceMetricSimdBlock4<A, K, O>,
        SS::Stack<O>: StackTrait<O, SS>,
    {
        SS::backtracking_query_with_stack::<A, T, O, D, QC, LS, K, B>(
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
    pub(crate) fn backtracking_query_with_stack_impl<QC, O, D>(
        &self,
        query_ctx: &mut QC,
        stack: &mut QueryStack<O, SS>,
        mut process_leaf: impl FnMut(&LeafView<A, T, K, B>, &[O; K], &mut QC),
    ) where
        QC: QueryContext<A, O, K>,
        O: AxisUnified<Coord = O> + BacktrackBlock3 + BacktrackBlock4,
        D: DistanceMetricUnified<A, K, Output = O>
            + DistanceMetricSimdBlock3<A, K, O>
            + DistanceMetricSimdBlock4<A, K, O>,
    {
        let stems_ptr = NonNull::new(self.stems.as_ptr() as *mut u8).unwrap();
        let mut stem_strat: SS = SS::new(stems_ptr);

        let query: [A; K] = *query_ctx.query();
        let mut query_wide: [O; K] = [O::zero(); K];
        for dim in 0..K {
            query_wide[dim] = D::widen_coord(query[dim]);
        }

        let mut off = [O::zero(); K];
        stack.push(scalar_ctx_from_parts::<O, SS>(
            stem_strat.deferred_state(),
            O::zero(),
            O::zero(),
        ));

        while let Some(stack_ctx) = stack.pop() {
            let (stem_state, old_off, rd) = scalar_ctx_into_parts::<O, SS>(stack_ctx);
            stem_strat.rehydrate_deferred_state(stem_state);
            let mut dim = stem_strat.dim();
            tracing::trace!(%dim, %old_off, %rd, ?off, "Popped stack context");

            let max_dist = query_ctx.max_dist();
            let rd_vs_max = O::cmp(rd, max_dist);

            // TOOO: investigate into whether prune_on_equal_max_dist can be removed
            let should_prune = rd_vs_max == std::cmp::Ordering::Greater
                || (query_ctx.prune_on_equal_max_dist() && rd_vs_max == std::cmp::Ordering::Equal);
            if should_prune {
                tracing::trace!(%rd, %max_dist, "SCALAR Prune check: PRUNE");
                continue;
            }
            tracing::trace!(%rd, %max_dist, "SCALAR Prune check: VISIT");

            tracing::trace!(
                "Restoring off[{}]. was {}, now {}",
                dim,
                unsafe { *off.get_unchecked(dim) },
                old_off
            );
            unsafe { *off.get_unchecked_mut(dim) = old_off };

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
                let leaf_view = self.leaves.leaf_view(leaf_idx);
                process_leaf(&leaf_view, &query_wide, query_ctx);
            }
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
        stack: &mut QueryStack<O, SS>,
    ) -> Option<usize>
    where
        O: AxisUnified<Coord = O> + BacktrackBlock3 + BacktrackBlock4,
        D: DistanceMetricUnified<A, K, Output = O>
            + DistanceMetricSimdBlock3<A, K, O>
            + DistanceMetricSimdBlock4<A, K, O>,
    {
        loop {
            // Check if current stem points directly to a leaf
            // For Immutable trees, this should optimise away since resolve_terminal_stem_idx
            // will always return None
            if let Some(leaf_idx) =
                LS::Mutability::resolve_terminal_stem_idx(self, stem_strat.stem_idx())
            {
                return Some(leaf_idx);
            }

            // Delegate to stem strategy for traversal step
            // Default impl does level-by-level, block-based strategies do block-at-once
            // SAFETY: Cast concrete QueryStack to associated type SS::Stack for trait call
            let stack_ref = unsafe { &mut *(stack as *mut QueryStack<O, SS> as *mut SS::Stack<O>) };
            let should_continue = stem_strat.backtracking_traverse_step::<A, O, D, K>(
                &self.stems,
                query,
                query_wide,
                off,
                dim,
                rd,
                self.max_stem_level,
                best_dist,
                stack_ref,
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
            self.stem_leaf_resolution
                .resolve_terminal_stem_idx(stem_strat.stem_idx(), stem_strat.leaf_idx()),
        )
    }

    /// Implementation of backtracking query with SIMD stack.
    /// Called by DonnellyMarkerSimd's backtracking_query_with_stack override.
    #[inline(always)]
    pub(crate) fn backtracking_query_with_simd_stack_impl<QC, O, D>(
        &self,
        query_ctx: &mut QC,
        stack: &mut crate::kd_tree::query_stack_simd::SimdQueryStack<O, SS>,
        mut process_leaf: impl FnMut(&LeafView<A, T, K, B>, &[O; K], &mut QC),
    ) where
        QC: QueryContext<A, O, K>,
        O: AxisUnified<Coord = O> + SimdPrune + BacktrackBlock3 + BacktrackBlock4,
        D: DistanceMetricUnified<A, K, Output = O>
            + DistanceMetricSimdBlock3<A, K, O>
            + DistanceMetricSimdBlock4<A, K, O>,
        SS: StemStrategy<
            StackContext<O> = crate::kd_tree::query_stack_simd::SimdQueryStackContext<O, SS>,
        >,
    {
        use crate::kd_tree::query_stack_simd::SimdQueryStackContext;

        let stems_ptr = NonNull::new(self.stems.as_ptr() as *mut u8).unwrap();
        let stem_strat: SS = SS::new(stems_ptr);

        let query: [A; K] = *query_ctx.query();
        let mut query_wide: [O; K] = [O::zero(); K];
        for dim in 0..K {
            query_wide[dim] = D::widen_coord(query[dim]);
        }

        let mut off = [O::zero(); K];
        stack.push(SimdQueryStackContext::new_single(stem_strat));

        // Backtracking loop
        while let Some(ctx) = stack.pop() {
            match ctx {
                SimdQueryStackContext::Single {
                    stem_strat: mut ss,
                    old_off,
                    rd,
                } => {
                    // Single entry - standard scalar processing
                    let mut dim = ss.dim();
                    tracing::trace!(%dim, %old_off, %rd, ?off, "Popped single context");

                    let max_dist = query_ctx.max_dist();
                    let rd_vs_max = O::cmp(rd, max_dist);
                    // TOOO: investigate into whether prune_on_equal_max_dist can be removed
                    let should_prune = rd_vs_max == std::cmp::Ordering::Greater
                        || (query_ctx.prune_on_equal_max_dist()
                            && rd_vs_max == std::cmp::Ordering::Equal);
                    if should_prune {
                        tracing::trace!(%rd, %max_dist, "Prune check: PRUNE");
                        continue;
                    }
                    tracing::trace!(%rd, %max_dist, "SIMD Prune check: VISIT");

                    tracing::trace!(
                        "Restoring off[{}]. was {}, now {}",
                        dim,
                        unsafe { *off.get_unchecked(dim) },
                        old_off
                    );
                    unsafe { *off.get_unchecked_mut(dim) = old_off };

                    let best_dist = query_ctx.max_dist();
                    if let Some(leaf_idx) = self.traverse_to_leaf_simd::<O, D>(
                        &query,
                        &query_wide,
                        &mut ss,
                        &mut off,
                        &mut dim,
                        rd,
                        best_dist,
                        stack,
                    ) {
                        tracing::trace!(%leaf_idx, "processing leaf");
                        let leaf_view = self.leaves.leaf_view(leaf_idx);
                        process_leaf(&leaf_view, &query_wide, query_ctx);
                    }
                }
                SimdQueryStackContext::Block {
                    siblings,
                    rd_values,
                    new_off_values,
                    sibling_mask,
                    dim: dim_val,
                    old_off,
                } => {
                    tracing::trace!(%dim_val, %old_off, ?rd_values, %sibling_mask, "Popped block context");
                    // Restore the parent split-dimension offset captured for this block context.
                    unsafe { *off.get_unchecked_mut(dim_val) = old_off };

                    // SIMD pruning: check which siblings pass the backtrack test
                    let max_dist = query_ctx.max_dist();
                    let surviving_mask =
                        simd::simd_prune_block::<O>(&rd_values, max_dist, sibling_mask);

                    if surviving_mask == 0 {
                        tracing::trace!("All siblings pruned");
                        continue;
                    }

                    tracing::trace!(
                        surviving_mask = format!("{:08b}", surviving_mask),
                        "Some siblings survive"
                    );

                    // Save the current off state before processing siblings
                    let saved_off = off;

                    // Process each surviving sibling
                    for sibling_idx in 0..8 {
                        if surviving_mask & (1 << sibling_idx) != 0 {
                            let mut ss = siblings[sibling_idx].clone();
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
                            unsafe { *off.get_unchecked_mut(dim_val) = new_off };

                            let best_dist = query_ctx.max_dist();
                            if let Some(leaf_idx) = self.traverse_to_leaf_simd::<O, D>(
                                &query,
                                &query_wide,
                                &mut ss,
                                &mut off,
                                &mut dim,
                                rd,
                                best_dist,
                                stack,
                            ) {
                                tracing::trace!(%leaf_idx, "processing leaf");
                                let leaf_view = self.leaves.leaf_view(leaf_idx);
                                process_leaf(&leaf_view, &query_wide, query_ctx);
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
        off: &mut [O; K],
        dim: &mut usize,
        rd: O,
        best_dist: O,
        stack: &mut crate::kd_tree::query_stack_simd::SimdQueryStack<O, SS>,
    ) -> Option<usize>
    where
        O: AxisUnified<Coord = O> + BacktrackBlock3 + BacktrackBlock4,
        D: DistanceMetricUnified<A, K, Output = O>
            + DistanceMetricSimdBlock3<A, K, O>
            + DistanceMetricSimdBlock4<A, K, O>,
        SS: StemStrategy<
            StackContext<O> = crate::kd_tree::query_stack_simd::SimdQueryStackContext<O, SS>,
        >,
    {
        let use_scalar_step = matches!(
            self.stem_leaf_resolution,
            crate::kd_tree::StemLeafResolution::Mapped { .. }
        );

        loop {
            let stem_idx = stem_strat.stem_idx();
            // Check if current stem points directly to a leaf
            if let Some(leaf_idx) = LS::Mutability::resolve_terminal_stem_idx(self, stem_idx) {
                return Some(leaf_idx);
            }
            let stem_oob = stem_idx >= self.stems.len();
            if stem_oob
                && matches!(
                    self.stem_leaf_resolution,
                    crate::kd_tree::StemLeafResolution::Mapped { .. }
                )
            {
                tracing::warn!(
                    stem_idx,
                    stems_len = self.stems.len(),
                    level = stem_strat.level(),
                    leaf_idx_prefix = stem_strat.leaf_idx(),
                    "SIMD traversal reached non-terminal out-of-bounds stem index; skipping subtree"
                );
                return None;
            }

            let should_continue = if use_scalar_step {
                use crate::kd_tree::query_stack_simd::SimdQueryStackContext;

                if stem_strat.level() > self.max_stem_level {
                    false
                } else {
                    let dim_val = *dim;
                    let pivot = if stem_oob {
                        A::max_value()
                    } else {
                        unsafe { *self.stems.get_unchecked(stem_idx) }
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

                        if O::cmp(rd_far, best_dist) != std::cmp::Ordering::Greater {
                            stack.push(SimdQueryStackContext::Single {
                                stem_strat: far_ctx,
                                old_off: new_off,
                                rd: rd_far,
                            });
                        }
                    } else {
                        // +Inf pivots can still encode structural branches in left-aligned trees.
                        // Traversing right should not add geometric distance in this case.
                        let far_ctx = stem_strat.branch_relative(false);
                        if O::cmp(rd, best_dist) != std::cmp::Ordering::Greater {
                            stack.push(SimdQueryStackContext::Single {
                                stem_strat: far_ctx,
                                old_off,
                                rd,
                            });
                        }
                    }

                    *dim = stem_strat.dim();
                    true
                }
            } else {
                // Delegate to strategy-specific block traversal step.
                // SAFETY: Cast concrete SimdQueryStack to associated type SS::Stack for trait call.
                let stack_ref = unsafe {
                    &mut *(stack as *mut crate::kd_tree::query_stack_simd::SimdQueryStack<O, SS>
                        as *mut SS::Stack<O>)
                };
                stem_strat.backtracking_traverse_step::<A, O, D, K>(
                    &self.stems,
                    query,
                    query_wide,
                    off,
                    dim,
                    rd,
                    self.max_stem_level,
                    best_dist,
                    stack_ref,
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
            self.stem_leaf_resolution
                .resolve_terminal_stem_idx(stem_strat.stem_idx(), stem_strat.leaf_idx()),
        )
    }
}
