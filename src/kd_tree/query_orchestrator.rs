use crate::kd_tree::leaf_view::LeafView;
use crate::kd_tree::query_stack::{QueryStack, QueryStackContext};
use crate::kd_tree::traits::QueryContext;
use crate::kd_tree::KdTree;
use crate::traits_unified_2::{
    AxisUnified, Basics, DistanceMetricUnified, LeafStrategy, Mutability,
};
use crate::StemStrategy;
use std::ptr::NonNull;

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

        let leaf_view = self.leaves.leaf_view(leaf_idx);
        process_leaf(&leaf_view);
    }

    /// Get the leaf index for a query (unmapped leaves)
    #[inline]
    pub(crate) fn get_leaf_idx_unmapped(&self, query: &[A; K]) -> usize {
        let stems_ptr = NonNull::new(self.stems.as_ptr() as *mut u8).unwrap();
        let mut stem_strat: SS = SS::new(stems_ptr);

        while stem_strat.level() <= self.max_stem_level {
            let pivot = unsafe { self.stems.get_unchecked(stem_strat.stem_idx()) };
            let is_right_child: bool = *unsafe { query.get_unchecked(stem_strat.dim()) } >= *pivot;
            stem_strat.traverse(is_right_child);
        }

        stem_strat.leaf_idx()
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

        match &self.stem_leaf_resolution {
            crate::kd_tree::StemLeafResolution::Mapped { leaf_idx_map, .. } => {
                leaf_idx_map[stem_strat.stem_idx()].unwrap().get()
            }
            _ => unreachable!(),
        }
    }

    /// Check if a stem points directly to a leaf
    #[inline(always)]
    pub(crate) fn resolve_terminal_stem(&self, stem_idx: usize) -> Option<usize> {
        match &self.stem_leaf_resolution {
            crate::kd_tree::StemLeafResolution::Mapped {
                min_stem_leaf_idx,
                leaf_idx_map,
            } => {
                if stem_idx >= *min_stem_leaf_idx {
                    let map_idx = stem_idx - *min_stem_leaf_idx;
                    leaf_idx_map
                        .get(map_idx)
                        .and_then(|opt| opt.map(|n| n.get()))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Backtracking query
    #[inline(always)]
    pub(crate) fn backtracking_query<QC, O, D>(
        &self,
        query_ctx: &mut QC,
        process_leaf: impl FnMut(&LeafView<A, T, K, B>, &mut QC),
    ) where
        QC: QueryContext<A, O, K>,
        O: AxisUnified<Coord = O>,
        D: DistanceMetricUnified<A, K, Output = O>,
    {
        let mut stack = QueryStack::new();
        self.backtracking_query_with_stack::<QC, O, D>(query_ctx, &mut stack, process_leaf);
    }

    /// Backtracking query with explicit stack
    #[inline(always)]
    pub(crate) fn backtracking_query_with_stack<QC, O, D>(
        &self,
        query_ctx: &mut QC,
        stack: &mut QueryStack<O, SS>,
        mut process_leaf: impl FnMut(&LeafView<A, T, K, B>, &mut QC),
    ) where
        QC: QueryContext<A, O, K>,
        O: AxisUnified<Coord = O>,
        D: DistanceMetricUnified<A, K, Output = O>,
    {
        let stems_ptr = NonNull::new(self.stems.as_ptr() as *mut u8).unwrap();
        let stem_strat: SS = SS::new(stems_ptr);

        let query: [A; K] = *query_ctx.query();
        let mut query_wide: [O; K] = [O::zero(); K];
        for dim in 0..K {
            query_wide[dim] = D::widen_coord(query[dim]);
        }

        let mut off = [O::zero(); K];
        stack.push(QueryStackContext::new(stem_strat));

        while let Some(stack_ctx) = stack.pop() {
            let (mut stem_strat, old_off, rd) = stack_ctx.into_parts();
            let mut dim = stem_strat.dim();
            tracing::trace!(%dim, %old_off, %rd, ?off, "Popped stack context");

            let max_dist = query_ctx.max_dist();
            if O::cmp(rd, max_dist) == std::cmp::Ordering::Greater {
                tracing::trace!(%rd, %max_dist, "Prune check: PRUNE");
                continue;
            }
            tracing::trace!(%rd, %max_dist, "Prune check: VISIT");

            tracing::trace!("Restoring off[{}]. was {}, now {}", dim, off[dim], old_off);
            off[dim] = old_off;

            let leaf_idx = self.traverse_to_leaf::<O, D>(
                &query,
                &query_wide,
                &mut stem_strat,
                &mut off,
                &mut dim,
                rd,
                stack,
            );

            tracing::trace!(%leaf_idx, "processing leaf");
            let leaf_view = self.leaves.leaf_view(leaf_idx);
            process_leaf(&leaf_view, query_ctx);
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
        stack: &mut QueryStack<O, SS>,
    ) -> usize
    where
        O: AxisUnified<Coord = O>,
        D: DistanceMetricUnified<A, K, Output = O>,
    {
        while stem_strat.level() <= self.max_stem_level {
            // Check if current stem points directly to a leaf
            // For Immutable trees, this should optimise away since resolve_terminal_stem_idx
            // will always return None
            if let Some(leaf_idx) =
                LS::Mutability::resolve_terminal_stem_idx(self, stem_strat.stem_idx())
            {
                return leaf_idx;
            }

            let pivot = *unsafe { self.stems.get_unchecked(stem_strat.stem_idx()) };

            // if the pivot is Inf or MAX_VAL, we can never hit the right branch, so
            // we will never need to backtrack to where we are now.
            if pivot < A::max_value() {
                let query_elem = *unsafe { query.get_unchecked(*dim) };
                let is_right_child = query_elem >= pivot;

                let far_ctx = stem_strat.branch_relative(is_right_child);

                let pivot_wide: O = D::widen_coord(pivot);
                let query_elem_wide = *unsafe { query_wide.get_unchecked(*dim) };

                let new_off = O::saturating_dist(query_elem_wide, pivot_wide);
                let old_off = *unsafe { off.get_unchecked(*dim) };
                let rd_far = O::saturating_add(rd, D::dist1(new_off, old_off));

                tracing::trace!(
                    "new_off = dist({}, {}) = {}. rd = {}, rd_far = {}, off = {:?}",
                    query_elem_wide,
                    pivot_wide,
                    new_off,
                    rd,
                    rd_far,
                    off,
                );

                stack.push(QueryStackContext {
                    stem_strat: far_ctx,
                    old_off: new_off,
                    rd: rd_far,
                });
            } else {
                stem_strat.traverse(false);
            }

            *dim = stem_strat.dim();
        }

        stem_strat.leaf_idx()
    }
}
