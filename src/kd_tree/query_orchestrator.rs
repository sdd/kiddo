use crate::kd_tree::leaf_view::LeafView;
use crate::kd_tree::query_stack::{QueryStack, QueryStackContext};
use crate::kd_tree::traits::QueryContext;
use crate::kd_tree::KdTree;
use crate::traits_unified_2::{AxisUnified, Basics, DistanceMetricUnified, LeafStrategy};
use crate::StemStrategy;
use std::ptr::NonNull;

impl<A, T, SS, LS, const K: usize, const B: usize> KdTree<A, T, SS, LS, K, B>
where
    A: AxisUnified<Coord = A>,
    SS: StemStrategy,
{
    /// Get the leaf index for a query
    #[inline]
    pub(crate) fn get_leaf_idx(&self, query: &[A; K]) -> usize {
        let stems_ptr = NonNull::new(self.stems.as_ptr() as *mut u8).unwrap();
        let mut stem_strat: SS = SS::new(stems_ptr);

        while stem_strat.level() <= self.max_stem_level {
            let pivot = unsafe { self.stems.get_unchecked(stem_strat.stem_idx()) };
            let is_right_child: bool = *unsafe { query.get_unchecked(stem_strat.dim()) } >= *pivot;
            stem_strat.traverse(is_right_child);
        }

        stem_strat.leaf_idx()
    }
}

impl<A, T, SS, LS, const K: usize, const B: usize> KdTree<A, T, SS, LS, K, B>
where
    A: AxisUnified<Coord = A>,
    T: Basics + Copy + Default + PartialOrd + PartialEq,
    LS: LeafStrategy<A, T, SS, K, B>,
    SS: StemStrategy,
{
    /// Backtracking query.
    ///
    /// Used for exact queries. Wraps `backtracking_query_with_stack`
    /// and provides a default stack.
    /// Usually you would want to call one of the high-level query functions
    /// rather than this function directly.
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
        // TODO: replace with TLS-backed stack
        let mut stack = QueryStack::new();
        self.backtracking_query_with_stack::<QC, O, D>(query_ctx, &mut stack, process_leaf);
    }

    /// Non-backtracking query
    ///
    /// Used for approx-NN queries
    #[inline]
    pub(crate) fn straight_query<QC, O>(
        &self,
        query_ctx: QC,
        mut process_leaf: impl FnMut(&LeafView<A, T, K, B>),
    ) where
        QC: QueryContext<A, O, K>,
    {
        let leaf_idx = self.get_leaf_idx(query_ctx.query());

        let leaf_view = self.leaves.leaf_view(leaf_idx);
        process_leaf(&leaf_view);
    }

    /// Backtracking query with explicit stack.
    ///
    /// Used for exact queries. Wrapped by `backtracking_query`.
    /// Usually you would want to call one of the high-level query functions
    /// rather than this function directly.
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

            while stem_strat.level() <= self.max_stem_level {
                let pivot = *unsafe { self.stems.get_unchecked(stem_strat.stem_idx()) };
                let query_elem = *unsafe { query.get_unchecked(dim) };
                let is_right_child = query_elem >= pivot;

                let far_ctx = stem_strat.branch_relative(is_right_child);

                let pivot_wide: O = D::widen_coord(pivot);
                let query_elem_wide = *unsafe { query_wide.get_unchecked(dim) };

                let new_off = O::saturating_dist(query_elem_wide, pivot_wide);
                let old_off = *unsafe { off.get_unchecked(dim) };
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
                    // dim,
                    old_off: new_off,
                    rd: rd_far,
                });

                dim = stem_strat.dim();
            }

            tracing::trace!(leaf_idx = %stem_strat.leaf_idx(), "processing leaf");
            let leaf_view = self.leaves.leaf_view(stem_strat.leaf_idx());
            process_leaf(&leaf_view, query_ctx);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kd_tree::leaf_strategies::dummy::DummyLeafStrategy;
    use crate::Eytzinger;

    #[test]
    fn test_get_leaf_idx() {
        let tree: KdTree<f32, u32, Eytzinger<3>, DummyLeafStrategy, 3, 32> = KdTree::default();

        let query = [0.0f32; 3];

        let result = tree.get_leaf_idx(&query);

        assert_eq!(result, 3);
    }
}
