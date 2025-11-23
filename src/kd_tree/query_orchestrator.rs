use crate::kd_tree::query_stack::QueryStack;
use crate::kd_tree::traits::{QueryContext, ResultContext};
use crate::kd_tree::KdTree;
use crate::traits_unified::{AxisUnified, Basics, LeafStrategy};
use crate::StemStrategy;

impl<A, T, SS, LS, const K: usize, const B: usize> KdTree<A, T, SS, LS, K, B>
where
    A: AxisUnified,
    T: Basics + Copy + Default,
    LS: LeafStrategy<A, T, SS, K, B>,
    SS: StemStrategy,
{
    // High-level entry: obtains a temporary stack and delegates.
    #[inline(always)]
    pub(crate) fn query<QC, RC>(&self, query_ctx: &QC, result_ctx: &mut RC)
    where
        QC: QueryContext<A, K>,
        RC: ResultContext<A::NumType>,
    {
        // TODO: replace with TLS-backed stack
        let mut stack = QueryStack::new();
        self.query_with_stack(query_ctx, result_ctx, &mut stack);
    }

    // Core entry used by power users with a reusable stack.
    #[inline(always)]
    pub(crate) fn query_with_stack<QC, RC>(
        &self,
        _query_ctx: &QC,
        _result_ctx: &mut RC,
        _stack: &mut QueryStack,
    ) where
        QC: QueryContext<A, K>,
        RC: ResultContext<A::NumType>,
    {
        // Skeleton only; wire StemStrategy-driven descent here.
        let _ = (&self.stems, &self.leaves, self.max_stem_level);
    }
}
