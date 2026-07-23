//! Benchmark-only SIMD-descent layout reserving the final block level for descriptors.

use std::ptr::NonNull;

use crate::stem_strategy::donnelly::core::DonnellyCore;
use crate::{Axis, StemStrategy};

/// Block-height-three, block-dimension traversal with terminal leaf metadata.
///
/// Complete pivot blocks are traversed block-at-once by the benchmark query
/// entry point. The final block contains two pivot levels and four leaf
/// descriptors in place of its third pivot level.
#[doc(hidden)]
#[derive(Copy, Clone, Debug)]
pub struct DonnellySimdDescentLeafEmbedded3 {
    core: DonnellyCore<3>,
}

impl DonnellySimdDescentLeafEmbedded3 {
    #[inline(always)]
    pub(crate) fn traverse_block<const K: usize>(&mut self, child_idx: u8) {
        self.core.traverse_block::<K>(child_idx, 3);
    }

    #[inline(always)]
    pub(crate) fn leaf_idx_with_terminal_rank(&self, terminal_rank: u8) -> usize {
        debug_assert!(terminal_rank < 4);
        self.core.leaf_idx().wrapping_shl(2) | usize::from(terminal_rank)
    }
}

impl StemStrategy for DonnellySimdDescentLeafEmbedded3 {
    const ROOT_IDX: usize = 0;
    const BLOCK_SIZE: usize = 3;
    const TERMINAL_METADATA_LEVELS: usize = 1;

    type DeferredState = Self;
    type StackContext<A> = crate::kd_tree::query_stack::QueryStackContext<A, Self::DeferredState>;
    type Stack<A> = crate::kd_tree::query_stack::QueryStack<A, Self>;

    #[inline(always)]
    fn new(stems_ptr: NonNull<u8>) -> Self {
        Self {
            core: DonnellyCore::new(stems_ptr),
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
    fn dim<const K: usize>(&self) -> usize {
        self.core.level() as usize / 3 % K
    }

    #[inline(always)]
    fn construction_dim<const K: usize>(&self) -> usize {
        self.core.level() as usize / 3 % K
    }

    #[inline(always)]
    fn level(&self) -> i32 {
        self.core.level()
    }

    #[inline(always)]
    fn traverse<A: Axis<Coord = A>, const K: usize>(&mut self, is_right: bool) {
        self.core.traverse::<A, K>(is_right);
    }

    #[inline(always)]
    fn traverse_head<A: Axis<Coord = A>, const K: usize>(&mut self, is_right: bool) {
        self.core.traverse_head::<A, K>(is_right);
    }

    #[inline(always)]
    fn traverse_tail<A: Axis<Coord = A>, const K: usize>(&mut self, is_right: bool) {
        self.core.traverse_tail_with_block_size::<A, K>(is_right, 3);
    }

    #[inline(always)]
    fn branch<A: Axis<Coord = A>, const K: usize>(&mut self) -> Self {
        Self {
            core: self.core.branch::<A, K>(),
        }
    }

    #[inline(always)]
    fn child_indices<A: Axis<Coord = A>>(&self) -> (usize, usize) {
        self.core.child_indices::<A>()
    }

    fn get_leaf_idx<A: Axis<Coord = A>, const K: usize>(
        stems: &[A],
        query: &[A; K],
        max_stem_level: i32,
    ) -> usize {
        let stems_ptr = NonNull::new(stems.as_ptr() as *mut u8).unwrap();
        let mut strat = Self::new(stems_ptr);

        while strat.level() <= max_stem_level {
            let pivot = unsafe { *stems.get_unchecked(strat.stem_idx()) };
            let query_value = unsafe { *query.get_unchecked(strat.dim::<K>()) };
            strat.traverse::<A, K>(query_value >= pivot);
        }

        strat.leaf_idx()
    }
}
