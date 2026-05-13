//! Eytzinger Stem Ordering with prefetching for 2 levels

use crate::stem_strategy::prefetch::{prefetch_t0, prefetch_t1};
use crate::StemStrategy;
use std::ptr::NonNull;

/// Eytzinger Stem Ordering
#[derive(Clone, Debug)]
pub struct EytzingerPfFar<const K: usize, const VB: usize> {
    stem_idx: u32,
    dim: usize,
    level: i32,

    stems_ptr: NonNull<u8>,
}

// FIXME: this is a hack to make the compiler happy. remove after testing
unsafe impl<const K: usize, const VB: usize> Send for EytzingerPfFar<K, VB> {}
unsafe impl<const K: usize, const VB: usize> Sync for EytzingerPfFar<K, VB> {}

impl<const K: usize, const VB: usize> StemStrategy for EytzingerPfFar<K, VB> {
    const ROOT_IDX: usize = 1;

    type DeferredState = Self;
    type StackContext<A> = crate::kd_tree::query_stack::QueryStackContext<A, Self::DeferredState>;
    type Stack<A> = crate::kd_tree::query_stack::QueryStack<A, Self>;

    fn new(stems_ptr: NonNull<u8>) -> Self {
        Self {
            stem_idx: Self::ROOT_IDX as u32,
            dim: 0,
            level: 0,
            stems_ptr,
        }
    }

    fn stem_idx(&self) -> usize {
        self.stem_idx as usize
    }
    fn deferred_state(&self) -> Self::DeferredState {
        self.clone()
    }
    fn rehydrate_deferred_state(&mut self, state: Self::DeferredState) {
        *self = state;
    }
    fn leaf_idx(&self) -> usize {
        let mask = 1u32.wrapping_shl(self.level as u32);
        (self.stem_idx & !mask) as usize
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn level(&self) -> i32 {
        self.level
    }

    #[inline]
    fn traverse(&mut self, is_right_child: bool) {
        self.stem_idx = Self::step_pure(self.stem_idx, is_right_child, self.stems_ptr);

        self.level = self.level.wrapping_add(1);

        let wrap_dim_mask = 0usize.wrapping_sub((self.dim == (K - 1)) as usize);
        self.dim = self.dim.wrapping_add(1) & !wrap_dim_mask;
    }

    #[cfg(feature = "simulator")]
    fn simulate_traverse(
        &mut self,
        is_right: bool,
        event_tx: &std::sync::mpsc::Sender<crate::test_utils::cache_simulator::Event>,
    ) {
        self.traverse(is_right);

        // MCA analysis shows that Eytzinger step_pure is just one LEA instr with est 3.5IPC and est
        // RThroughput of 0.5. Adding the estimate for the level and dim updating gets us to 1.5 to 2 cycles
        let _ = event_tx.send(crate::test_utils::cache_simulator::Event::Working(2));
    }

    fn branch(&mut self) -> Self {
        self.stem_idx = self.stem_idx.wrapping_shl(1);
        let right = self.stem_idx | 1;

        self.level = self.level.wrapping_add(1);

        let wrap_dim_mask = 0usize.wrapping_sub((self.dim == (K - 1)) as usize);
        self.dim = self.dim.wrapping_add(1) & !wrap_dim_mask;

        Self {
            stem_idx: right,
            ..*self
        }
    }

    fn get_stem_node_count_from_leaf_node_count(leaf_node_count: usize) -> usize {
        if leaf_node_count < 2 {
            0
        } else {
            leaf_node_count.next_power_of_two()
        }
    }
    fn stem_node_padding_factor() -> usize {
        1
    }

    fn child_indices(&self) -> (usize, usize) {
        unimplemented!("child_indices not yet implemented for EytzingerPfFar")
    }
}

impl<const K: usize, const VB: usize> EytzingerPfFar<K, VB> {
    #[allow(missing_docs)]
    #[inline(always)]
    pub fn step_pure(stem_idx: u32, is_right_child: bool, stems_ptr: NonNull<u8>) -> u32 {
        let result = stem_idx.wrapping_shl(1) | is_right_child as u32;

        unsafe {
            let nxt_ptr = stems_ptr
                .as_ptr()
                .add((result.wrapping_shl(1) as usize) * VB);
            prefetch_t0(nxt_ptr);

            let far_ptr = stems_ptr
                .as_ptr()
                .add((result.wrapping_shl(4) as usize) * VB);
            prefetch_t1(far_ptr);
        }

        result
    }
}

/// Exposed pure function for use with cargo-asm
#[inline(never)]
pub fn calc_child_idx(curr_idx: u32, is_right_child: bool, stems_ptr: NonNull<u8>) -> u32 {
    EytzingerPfFar::<3, 8>::step_pure(curr_idx, is_right_child, stems_ptr)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eytzinger_pf_far_basics_and_traverse() {
        let stems = vec![0u8; 256];
        let stems_ptr = NonNull::new(stems.as_ptr() as *mut u8).unwrap();
        let mut strat = EytzingerPfFar::<3, 8>::new(stems_ptr);

        assert_eq!(strat.stem_idx(), 1);
        assert_eq!(strat.level(), 0);
        assert_eq!(strat.dim(), 0);
        assert_eq!(strat.leaf_idx(), 0);

        strat.traverse(false);
        assert_eq!(strat.stem_idx(), 2);
        assert_eq!(strat.level(), 1);
        assert_eq!(strat.dim(), 1);
        assert_eq!(strat.leaf_idx(), 0);

        strat.traverse(true);
        assert_eq!(strat.stem_idx(), 5);
        assert_eq!(strat.level(), 2);
        assert_eq!(strat.dim(), 2);
        assert_eq!(strat.leaf_idx(), 1);

        strat.traverse(false);
        assert_eq!(strat.stem_idx(), 10);
        assert_eq!(strat.level(), 3);
        assert_eq!(strat.dim(), 0);
        assert_eq!(strat.leaf_idx(), 2);
    }

    #[test]
    fn eytzinger_pf_far_branch_returns_right_and_mutates_self_left() {
        let stems = vec![0u8; 256];
        let stems_ptr = NonNull::new(stems.as_ptr() as *mut u8).unwrap();
        let mut strat = EytzingerPfFar::<3, 8>::new(stems_ptr);

        let right = strat.branch();

        assert_eq!(strat.stem_idx(), 2);
        assert_eq!(strat.level(), 1);
        assert_eq!(strat.dim(), 1);

        assert_eq!(right.stem_idx(), 3);
        assert_eq!(right.level(), 1);
        assert_eq!(right.dim(), 1);
    }

    #[test]
    fn eytzinger_pf_far_deferred_state_round_trip_restores_full_state() {
        let stems = vec![0u8; 256];
        let stems_ptr = NonNull::new(stems.as_ptr() as *mut u8).unwrap();
        let mut original = EytzingerPfFar::<4, 8>::new(stems_ptr);
        original.traverse(true);
        original.traverse(false);

        let state = original.deferred_state();

        let mut restored = EytzingerPfFar::<4, 8>::new(NonNull::dangling());
        restored.rehydrate_deferred_state(state);

        assert_eq!(restored.stem_idx(), original.stem_idx());
        assert_eq!(restored.level(), original.level());
        assert_eq!(restored.dim(), original.dim());
        assert_eq!(restored.leaf_idx(), original.leaf_idx());
    }

    #[test]
    fn eytzinger_pf_far_step_pure_and_calc_child_idx_match_eytzinger_layout() {
        let stems = vec![0u8; 512];
        let stems_ptr = NonNull::new(stems.as_ptr() as *mut u8).unwrap();

        assert_eq!(EytzingerPfFar::<3, 8>::step_pure(1, false, stems_ptr), 2);
        assert_eq!(EytzingerPfFar::<3, 8>::step_pure(1, true, stems_ptr), 3);
        assert_eq!(EytzingerPfFar::<3, 8>::step_pure(5, false, stems_ptr), 10);
        assert_eq!(EytzingerPfFar::<3, 8>::step_pure(5, true, stems_ptr), 11);

        assert_eq!(calc_child_idx(1, false, stems_ptr), 2);
        assert_eq!(calc_child_idx(1, true, stems_ptr), 3);
        assert_eq!(calc_child_idx(5, false, stems_ptr), 10);
        assert_eq!(calc_child_idx(5, true, stems_ptr), 11);
    }

    #[test]
    fn eytzinger_pf_far_leaf_count_metadata_matches_expected_padding() {
        type Strat = EytzingerPfFar<3, 8>;

        assert_eq!(Strat::get_stem_node_count_from_leaf_node_count(0), 0);
        assert_eq!(Strat::get_stem_node_count_from_leaf_node_count(1), 0);
        assert_eq!(Strat::get_stem_node_count_from_leaf_node_count(2), 2);
        assert_eq!(Strat::get_stem_node_count_from_leaf_node_count(3), 4);
        assert_eq!(Strat::get_stem_node_count_from_leaf_node_count(8), 8);
        assert_eq!(Strat::get_stem_node_count_from_leaf_node_count(9), 16);
        assert_eq!(Strat::stem_node_padding_factor(), 1);
    }

    #[test]
    fn eytzinger_pf_far_child_indices_is_currently_unimplemented() {
        let stems = [0u8; 64];
        let stems_ptr = NonNull::new(stems.as_ptr() as *mut u8).unwrap();
        let strat = EytzingerPfFar::<3, 8>::new(stems_ptr);

        let result = std::panic::catch_unwind(|| strat.child_indices());
        assert!(result.is_err());
    }
}
