//! Eytzinger Stem Strategy with prefetch

use crate::stem_strategy::prefetch::{prefetch_t0, prefetch_t1};
use crate::{Axis, StemStrategy};
use std::ptr::NonNull;

#[derive(Clone, Debug, PartialOrd, Ord, Eq, PartialEq)]
pub enum PrefetchAction {
    T0 = 0,
    T1 = 1,
    None = -1,
}

// Eytzinger stem strategy, default prefetch
pub type Eytzinger = EytzingerFlexPf<0, 1>;

// Eytzinger stem strategy, no prefetch
pub type EytzingerNoPf = EytzingerFlexPf<-1, -1>;

/// Eytzinger stem strategy, customizable prefetch
#[derive(Clone, Debug)]
pub struct EytzingerFlexPf<const PF1: isize = 0, const PF2: isize = 1> {
    stem_idx: u32,
    dim: usize,
    level: i32,

    stems_ptr: NonNull<u8>,
}

pub struct EytzingerFlexDeferred {
    stem_idx: u32,
    level: u16,
    dim: u16,
}

unsafe impl<const PF1: isize, const PF2: isize> Send for EytzingerFlexPf<PF1, PF2> {}
unsafe impl<const PF1: isize, const PF2: isize> Sync for EytzingerFlexPf<PF1, PF2> {}

impl<const PF1: isize, const PF2: isize> StemStrategy for EytzingerFlexPf<PF1, PF2> {
    const ROOT_IDX: usize = 1;

    type DeferredState = EytzingerFlexDeferred;
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
        EytzingerFlexDeferred {
            stem_idx: self.stem_idx,
            level: self.level as u16,
            dim: self.dim as u16,
        }
    }
    fn rehydrate_deferred_state(&mut self, state: Self::DeferredState) {
        self.stem_idx = state.stem_idx;
        self.level = state.level as i32;
        self.dim = state.dim as usize;
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
    fn traverse<A: Axis, const K: usize>(&mut self, is_right_child: bool) {
        self.stem_idx = Self::step_pure::<A>(self.stem_idx, is_right_child, self.stems_ptr);

        self.level = self.level.wrapping_add(1);

        let wrap_dim_mask = 0usize.wrapping_sub((self.dim == (K - 1)) as usize);
        self.dim = self.dim.wrapping_add(1) & !wrap_dim_mask;
    }

    #[cfg(feature = "simulator")]
    fn simulate_traverse<A, const K: usize>(
        &mut self,
        is_right: bool,
        event_tx: &std::sync::mpsc::Sender<crate::test_utils::cache_simulator::Event>,
    ) where
        A: Axis<Coord = A>,
    {
        self.traverse::<A, K>(is_right);

        // MCA analysis shows that Eytzinger step_pure is just one LEA instr with est 3.5IPC and est
        // RThroughput of 0.5. Adding the estimate for the level and dim updating gets us to 1.5 to 2 cycles
        let _ = event_tx.send(crate::test_utils::cache_simulator::Event::Working(2));
    }

    fn branch<const K: usize>(&mut self) -> Self {
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

    fn child_indices(&self) -> (usize, usize) {
        let left = (self.stem_idx << 1) as usize;
        let right = left | 1;
        (left, right)
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
}

impl<const PF1: isize, const PF2: isize> EytzingerFlexPf<PF1, PF2> {
    #[allow(missing_docs)]
    #[inline(always)]
    pub fn step_pure<A: Axis>(stem_idx: u32, is_right_child: bool, stems_ptr: NonNull<u8>) -> u32 {
        let result = stem_idx.wrapping_shl(1) | is_right_child as u32;

        match PF1 {
            0 => unsafe {
                let nxt_ptr = stems_ptr
                    .as_ptr()
                    .add((result.wrapping_shl(1) as usize) * A::VALUE_WIDTH_BYTES);
                prefetch_t0(nxt_ptr);
            },
            1 => unsafe {
                let nxt_ptr = stems_ptr
                    .as_ptr()
                    .add((result.wrapping_shl(1) as usize) * A::VALUE_WIDTH_BYTES);
                prefetch_t1(nxt_ptr);
            },
            _ => {}
        };

        match PF2 {
            0 => unsafe {
                let far_ptr = stems_ptr
                    .as_ptr()
                    .add((result.wrapping_shl(4) as usize) * A::VALUE_WIDTH_BYTES);
                prefetch_t0(far_ptr);
            },
            1 => unsafe {
                let far_ptr = stems_ptr
                    .as_ptr()
                    .add((result.wrapping_shl(4) as usize) * A::VALUE_WIDTH_BYTES);
                prefetch_t1(far_ptr);
            },
            _ => {}
        };

        result
    }
}

/// Exposed pure function for use with cargo-asm
#[inline(never)]
pub fn calc_child_idx(curr_idx: u32, is_right_child: bool, stems_ptr: NonNull<u8>) -> u32 {
    EytzingerFlexPf::<0, 1>::step_pure::<f64>(curr_idx, is_right_child, stems_ptr)
}
