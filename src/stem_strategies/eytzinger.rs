//! Eytzinger Stem Strategy

use crate::StemStrategy;
use aligned_vec::AVec;
use std::ptr::NonNull;

/// Eytzinger Stem Ordering
#[derive(Clone, Debug)]
pub struct Eytzinger<const K: usize> {
    stem_idx: u32,
    dim: usize,
    level: i32,

    stems_ptr: NonNull<u8>,
}

// FIXME: this is a hack to make the compiler happy. remove after testing
unsafe impl<const K: usize> Send for Eytzinger<K> {}
unsafe impl<const K: usize> Sync for Eytzinger<K> {}

impl<const K: usize> StemStrategy for Eytzinger<K> {
    fn new(stems_ptr: NonNull<u8>) -> Self {
        Self {
            stem_idx: 1,
            dim: 0,
            level: 0,
            stems_ptr,
        }
    }

    fn stem_idx(&self) -> usize {
        self.stem_idx as usize
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
        event_tx: &std::sync::mpsc::Sender<crate::cache_simulator::Event>,
    ) {
        self.traverse(is_right);

        // MCA analysis shows that Eytzinger step_pure is just one LEA instr with est 3.5IPC and est
        // RThroughput of 0.5. Adding the estimate for the level and dim updating gets us to 1.5 to 2 cycles
        let _ = event_tx.send(crate::cache_simulator::Event::Working(2));
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
    fn trim_unneeded_stems<A>(_stems: &mut AVec<A>, _max_stem_level: usize) {}
}

impl<const K: usize> Eytzinger<K> {
    #[allow(missing_docs)]
    #[inline(always)]
    pub fn step_pure(stem_idx: u32, is_right_child: bool, _stems_ptr: NonNull<u8>) -> u32 {
        stem_idx.wrapping_shl(1) | is_right_child as u32
    }
}

/// Exposed pure function for use with cargo-asm
#[inline(never)]
pub fn calc_child_idx(curr_idx: u32, is_right_child: bool, stems_ptr: NonNull<u8>) -> u32 {
    Eytzinger::<3>::step_pure(curr_idx, is_right_child, stems_ptr)
}
