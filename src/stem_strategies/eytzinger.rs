use aligned_vec::AVec;

use crate::StemStrategy;

/// Eytzinger Stem Ordering
#[derive(Clone, Debug)]
pub struct Eytzinger<const K: usize> {
    stem_idx: u32,
    dim: usize,
    level: i32,
}

impl<const K: usize> StemStrategy for Eytzinger<K> {
    fn new() -> Self {
        Self {
            stem_idx: 1,
            dim: 0,
            level: 0,
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

    fn traverse(&mut self, is_right_child: bool) {
        self.stem_idx = self.stem_idx.wrapping_shl(1) | is_right_child as u32;

        self.level = self.level.wrapping_add(1);

        let wrap_dim_mask = 0usize.wrapping_sub((self.dim == (K - 1)) as usize);
        self.dim = self.dim.wrapping_add(1) & !wrap_dim_mask;
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

/// Get the child index of a node.
pub fn get_child_idx(is_right_child: bool, curr_idx: usize) -> usize {
    curr_idx << 1 | is_right_child as usize
}
