use aligned_vec::AVec;

use crate::StemStrategy;

/// Eytzinger Stem Ordering
#[derive(Clone, Debug)]
pub struct Eytzinger;

impl StemStrategy for Eytzinger {
    fn new_query() -> Self {
        Self
    }
    fn get_child_idx(&mut self, is_right_child: bool, curr_idx: usize) -> usize {
        curr_idx << 1 | is_right_child as usize
    }

    fn get_both_child_idx(&mut self, curr_idx: usize) -> (usize, usize) {
        let left = curr_idx << 1;
        let right = left | 1;
        (left, right)
    }

    fn get_closer_and_further_child_idx(
        &mut self,
        curr_idx: usize,
        is_right_child: bool,
    ) -> (usize, usize) {
        let left = curr_idx << 1;
        let right = left | 1;

        if is_right_child {
            (right, left)
        } else {
            (left, right)
        }
    }

    fn get_initial_idx() -> usize {
        1
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
