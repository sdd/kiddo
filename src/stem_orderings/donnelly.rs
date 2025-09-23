use crate::modified_van_emde_boas::modified_van_emde_boas_get_child_idx_v2_branchless;
use crate::traits::Axis;
use crate::StemOrdering;
use aligned_vec::AVec;
use cmov::Cmov;

/// Donnelly Stem Ordering
#[derive(Clone, Debug)]
pub struct Donnelly {
    minor_level: u32,
}

impl StemOrdering for Donnelly {
    fn new_query() -> Self {
        Self { minor_level: 0 }
    }

    fn get_child_idx(&mut self, is_right_child: bool, curr_idx: usize) -> usize {
        let result = modified_van_emde_boas_get_child_idx_v2_branchless(
            curr_idx as u32,
            is_right_child,
            self.minor_level,
        );

        self.minor_level += 1;
        self.minor_level.cmovnz(&0, u8::from(self.minor_level == 3));

        result as usize
    }

    fn get_both_child_idx(&mut self, curr_idx: usize) -> (usize, usize) {
        let left = modified_van_emde_boas_get_child_idx_v2_branchless(
            curr_idx as u32,
            false,
            self.minor_level,
        );

        let right = modified_van_emde_boas_get_child_idx_v2_branchless(
            curr_idx as u32,
            true,
            self.minor_level,
        );

        self.minor_level += 1;
        self.minor_level.cmovnz(&0, u8::from(self.minor_level == 3));

        (left as usize, right as usize)
    }

    fn get_closer_and_further_child_idx(
        &mut self,
        curr_idx: usize,
        is_right_child: bool,
    ) -> (usize, usize) {
        let (left, right) = self.get_both_child_idx(curr_idx);

        if is_right_child {
            (right, left)
        } else {
            (left, right)
        }
    }

    fn get_initial_idx() -> usize {
        0
    }

    fn get_stem_node_count_from_leaf_node_count(leaf_node_count: usize) -> usize {
        if leaf_node_count < 2 {
            0
        } else {
            leaf_node_count.next_power_of_two() - 1
        }
    }

    // TODO: It would be nice to be able to determine the exact required length up-front.
    //  Instead, we just trim the stems afterwards by traversing right-child non-inf nodes
    //  till we hit max level to get the max used stem
    fn stem_node_padding_factor() -> usize {
        5
    }
    fn trim_unneeded_stems<A: Axis>(stems: &mut AVec<A>, max_stem_level: usize) {
        if !stems.is_empty() {
            let mut level: usize = 0;
            let mut minor_level: u64 = 0;
            let mut stem_idx = 0;
            loop {
                let val = stems[stem_idx];
                let is_right_child = val.is_finite();
                stem_idx = modified_van_emde_boas_get_child_idx_v2_branchless(
                    stem_idx as u32,
                    is_right_child,
                    minor_level as u32,
                ) as usize;
                level += 1;
                minor_level += 1;
                minor_level.cmovnz(&0, u8::from(minor_level == 3));
                if level == max_stem_level {
                    break;
                }
            }
            stems.truncate(stem_idx + 1);
        }
    }
}
