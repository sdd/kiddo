use crate::donnelly_stem_layout::donnelly_get_idx_v2_branchless;
use crate::traits::Axis;
use crate::StemStrategy;
use aligned_vec::AVec;
use cmov::Cmov;

#[derive(Copy, Clone)]
pub struct DonnellyFullArith<const L: u32, const CL: u32, const VB: u32> {
    minor: u32,     // 0..L-1
    min_idx: u32,   // index within block
    maj_idx: u32,   // block number
    base_maj: u32,  // maj_idx << L
    base_majL: u32, // (maj_idx << L) << L  == maj_idx << (2*L)
}

impl<const L: u32, const CL: u32, const VB: u32> StemStrategy for DonnellyFullArith<L, CL, VB> {
    #[inline(always)]
    fn new_query() -> Self {
        debug_assert!(L >= 2 && L <= 8);
        Self {
            minor: 0,
            min_idx: 0,
            maj_idx: 0,
            base_maj: 0,
            base_majL: 0,
        }
    }

    #[inline(always)]
    fn get_child_idx(&mut self, is_right: bool, _curr_idx: usize) -> usize {
        Self::get_child_idx_inner(
            self.minor,
            self.min_idx,
            self.maj_idx,
            self.base_maj,
            self.base_majL,
            is_right,
        )
    }

    fn get_both_child_idx(&mut self, _child_idx: usize) -> (usize, usize) {
        unimplemented!()
        // let left_child = self.get_child_idx_no_advance(false);
        // let right_child = self.get_child_idx_no_advance(true);
        //
        // (left_child, right_child)
    }

    fn get_closer_and_further_child_idx(
        &mut self,
        curr_idx: usize,
        is_right_child: bool,
    ) -> (usize, usize) {
        unimplemented!()
        // let (left, right) = self.get_both_child_idx(curr_idx);
        //
        // if is_right_child {
        //     (right, left)
        // } else {
        //     (left, right)
        // }
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
                stem_idx = donnelly_get_idx_v2_branchless(
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

impl<const L: u32, const CL: u32, const VB: u32> DonnellyFullArith<L, CL, VB> {
    #[inline(always)]
    fn get_child_idx_no_advance(&mut self, is_right: bool) -> usize {
        let minor = self.minor;
        let min_idx = self.min_idx;
        let maj_idx = self.maj_idx;
        let base_maj = self.base_maj;
        let base_majL = self.base_majL;

        Self::get_child_idx_inner(minor, min_idx, maj_idx, base_maj, base_majL, is_right)
    }

    #[inline(always)]
    pub fn get_child_idx_inner(
        mut minor: u32,
        mut min_idx: u32,
        mut maj_idx: u32,
        mut base_maj: u32,
        mut base_majL: u32,

        is_right: bool,
    ) -> usize {
        let right = is_right as u32;

        // bump minor without %
        let t = minor + 1;
        let inc = (t == L) as u32;
        minor = t - inc * L;

        // same-block candidate (cheap ALU)
        let incr = ((min_idx << 1) + 1) + right;
        let new_min = min_idx + incr;
        let same = base_maj + new_min;

        // next-block candidate (all ALU)
        let min_row = min_idx.wrapping_sub(L - 1);
        let term = ((min_row << 1) + 1 + right) << L;

        // update bases branchlessly
        let inc_base = inc << L;
        let inc_baseL = inc << (2 * L);
        let base_maj2 = base_maj + inc_base;
        let base_majL2 = base_majL + inc_baseL;

        let next = base_majL2 + term;

        // state
        min_idx = new_min;
        maj_idx += inc;
        base_maj = base_maj2;
        base_majL = base_majL2;

        // final select (LLVM will typically pick cmov on x86_64)
        let m = 0u32.wrapping_sub(inc);
        ((same & !m) | (next & m)) as usize
    }
}

// Exposed pure function for use with cargo-asm
#[inline(never)]
pub fn calc_child_idx(
    minor: u32,
    min_idx: u32,
    maj_idx: u32,
    base_maj: u32,
    base_majL: u32,
    is_right: bool,
) -> usize {
    DonnellyFullArith::<3, 64, 4>::get_child_idx_inner(
        minor, min_idx, maj_idx, base_maj, base_majL, is_right,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case(vec![], 0)]
    #[case(vec![false], 1)] // 1 Maj idx: 1
    #[case(vec![true], 2)] // 2
    #[case(vec![false, false], 3)] // 3
    #[case(vec![false, true], 4)] // 4
    #[case(vec![true, false], 5)] // 5
    #[case(vec![true, true], 6)] // 6
    #[case(vec![false, false, false], 8)] // 7
    #[case(vec![false, false, true], 16)] // 8
    #[case(vec![false, true, false], 24)] // 9
    #[case(vec![false, true, true], 32)] // 10
    #[case(vec![true, false, false], 40)] // 11
    #[case(vec![true, false, true], 48)] // 12
    #[case(vec![true, true, false], 56)] // 13
    #[case(vec![true, true, true], 64)] // 14
    #[case(vec![false, false, false, false], 9)] // 15 Maj idx: 2
    #[case(vec![false, false, false, true], 10)] // 16
    #[case(vec![false, false, false, false, false], 11)] // 17
    #[case(vec![false, false, false, false, true], 12)] // 18
    #[case(vec![false, false, false, true, false], 13)] // 19
    #[case(vec![false, false, false, true, true], 14)] // 20
    #[case(vec![false, false, false, false, false, false], 72)] // 21
    #[case(vec![false, false, false, false, false, true], 80)] // 22
    #[case(vec![false, false, false, false, true, false], 88)] // 23
    #[case(vec![false, false, false, false, true, true], 96)] // 24
    #[case(vec![false, false, false, true, false, false], 104)] // 25
    #[case(vec![false, false, false, true, false, true], 112)] // 26
    #[case(vec![false, false, false, true, true, false], 120)] // 27
    #[case(vec![false, false, false, true, true, true], 128)] // 28
    #[case(vec![false, false, true, false], 17)] // 29  Maj index: 3
    #[case(vec![false, false, true, true], 18)] // 30
    #[case(vec![false, false, true, false, false], 19)] // 31
    #[case(vec![false, false, true, false, true], 20)] // 32
    #[case(vec![false, false, true, true, false], 21)] // 33
    #[case(vec![false, false, true, true, true], 22)] // 34
    #[case(vec![false, false, true, false, false, false], 136)] // 35
    #[case(vec![false, false, true, false, false, true], 144)] // 36
    #[case(vec![false, false, true, false, true, false], 152)] // 37
    #[case(vec![false, false, true, false, true, true], 160)] // 38
    #[case(vec![false, false, true, true, false, false], 168)] // 39
    #[case(vec![false, false, true, true, false, true], 176)] // 40
    #[case(vec![false, false, true, true, true, false], 184)] // 41
    #[case(vec![false, false, true, true, true, true], 192)] // 42
    fn donnelly_v2_get_child_idx_produces_correct_values(
        #[case] input: Vec<bool>,
        #[case] expected: usize,
    ) {
        let mut stem_strat = DonnellyFullArith::<3, 64, 4>::new_query();
        let mut result = 0;
        input.iter().for_each(|selection| {
            result = stem_strat.get_child_idx(*selection, result);
        });

        assert_eq!(result, expected);
    }
}
