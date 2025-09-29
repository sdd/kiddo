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
        self.step(is_right)
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
                let val = &stems[stem_idx];
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
    // Helpers derived from type-level params
    #[inline(always)]
    const fn items_per_line() -> u32 {
        CL / VB
    }
    #[inline(always)]
    const fn log2_items_per_line() -> u32 {
        Self::items_per_line().ilog2()
    }
    #[inline(always)]
    const fn last_row_start() -> u32 {
        (1u32 << (L - 1)) - 1
    } // 2^(L-1) - 1

    /// The pure step: returns (result_index, minor_lvl', min_idx', maj_idx', base_maj', base_majL').
    #[inline(always)]
    fn step_pure(
        mut minor_lvl: u32, // 0..L-1
        mut min_idx: u32,   // index within current minor triangle
        mut maj_idx: u32,   // which minor triangle (block) we’re in
        mut base_maj: u32,  // maj_idx << LOG2_ITEMS_PER_LINE
        mut base_majL: u32, // maj_idx << (LOG2_ITEMS_PER_LINE + L)  (kept for invariants)
        is_right: bool,
    ) -> (usize, u32, u32, u32, u32, u32) {
        debug_assert!(L >= 2 && L <= 8);

        let right = is_right as u32;

        // ---- advance minor level and detect boundary ----
        let t = minor_lvl.wrapping_add(1);
        let wrapped = (t == L) as u32; // 1 at boundary, else 0
        minor_lvl = t.wrapping_sub(wrapped * L); // wrap to 0 at boundary

        // ---- candidate inside SAME minor triangle ----
        // child index within current triangle
        let child_same = (min_idx << 1).wrapping_add(1).wrapping_add(right);
        let same = base_maj.wrapping_add(child_same);

        // ---- candidate in the NEXT minor triangle (at boundary only) ----
        // column in the last row of the current triangle
        let r = min_idx.wrapping_sub(Self::last_row_start());
        // child index inside the next triangle (its level-1 row)
        let child_next = (r << 1).wrapping_add(1).wrapping_add(right);

        // next triangle’s base (advance by ITEMS_PER_LINE)
        let base_step = wrapped << Self::log2_items_per_line();
        let base_maj_nxt = base_maj.wrapping_add(base_step);
        let next = base_maj_nxt.wrapping_add(child_next);

        // ---- branchless select between SAME and NEXT ----
        let m = 0u32.wrapping_sub(wrapped); // 0xFFFF_FFFF if wrapped else 0
        let res = ((same & !m) | (next & m)) as usize;

        // ---- state update ----
        // min_idx becomes child_same unless we wrapped, then it becomes child_next
        min_idx = (child_same & !m) | (child_next & m);

        // maj_idx/base increments only on boundary
        maj_idx = maj_idx.wrapping_add(wrapped);
        base_maj = base_maj_nxt;

        // keep base_majL consistent (it isn’t used by the step, but you store it in state)
        let base_step_L = wrapped << (Self::log2_items_per_line() + L);
        base_majL = base_majL.wrapping_add(base_step_L);

        (res, minor_lvl, min_idx, maj_idx, base_maj, base_majL)
    }

    #[inline(always)]
    fn step(&mut self, is_right: bool) -> usize {
        let (child_idx, minor, min_idx, maj_idx, base_maj, base_majL) = Self::step_pure(
            self.minor,
            self.min_idx,
            self.maj_idx,
            self.base_maj,
            self.base_majL,
            is_right,
        );

        self.minor = minor;
        self.min_idx = min_idx;
        self.maj_idx = maj_idx;
        self.base_maj = base_maj;
        self.base_majL = base_majL;

        child_idx
    }
}

/// Exposed pure function for use with cargo-asm
#[inline(never)]
pub fn calc_child_idx(
    minor: u32,
    min_idx: u32,
    maj_idx: u32,
    base_maj: u32,
    base_majL: u32,
    is_right: bool,
) -> (usize, u32, u32, u32, u32, u32) {
    DonnellyFullArith::<3, 64, 4>::step_pure(minor, min_idx, maj_idx, base_maj, base_majL, is_right)
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
        let mut stem_strat = DonnellyFullArith::<3, 64, 8>::new_query();
        let mut result = 0;
        input.iter().for_each(|selection| {
            result = stem_strat.get_child_idx(*selection, result);
        });

        assert_eq!(result, expected);
    }
}
