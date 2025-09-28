use crate::traits::Axis;
use crate::StemStrategy;
use aligned_vec::AVec;
use cmov::Cmov;
use crate::donnelly_stem_layout::donnelly_get_idx_v2_branchless;

/// Donnelly v2 Strategy
///
/// A modification of the van Emde Boas layout, improved
/// for better cache sympathy. The first slot per cache line
/// is unused rather than the last one. This allows us to
/// use the simple Eytzinger traversal logic within triangles.
/// - L:     levels per block
/// - CL:    cache line width in bytes (64 or 128)
/// - VB:    value width in bytes (4 or 8)
#[derive(Copy, Clone)]
pub struct Donnelly<const L: u32, const CL: u32, const VB: u32> {
    minor_level: u32,
    curr_idx: u32,
}

impl<const L: u32, const CL: u32, const VB: u32> StemStrategy for Donnelly<L, CL, VB> {
    fn get_initial_idx() -> usize {
        1
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


    fn new_query() -> Self {
        // L must be in {3, 4, 5,...}. We rely on it being a small constant.
        debug_assert!(L >= 2 && L <= 8);

        // Won't work if we're using items wider than a cache line
        debug_assert!(CL > VB);

        Self {
            minor_level: 0,
            curr_idx: 1,
        }
    }

    fn get_child_idx(&mut self, is_right_child: bool, curr_idx: usize) -> usize {
        let (curr_idx, minor_level) =
            Self::step_pure(
                self.curr_idx,
                self.minor_level,
                is_right_child
            );

        self.minor_level = minor_level;
        self.curr_idx = curr_idx;

        curr_idx as usize
    }

    #[inline(always)]
    fn get_both_child_idx(&mut self, _curr_idx: usize) -> (usize, usize) {
        let (l, r) = Self::both_children_pure(self.curr_idx, self.minor_level);
        (l as usize, r as usize)
    }

    #[inline(always)]
    fn get_closer_and_further_child_idx(
        &mut self,
        curr_idx: usize,
        is_right_child: bool,
    ) -> (usize, usize) {
        let (l, r) = self.get_both_child_idx(curr_idx);
        if is_right_child { (r, l) } else { (l, r) }
    }
}

impl<const L: u32, const CL: u32, const VB: u32> Donnelly<L, CL, VB> {
    // ---- layout helpers ----
    #[inline(always)]
    const fn items_per_line() -> u32 { CL / VB }
    #[inline(always)]
    const fn log2_items_per_line() -> u32 { Self::items_per_line().ilog2() }
    #[inline(always)]
    const fn line_mask() -> u32 { Self::items_per_line() - 1 }
    #[inline(always)]
    const fn line_mask_inv() -> u32 { !Self::line_mask() }

    #[inline(always)]
    fn step_pure(
        mut curr_idx: u32,
        mut minor_level: u32,
        is_right_child: bool,
    ) -> (u32, u32) {
        debug_assert!(L >= 2 && L <= 8);
        let is_right_child = u32::from(is_right_child);

        // index into current minor triangle / cache line
        let min_idx = curr_idx & Self::line_mask();

        // column index (only relevant on last row of minor triangle)
        let col_idx = min_idx.wrapping_sub(Self::line_mask().wrapping_shr(1));

        // boolean indicating if we're transitioning to the next minor triangle
        let inc_major_level = (minor_level.wrapping_add(1) == Self::log2_items_per_line()) as u32;
        let inc_major_level_mask = 0u32.wrapping_sub(inc_major_level);

        curr_idx = (curr_idx & Self::line_mask_inv())
            .wrapping_add(1)
            .wrapping_add(is_right_child);

        curr_idx = (curr_idx
            .wrapping_add(col_idx.wrapping_shl(1))
            .wrapping_shl(Self::log2_items_per_line())
            & inc_major_level_mask) | (curr_idx.wrapping_add(min_idx.wrapping_shl(1)) & !inc_major_level_mask);

        minor_level += 1;
        minor_level = minor_level & !inc_major_level_mask;

        (curr_idx, minor_level)
    }

    /// Two-children step in one pass (left=false, right=true).
    /// Advances minor_level once; does NOT change curr_idx (so caller can choose a child later).
    #[inline(always)]
    fn both_children_pure(curr_idx: u32, minor_level: u32) -> (u32, u32) {
        // precompute pieces identical to step_pure
        let line_mask      = Self::line_mask();
        let line_mask_inv  = Self::line_mask_inv();
        let l2_items       = Self::log2_items_per_line();

        let min_idx        = curr_idx & line_mask;
        let min_row_idx    = min_idx.wrapping_sub(minor_level).wrapping_sub(1);

        let inc_major      = (minor_level.wrapping_add(1) == l2_items) as u32;
        let inc_mask       = 0u32.wrapping_sub(inc_major);

        let base_no_right  = (curr_idx & line_mask_inv).wrapping_add(1);

        // same-block left/right
        let same_left      = base_no_right.wrapping_add(min_idx << 1);
        let same_right     = same_left.wrapping_add(1);

        // next-block left/right (note: add right after shift by L)
        let next_pre       = base_no_right.wrapping_add(min_row_idx << 1);
        let next_left      = next_pre.wrapping_shl(l2_items);
        let next_right     = next_left.wrapping_add(1 << l2_items);

        // masked select between same/next for both children
        let left  = (same_left  & !inc_mask) | (next_left  & inc_mask);
        let right = (same_right & !inc_mask) | (next_right & inc_mask);

        (left, right)
    }


}

/// Exposed pure function for use with cargo-asm
#[inline(never)]
pub fn calc_child_idx(
    curr_idx: u32,
    minor_index: u32,
    is_right_child: bool,
) -> (u32, u32) {
    Donnelly::<3, 64, 8>::step_pure(
        curr_idx, minor_index, is_right_child,
    )
}

/// Exposed pure function for use with cargo-asm
#[inline(never)]
pub fn calc_both_child_idx(
    curr_idx: u32,
    minor_index: u32,
) -> (u32, u32) {
    Donnelly::<3, 64, 8>::both_children_pure(
        curr_idx, minor_index,
    )
}


#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[test]
    fn donnelly_v2_get_child_idx_produces_correct_values_0() {
        let input = vec![true, true, true];
        let expected = 64;

        let mut stem_strat = Donnelly::<3, 64, 8>::new_query();
        let mut result = 0;
        input.iter().for_each(|selection| {
            result = stem_strat.get_child_idx(*selection, result);
        });

        assert_eq!(result, expected);
    }

    #[rstest]
    #[case(vec![], 1)] // Test case 1
    #[case(vec![false], 2)] // 2 Maj idx: 1
    #[case(vec![true], 3)] // 3
    #[case(vec![false, false], 4)] // 4
    #[case(vec![false, true], 5)] // 5
    #[case(vec![true, false], 6)] // 6
    #[case(vec![true, true], 7)] // 7
    #[case(vec![false, false, false], 9)] // 8
    #[case(vec![false, false, true], 17)] // 9
    #[case(vec![false, true, false], 25)] // 10
    #[case(vec![false, true, true], 33)] // 11
    #[case(vec![true, false, false], 41)] // 12
    #[case(vec![true, false, true], 49)] // 13
    #[case(vec![true, true, false], 57)] // 14
    #[case(vec![true, true, true], 65)] // 15
    #[case(vec![false, false, false, false], 10)] // 16 Maj idx: 2
    #[case(vec![false, false, false, true], 11)] // 17
    #[case(vec![false, false, false, false, false], 12)] // 18
    #[case(vec![false, false, false, false, true], 13)] // 19
    #[case(vec![false, false, false, true, false], 14)] // 20
    #[case(vec![false, false, false, true, true], 15)] // 21
    #[case(vec![false, false, false, false, false, false], 73)] // 22
    #[case(vec![false, false, false, false, false, true], 81)] // 23
    #[case(vec![false, false, false, false, true, false], 89)] // 24
    #[case(vec![false, false, false, false, true, true], 97)] // 25
    #[case(vec![false, false, false, true, false, false], 105)] // 26
    #[case(vec![false, false, false, true, false, true], 113)] // 27
    #[case(vec![false, false, false, true, true, false], 121)] // 28
    #[case(vec![false, false, false, true, true, true], 129)] // 29
    #[case(vec![false, false, true, false], 18)] // 30  Maj index: 3
    #[case(vec![false, false, true, true], 19)] // 31
    #[case(vec![false, false, true, false, false], 20)] // 32
    #[case(vec![false, false, true, false, true], 21)] // 33
    #[case(vec![false, false, true, true, false], 22)] // 34
    #[case(vec![false, false, true, true, true], 23)] // 35
    #[case(vec![false, false, true, false, false, false], 137)] // 36
    #[case(vec![false, false, true, false, false, true], 145)] // 37
    #[case(vec![false, false, true, false, true, false], 153)] // 38
    #[case(vec![false, false, true, false, true, true], 161)] // 39
    #[case(vec![false, false, true, true, false, false], 169)] // 40
    #[case(vec![false, false, true, true, false, true], 177)] // 41
    #[case(vec![false, false, true, true, true, false], 185)] // 42
    #[case(vec![false, false, true, true, true, true], 193)] // 43
    fn donnelly_v2_get_child_idx_produces_correct_values(
        #[case] input: Vec<bool>,
        #[case] expected: usize,
    ) {
        let mut stem_strat = Donnelly::<3, 64, 8>::new_query();
        let mut result = 0;
        input.iter().for_each(|selection| {
            result = stem_strat.get_child_idx(*selection, result);
        });

        assert_eq!(result, expected);
    }
}