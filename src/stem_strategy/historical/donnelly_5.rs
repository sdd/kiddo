//! V5 Donnelly Stem Strategy


use crate::donnelly_stem_layout::donnelly_get_idx_v2_branchless;
use crate::traits::Axis;
use crate::StemStrategy;
use aligned_vec::AVec;
use cmov::Cmov;

/// Donnelly layout traversal using full arithmetic (no LUT).
///
/// Internal state is reduced to two integers:
/// - `combined_idx` packs both `major_idx` and `minor_level`.
///   * Layout: high bits = major index, low LOG2_L bits = minor level.
///   * Incrementing this bumps both minor level and major index automatically.
/// - `minor_index` tracks position within the current minor triangle.
#[derive(Copy, Clone)]
pub struct DonnellyFullArithCombined<
    const L: u32,
    const CACHELINE_BYTES: u32,
    const VALUE_BYTES: u32,
> {
    /// Packed major index and minor level.
    ///
    /// `combined_idx = (major_idx << LOG2_L) | minor_level`
    combined_idx: u32,

    /// Index within the current minor triangle.
    minor_index: u32,
}

impl<const L: u32, const CL: u32, const VB: u32> StemStrategy
    for DonnellyFullArithCombined<L, CL, VB>
{
    /// Construct a new traversal state at the root of the tree.
    #[inline(always)]
    fn new_query() -> Self {
        debug_assert!(L >= 2 && L <= 8);
        Self {
            combined_idx: 0,
            minor_index: 0,
        }
    }

    /// Wrapper around a pure function so that we can easily run cargo-asm vs the inner
    #[inline(always)]
    fn get_child_idx(&mut self, is_right_child: bool, _curr_idx: usize) -> usize {
        let (new_combined, new_minor, idx) =
            Self::get_child_idx_pure(self.combined_idx, self.minor_index, is_right_child);

        self.combined_idx = new_combined;
        self.minor_index = new_minor;
        idx
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

impl<const L: u32, const CL: u32, const VB: u32> DonnellyFullArithCombined<L, CL, VB> {
    // Number of low bits used for the minor level (log2 of L).
    const LOG2_L: u32 = const_ceil_log2::<L>();

    #[inline(always)]
    pub fn get_child_idx_pure(
        combined_idx: u32, // packed: (major_idx << BITS_FOR_MINOR) | minor_level
        minor_index: u32,  // index within current minor triangle
        is_right_child: bool,
    ) -> (
        u32,   /*new_combined*/
        u32,   /*new_minor*/
        usize, /*child idx*/
    ) {
        // Number of low bits we reserve to encode [0..L-1]
        let bits_for_minor: u32 = const_ceil_log2::<L>();
        debug_assert!(L >= 2 && L <= 8);

        let right_flag: u32 = is_right_child as u32;

        // --- Decode current block/index from packed state ---
        let major_idx: u32 = combined_idx >> bits_for_minor;
        let base_major: u32 = major_idx << L; // (= major_idx * 2^L)
        let base_majorL: u32 = major_idx << (2 * L); // (= major_idx * 2^(2L))

        // --- Candidate A (stays in same block) ---
        // new_minor = minor_index + ((minor_index << 1) + 1 + right)
        let incr = ((minor_index << 1) + 1) + right_flag;
        let new_minor = minor_index + incr;
        let same_block = base_major + new_minor;

        // --- Candidate B (crosses into the next block) ---
        // next_term = (( (minor_index - (L-1)) << 1 ) + 1 + right ) << L
        // Use wrapping_sub to avoid UB on underflow during codegen; valid paths won’t underflow.
        let min_row = minor_index.wrapping_sub(L - 1);
        let next_term = ((min_row << 1) + 1 + right_flag) << L;
        let next_block = base_majorL + next_term;

        // --- Bump packed state: increments both minor level and major index on wrap ---
        let new_combined = combined_idx + 1;

        // --- Wrap detection and select ---
        // If (new_combined & ((1<<BITS_FOR_MINOR) - 1)) == 0, we just wrapped to a new block.
        let wrap_mask = (1u32 << bits_for_minor) - 1;
        let wrapped = (new_combined & wrap_mask) == 0;

        // On AArch64 this `if` commonly lowers to:
        //   ands wzr, new_combined, #wrap_mask
        //   csel result, next_block, same_block, eq
        let result = if wrapped { next_block } else { same_block };

        (new_combined, new_minor, result as usize)
    }
}

/// Small const helper: ceil(log2(x)) for x >= 2.
/// For L in [2..8], this yields bits = 1..3 as expected.
const fn const_ceil_log2<const L: u32>() -> u32 {
    // assume x >= 2
    let mut v = L - 1;
    let mut bits = 0;
    while v > 0 {
        v >>= 1;
        bits += 1;
    }
    bits
}

/// Exposed pure function for use with cargo-asm
#[inline(never)]
pub fn calc_child_idx(
    combined_idx: u32,
    minor_index: u32,
    is_right_child: bool,
) -> (u32, u32, usize) {
    DonnellyFullArithCombined::<3, 64, 4>::get_child_idx_pure(
        combined_idx,
        minor_index,
        is_right_child,
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
        let mut stem_strat = DonnellyFullArithCombined::<3, 64, 8>::new_query();
        let mut result = 0;
        input.iter().for_each(|selection| {
            result = stem_strat.get_child_idx(*selection, result);
        });

        assert_eq!(result, expected);
    }
}
