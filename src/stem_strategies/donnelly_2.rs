use crate::donnelly_stem_layout::donnelly_get_idx_v2_branchless;
use crate::traits::Axis;
use crate::StemStrategy;
use aligned_vec::AVec;
use cmov::Cmov;

// x86, f64 (64B lines, 3 levels per block)
pub type Donnelly3X86F64 = Donnelly<3, 64, 8>;

// x86, f32 (64B lines, 4 levels per block)
pub type Donnelly4X86F32 = Donnelly<4, 64, 4>;

// Apple M2+, f64 (128B lines, 4 levels per block)
pub type Donnelly4M2F64 = Donnelly<4, 128, 8>;

// Apple M2+, f32 (128B lines, 5 levels per block)
pub type Donnelly5M2F32 = Donnelly<5, 128, 4>;

/// Donnelly Strategy
///
/// A modification of the van Emde Boas layout, improved
/// for better cache sympathy.
/// - L:     levels per block
/// - CL:    cache line width in bytes (64 or 128)
/// - VB:    value width in bytes (4 or 8)
#[derive(Copy, Clone)]
pub struct Donnelly<const L: u32, const CL: u32, const VB: u32> {
    minor_level: u32,
    min_idx: u32,
    maj_idx: u32,
    base_maj: u32,
}

impl<const L: u32, const CL: u32, const VB: u32> Donnelly<L, CL, VB> {
    #[inline(always)]
    const fn items_per_line() -> u32 {
        CL / VB
    }

    #[inline(always)]
    const fn line_mask() -> u32 {
        Self::items_per_line() - 1
    }
}

impl<const L: u32, const CL: u32, const VB: u32> StemStrategy for Donnelly<L, CL, VB> {
    fn new_query() -> Self {
        // L must be in {3, 4, 5,...}. We rely on it being a small constant.
        debug_assert!(L >= 2 && L <= 8);

        // Won't work if we're using items wider than a cache line
        debug_assert!(CL > VB);

        Self {
            minor_level: 0,
            min_idx: 0,
            maj_idx: 0,
            base_maj: 0,
        }
    }

    fn get_child_idx(&mut self, is_right_child: bool, curr_idx: usize) -> usize {
        let result =
            donnelly_get_idx_v2_branchless(curr_idx as u32, is_right_child, self.minor_level);

        self.minor_level += 1;
        self.minor_level.cmovnz(&0, u8::from(self.minor_level == 3));

        result as usize
    }

    fn get_both_child_idx(&mut self, curr_idx: usize) -> (usize, usize) {
        let left = donnelly_get_idx_v2_branchless(curr_idx as u32, false, self.minor_level);

        let right = donnelly_get_idx_v2_branchless(curr_idx as u32, true, self.minor_level);

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
