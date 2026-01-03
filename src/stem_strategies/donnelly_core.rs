use crate::stem_strategies::prefetch::prefetch_t1;
use std::ptr::NonNull;
// use num_traits::WrappingShl;
use crate::StemStrategy;

/// Inner implementation that holds state and core logic.
/// BS::SIZE is used at runtime via the marker trait.
#[derive(Copy, Clone, Debug)]
pub(crate) struct DonnellyCore<const CL: u32, const VB: u32, const K: usize> {
    stem_idx: u32,
    dim: usize,
    level: i32,
    minor_level: u32,
    leaf_idx: usize,
    stems_ptr: NonNull<u8>,
}

// SAFETY: NonNull<u8> is not Send or Sync, preventing DonnellyCore from being automatically
// Send & Sync. But, we can safely manually declare DonnellyCore as Send and Sync here
// because we are only using it with prefetch instructions, which do not deref the pointer
// and are guaranteed to succeed even with an invalid pointer
unsafe impl<const CL: u32, const VB: u32, const K: usize> Send for DonnellyCore<CL, VB, K> {}
unsafe impl<const CL: u32, const VB: u32, const K: usize> Sync for DonnellyCore<CL, VB, K> {}

impl<const CL: u32, const VB: u32, const K: usize> StemStrategy for DonnellyCore<CL, VB, K> {
    const ROOT_IDX: usize = 0;

    #[inline(always)]
    fn new(stems_ptr: NonNull<u8>) -> Self {
        debug_assert!(CL > VB); // item wider than cache line would break layout

        Self {
            stem_idx: Self::ROOT_IDX as u32,
            dim: 0,
            level: 0,
            minor_level: 0,
            leaf_idx: 0,
            stems_ptr,
        }
    }

    #[inline(always)]
    fn stem_idx(&self) -> usize {
        self.stem_idx as usize
    }

    #[inline(always)]
    fn leaf_idx(&self) -> usize {
        self.leaf_idx
    }

    #[inline(always)]
    fn dim(&self) -> usize {
        self.dim
    }

    #[inline(always)]
    fn level(&self) -> i32 {
        self.level
    }

    #[inline(always)]
    fn traverse(&mut self, is_right: bool) {
        let (idx, lvl) = Self::step_pure(self.stem_idx, self.minor_level, is_right, self.stems_ptr);
        self.stem_idx = idx;
        self.minor_level = lvl;

        self.level = self.level.wrapping_add(1);

        let wrap_dim_mask = 0usize.wrapping_sub((self.dim == (K - 1)) as usize);
        self.dim = self.dim.wrapping_add(1) & !wrap_dim_mask;

        self.leaf_idx = self.leaf_idx.wrapping_shl(1) | is_right as usize;
    }

    /// Used when running loop-unrolled
    ///
    /// PRECONDITIONS: assumes that
    /// * we stay within a minor triangle;
    /// * we don't hit the bottom level of the tree as a whole
    #[inline(always)]
    fn traverse_head(&mut self, is_right: bool) {
        let (idx, lvl) =
            Self::step_pure_head(self.stem_idx, self.minor_level, is_right, self.stems_ptr);
        self.stem_idx = idx;
        self.minor_level = lvl;

        self.level = self.level.wrapping_add(1);

        let wrap_dim_mask = 0usize.wrapping_sub((self.dim == (K - 1)) as usize);
        self.dim = self.dim.wrapping_add(1) & !wrap_dim_mask;

        self.leaf_idx = self.leaf_idx.wrapping_shl(1) | is_right as usize;
    }

    #[inline(always)]
    fn branch(&mut self) -> Self {
        let (left, right) = Self::both_children_pure(self.stem_idx, self.minor_level);

        // mutate self into left
        self.stem_idx = left;
        self.minor_level = (self.minor_level + 1)
            & !(0u32.wrapping_sub((self.minor_level + 1 == Self::log2_items_per_line()) as u32));

        self.level = self.level.wrapping_add(1);

        let wrap_dim_mask = 0usize.wrapping_sub((self.dim == (K - 1)) as usize);
        self.dim = self.dim.wrapping_add(1) & !wrap_dim_mask;

        self.leaf_idx = self.leaf_idx.wrapping_shl(1);

        // return right child as a new strategy
        Self {
            stem_idx: right,
            leaf_idx: self.leaf_idx | 1,
            ..*self
        }
    }

    #[inline(always)]
    fn child_indices(&self) -> (usize, usize) {
        let res = DonnellyCore::<CL, VB, K>::both_children_pure(self.stem_idx, self.minor_level);
        (res.0 as usize, res.1 as usize)
    }
}

impl<const CL: u32, const VB: u32, const K: usize> DonnellyCore<CL, VB, K> {
    /// Traverse an entire block at once
    ///
    /// - `child_idx`: index of the child block to traverse to the root of
    /// - `block_size`: block height in levels
    ///
    /// We use the same dimension for the whole block, incrementing it for the next block
    ///
    /// PRECONDITIONS:
    /// - Tree height is padded to block boundary
    /// - Traversals must be exclusively block mode or per-level, not mixed
    #[allow(unused)] // used when simd feature is on
    #[inline(always)]
    pub(crate) fn traverse_block(&mut self, child_idx: u8, block_size: u32) {
        // sanity check to help enforce preconditions
        debug_assert!(self.minor_level == 0);

        self.stem_idx = Self::step_pure_block(self.stem_idx, child_idx);

        self.level = self.level.wrapping_add(block_size as i32);

        let wrap_dim_mask = 0usize.wrapping_sub((self.dim == (K - 1)) as usize);
        self.dim = (self.dim + 1) & !wrap_dim_mask;

        self.leaf_idx = self.leaf_idx.wrapping_shl(block_size) | (child_idx as usize);
    }

    /// Used when running loop-unrolled
    ///
    /// PRECONDITIONS: assumes that
    /// * we are on the bottom level of a minor triangle
    #[inline(always)]
    pub(crate) fn traverse_tail_with_block_size(&mut self, is_right: bool, block_size: u32) {
        let (idx, lvl) = Self::step_pure_tail(
            block_size,
            self.stem_idx,
            self.minor_level,
            is_right,
            self.stems_ptr,
        );
        self.stem_idx = idx;
        self.minor_level = lvl;

        self.level = self.level.wrapping_add(1);

        let wrap_dim_mask = 0usize.wrapping_sub((self.dim == (K - 1)) as usize);
        self.dim = self.dim.wrapping_add(1) & !wrap_dim_mask;

        self.leaf_idx = self.leaf_idx.wrapping_shl(1) | is_right as usize;
    }

    #[inline(always)]
    const fn items_per_line() -> u32 {
        CL / VB
    }
    #[inline(always)]
    const fn log2_items_per_line() -> u32 {
        Self::items_per_line().ilog2()
    }
    #[inline(always)]
    const fn line_mask() -> u32 {
        Self::items_per_line() - 1
    }
    #[inline(always)]
    const fn line_mask_inv() -> u32 {
        !Self::line_mask()
    }

    #[inline(always)]
    fn step_pure(
        curr_idx: u32,
        mut minor_level: u32,
        is_right_child: bool,
        _stems_ptr: NonNull<u8>,
    ) -> (u32, u32) {
        let is_right_child = u32::from(is_right_child);

        // index into current minor triangle / cache line
        let min_idx = curr_idx & Self::line_mask();

        // column in current minor triangle
        let min_col_idx = min_idx.wrapping_sub(minor_level).wrapping_sub(1);

        let base_no_right = (curr_idx & Self::line_mask_inv()).wrapping_add(1);
        let next_prefetch_base = base_no_right
            .wrapping_add(min_col_idx.wrapping_shl(1))
            .wrapping_shl(Self::log2_items_per_line());

        let base_with_side: u32 = base_no_right.wrapping_add(is_right_child);
        let same_base = base_with_side.wrapping_add(min_idx.wrapping_shl(1));

        let next_result_base = next_prefetch_base
            .wrapping_add(is_right_child.wrapping_shl(Self::log2_items_per_line()));

        let inc_major_level = (minor_level.wrapping_add(1) == Self::log2_items_per_line()) as u32;
        let inc_major_level_mask = 0u32.wrapping_sub(inc_major_level);

        let result =
            (next_result_base & inc_major_level_mask) | (same_base & !inc_major_level_mask);

        minor_level = minor_level.wrapping_add(1);
        minor_level &= !inc_major_level_mask;

        (result, minor_level)
    }

    #[inline(always)]
    fn step_pure_head(
        curr_idx: u32,
        mut minor_level: u32,
        is_right_child: bool,
        _stems_ptr: NonNull<u8>,
    ) -> (u32, u32) {
        let is_right_child = u32::from(is_right_child);

        // index into current minor triangle / cache line
        let minor_idx = curr_idx & Self::line_mask();

        let base_no_right = (curr_idx & Self::line_mask_inv()).wrapping_add(1);

        let base_with_side: u32 = base_no_right.wrapping_add(is_right_child);
        let result = base_with_side.wrapping_add(minor_idx.wrapping_shl(1));

        minor_level = minor_level.wrapping_add(1);
        // println!("is_right_child: {is_right_child}, min_idx: {minor_idx}, base_no_right: {base_no_right}, base_with_side: {base_with_side}, next stem_idx: {result}");

        (result, minor_level)
    }

    #[inline(always)]
    fn step_pure_tail(
        block_size: u32,
        curr_idx: u32,
        mut minor_level: u32,
        is_right_child: bool,
        stems_ptr: NonNull<u8>,
    ) -> (u32, u32) {
        let is_right_child = u32::from(is_right_child);

        // index into current minor triangle / cache line
        let min_idx = curr_idx & Self::line_mask();

        // row in current minor triangle
        let min_row_idx = min_idx.wrapping_sub(minor_level).wrapping_sub(1);

        let base_no_right = (curr_idx & Self::line_mask_inv()).wrapping_add(1);
        let next_prefetch_base = base_no_right
            .wrapping_add(min_row_idx.wrapping_shl(1))
            .wrapping_shl(Self::log2_items_per_line());

        let result = next_prefetch_base
            .wrapping_add(is_right_child.wrapping_shl(Self::log2_items_per_line()));

        // Prefetch result? Not much point, it's likely gonna be requested within 1 cycle
        // unsafe {
        //     let nxt_ptr = stems_ptr
        //         .as_ptr()
        //         .add((result * VB) as usize);
        //     prefetch_t0(nxt_ptr);
        // }

        // Prefetch deeper-level 8 base ptrs to L2
        let next_base_no_right = (result & Self::line_mask_inv()).wrapping_add(7);
        let next_next_prefetch_base = next_base_no_right.wrapping_shl(Self::log2_items_per_line());

        Self::prefetch_next_base(
            stems_ptr,
            next_next_prefetch_base,
            2u32.pow(block_size) as usize,
        );

        // println!("is_right_child: {is_right_child}, min_idx: {min_idx}, min_row_idx: {min_row_idx}, base_no_right: {base_no_right}, next_prefetch_base: {next_prefetch_base}, next stem_idx: {result}");
        // println!("next_next_prefetch_base: {next_next_prefetch_base} -> {}", next_next_prefetch_base + 128);

        minor_level = 0;

        (result, minor_level)
    }

    #[inline(always)]
    fn step_pure_block(curr_idx: u32, child_idx: u8) -> u32 {
        curr_idx
            .wrapping_add(1)
            .wrapping_shl(Self::log2_items_per_line())
            .wrapping_add((child_idx as u32).wrapping_shl(Self::log2_items_per_line()))
    }

    #[inline(always)]
    fn prefetch_next_base(stems_ptr: NonNull<u8>, next_base: u32, cache_line_count: usize) {
        #[cfg(target_arch = "x86_64")]
        const BYTES_PER_LINE: usize = 64;

        #[cfg(target_arch = "aarch64")]
        const BYTES_PER_LINE: usize = 64; // 64 for most ARM, 128 for Apple M-series

        let base_ptr = unsafe { stems_ptr.as_ptr().add((next_base as usize) * VB as usize) };

        for i in 0..cache_line_count {
            let ptr = unsafe { base_ptr.add(i * BYTES_PER_LINE) };
            unsafe { prefetch_t1(ptr) };
        }
    }

    /// Two-children step in one pass (left=false, right=true).
    /// Advances minor_level once; does NOT change curr_idx (so caller can choose a child later).
    #[inline(always)]
    pub(crate) fn both_children_pure(curr_idx: u32, minor_level: u32) -> (u32, u32) {
        // precompute pieces identical to step_pure
        let line_mask = Self::line_mask();
        let line_mask_inv = Self::line_mask_inv();
        let l2_items = Self::log2_items_per_line();

        let min_idx = curr_idx & line_mask;
        let min_row_idx = min_idx.wrapping_sub(minor_level).wrapping_sub(1);

        let inc_major = (minor_level.wrapping_add(1) == l2_items) as u32;
        let inc_mask = 0u32.wrapping_sub(inc_major);

        let base_no_right = (curr_idx & line_mask_inv).wrapping_add(1);

        // same-block left/right
        let same_left = base_no_right.wrapping_add(min_idx.wrapping_shl(1));
        let same_right = same_left.wrapping_add(1);

        // next-block left/right (note: add right after shift by L)
        let next_pre = base_no_right.wrapping_add(min_row_idx.wrapping_shl(1));
        let next_left = next_pre.wrapping_shl(l2_items);
        let next_right = next_left.wrapping_add(1u32.wrapping_shl(l2_items));

        // masked select between same/next for both children
        let left = (same_left & !inc_mask) | (next_left & inc_mask);
        let right = (same_right & !inc_mask) | (next_right & inc_mask);

        (left, right)
    }
}

/// Exposed pure function for use with cargo-asm
#[inline(never)]
pub fn calc_child_idx_hook(
    curr_idx: u32,
    minor_index: u32,
    is_right_child: bool,
    stems_ptr: NonNull<u8>,
) -> (u32, u32) {
    DonnellyCore::<64, 8, 3>::step_pure(curr_idx, minor_index, is_right_child, stems_ptr)
}

/// Exposed pure function for use with cargo-asm
#[inline(never)]
pub fn both_children_pure_hook(curr_idx: u32, minor_index: u32) -> (u32, u32) {
    DonnellyCore::<64, 8, 3>::both_children_pure(curr_idx, minor_index)
}

/// Exposed pure function for use with cargo-asm
#[inline(never)]
pub fn test_traverse_hook(is_right_child: bool, stems: *mut u8) -> usize {
    let stems_ptr = NonNull::new(stems).unwrap();

    let mut stem_strat = DonnellyCore::<64, 8, 3>::new(stems_ptr);

    stem_strat.traverse(is_right_child);
    stem_strat.traverse(!is_right_child);
    stem_strat.traverse(is_right_child);

    stem_strat.stem_idx()
}

#[cfg(test)]
mod tests {
    use super::*;
    use aligned_vec::avec;
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
    fn donnelly_core_get_child_idx_produces_correct_values(
        #[case] input: Vec<bool>,
        #[case] expected: usize,
    ) {
        let stems = avec![f64::INFINITY; 9];
        let stems_ptr = NonNull::new(stems.as_ptr() as *mut u8).unwrap();

        let mut stem_strat = DonnellyCore::<64, 8, 3>::new(stems_ptr);
        let mut result = 0;
        input.iter().for_each(|selection| {
            stem_strat.traverse(*selection);
            result = stem_strat.stem_idx();
        });

        assert_eq!(result, expected);
    }

    #[rstest]
    #[case(vec![], (1, 2))]
    #[case(vec![false], (3, 4))] // 1 Maj idx: 1
    #[case(vec![true], (5, 6))] // 2
    #[case(vec![false, false], (8, 16))] // 3
    #[case(vec![false, true], (24, 32))] // 4
    #[case(vec![true, false], (40, 48))] // 5
    #[case(vec![true, true], (56, 64))] // 6
    #[case(vec![false, false, false], (9, 10))] // 7
    #[case(vec![false, false, true], (17, 18))] // 8
    #[case(vec![false, true, false], (25, 26))] // 9
    #[case(vec![false, true, true], (33, 34))] // 10
    #[case(vec![true, false, false], (41, 42))] // 11
    #[case(vec![true, false, true], (49, 50))] // 12
    #[case(vec![true, true, false], (57, 58))] // 13
    #[case(vec![true, true, true], (65, 66))] // 14
    #[case(vec![false, false, false, false], (11, 12))] // 15 Maj idx: 2
    #[case(vec![false, false, false, true], (13, 14))] // 16
    #[case(vec![false, false, false, false, false], (72, 80))] // 17
    #[case(vec![false, false, false, false, true], (88, 96))] // 18
    #[case(vec![false, false, false, true, false], (104, 112))] // 19
    #[case(vec![false, false, false, true, true], (120, 128))] // 20
    #[case(vec![false, false, false, false, false, false], (73, 74))] // 21
    #[case(vec![false, false, false, false, false, true], (81, 82))] // 22
    #[case(vec![false, false, false, false, true, false], (89, 90))] // 23
    #[case(vec![false, false, false, false, true, true], (97, 98))] // 24
    #[case(vec![false, false, false, true, false, false], (105, 106))] // 25
    #[case(vec![false, false, false, true, false, true], (113, 114))] // 26
    #[case(vec![false, false, false, true, true, false], (121, 122))] // 27
    #[case(vec![false, false, false, true, true, true], (129, 130))] // 28
    #[case(vec![false, false, true, false], (19, 20))] // 29  Maj index: 3
    #[case(vec![false, false, true, true], (21, 22))] // 30
    #[case(vec![false, false, true, false, false], (136, 144))] // 31
    #[case(vec![false, false, true, false, true], (152, 160))] // 32
    #[case(vec![false, false, true, true, false], (168, 176))] // 33
    #[case(vec![false, false, true, true, true], (184, 192))] // 34
    #[case(vec![false, false, true, false, false, false], (137, 138))] // 35
    #[case(vec![false, false, true, false, false, true], (145, 146))] // 36
    #[case(vec![false, false, true, false, true, false], (153, 154))] // 37
    #[case(vec![false, false, true, false, true, true], (161, 162))] // 38
    #[case(vec![false, false, true, true, false, false], (169, 170))] // 39
    #[case(vec![false, false, true, true, false, true], (177, 178))] // 40
    #[case(vec![false, false, true, true, true, false], (185, 186))] // 41
    #[case(vec![false, false, true, true, true, true], (193, 194))] // 42
    fn donnelly_core_get_both_child_idxs_produces_correct_values(
        #[case] input: Vec<bool>,
        #[case] expected: (usize, usize),
    ) {
        let stems = avec![f64::INFINITY; 9];
        let stems_ptr = NonNull::new(stems.as_ptr() as *mut u8).unwrap();

        let mut stem_strat = DonnellyCore::<64, 8, 3>::new(stems_ptr);
        // let mut stem_strat = Donnelly::<3, 64, 4, 4>::new();

        // let last = input.last().unwrap();
        input.iter().for_each(|selection| {
            stem_strat.branch_relative(*selection);
        });

        let results = stem_strat.split();
        let result = (results.0.stem_idx(), results.1.stem_idx());

        assert_eq!(result, expected);
    }

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
    fn donnelly_core_get_child_idx_unrolled_produces_correct_values(
        #[case] input: Vec<bool>,
        #[case] expected: usize,
    ) {
        let stems = avec![f64::INFINITY; 9];
        let stems_ptr = NonNull::new(stems.as_ptr() as *mut u8).unwrap();

        let mut stem_strat = DonnellyCore::<64, 8, 3>::new(stems_ptr);
        let mut result = 0;
        let mut minor_tri_idx = 0;
        input.iter().for_each(|selection| {
            if minor_tri_idx == 2 {
                stem_strat.traverse_tail(*selection);
                minor_tri_idx = 0;
            } else {
                minor_tri_idx += 1;
                stem_strat.traverse_head(*selection);
            }

            result = stem_strat.stem_idx();
        });

        assert_eq!(result, expected);
    }

    #[rstest]
    #[case(vec![], 0)]
    #[case(vec![0], 8)] // 1
    #[case(vec![1], 16)] // 2
    #[case(vec![2], 24)] // 3
    #[case(vec![3], 32)] // 4
    #[case(vec![4], 40)] // 5
    #[case(vec![5], 48)] // 6
    #[case(vec![6], 56)] // 7
    #[case(vec![7], 64)] // 8
    #[case(vec![0, 0], 72)] // 9
    #[case(vec![0, 1], 80)] // 10
    #[case(vec![0, 2], 88)] // 11
    #[case(vec![0, 3], 96)] // 12
    #[case(vec![0, 4], 104)] // 13
    #[case(vec![0, 5], 112)] // 14
    #[case(vec![0, 6], 120)] // 15
    #[case(vec![0, 7], 128)] // 16
    #[case(vec![1, 0], 136)] // 17
    #[case(vec![1, 1], 144)] // 18
    #[case(vec![1, 2], 152)] // 19
    #[case(vec![1, 3], 160)] // 20
    #[case(vec![1, 4], 168)] // 21
    #[case(vec![1, 5], 176)] // 22
    #[case(vec![1, 6], 184)] // 23
    #[case(vec![1, 7], 192)] // 24
    fn donnelly_core_traverse_block_produces_correct_values(
        #[case] input: Vec<u8>,
        #[case] expected: usize,
    ) {
        let stems = avec![f64::INFINITY; 9];
        let stems_ptr = NonNull::new(stems.as_ptr() as *mut u8).unwrap();

        let mut stem_strat = DonnellyCore::<64, 8, 3>::new(stems_ptr);
        let mut result = 0;
        input.iter().for_each(|selection| {
            stem_strat.traverse_block(*selection, 3);
            result = stem_strat.stem_idx();
        });

        assert_eq!(result, expected);
    }
}
