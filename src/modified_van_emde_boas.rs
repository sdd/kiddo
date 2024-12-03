use cmov::Cmov;

const CACHE_LINE_WIDTH: u32 = 64; // Intel and AMD x86-64 have 64 byte cache lines. Apple M2 has 128
const FLOAT_WIDTH: u32 = 8; // f64 = 8 bytes; f32 = 4 bytes
const ITEMS_PER_CACHE_LINE: u32 = CACHE_LINE_WIDTH / FLOAT_WIDTH; // f64 = 8 items; f32 = 16 items
const ITEMS_PER_CACHE_LINE_MASK: u32 = ITEMS_PER_CACHE_LINE - 1;
const ITEMS_PER_CACHE_LINE_MASK_INV: u32 = !ITEMS_PER_CACHE_LINE_MASK;
const LOG2_ITEMS_PER_CACHE_LINE: u32 = ITEMS_PER_CACHE_LINE.ilog2(); // f64 = 3 levels; f32 = 4 levels

#[allow(dead_code)]
#[inline]
pub(crate) fn modified_van_emde_boas_get_child_idx_v2(
    curr_idx: u32,
    is_right_child: bool,
    level: u32,
) -> u32 {
    let minor_level = level % LOG2_ITEMS_PER_CACHE_LINE;
    let maj_idx = curr_idx >> LOG2_ITEMS_PER_CACHE_LINE;
    let min_idx = curr_idx & ITEMS_PER_CACHE_LINE_MASK;

    let is_right_child = u32::from(is_right_child);

    if (minor_level + 1) == LOG2_ITEMS_PER_CACHE_LINE {
        // next level is in new cacheline / minor triangle
        let min_row_idx = min_idx - minor_level - 1;
        ((maj_idx << LOG2_ITEMS_PER_CACHE_LINE) + (min_row_idx << 1) + 1 + is_right_child)
            << LOG2_ITEMS_PER_CACHE_LINE
    } else {
        // next level is in same cacheline / minor triangle
        (maj_idx << LOG2_ITEMS_PER_CACHE_LINE) + (min_idx << 1) + 1 + is_right_child
    }
}

#[allow(dead_code)]
#[inline]
pub(crate) fn modified_van_emde_boas_get_child_idx_v2_branchless(
    curr_idx: u32,
    is_right_child: bool,
    minor_level: u32,
) -> u32 {
    let min_idx = curr_idx & ITEMS_PER_CACHE_LINE_MASK;

    let min_row_idx = min_idx.overflowing_sub(minor_level).0.overflowing_sub(1).0;

    let is_right_child = u32::from(is_right_child);
    let inc_major_level = u8::from((minor_level.overflowing_add(1).0) == LOG2_ITEMS_PER_CACHE_LINE);

    let mut result: u32 = ((curr_idx & ITEMS_PER_CACHE_LINE_MASK_INV)
        .overflowing_add(1)
        .0)
        .overflowing_add(is_right_child)
        .0;
    result.cmovnz(
        &result
            .overflowing_add(min_row_idx.overflowing_shl(1).0)
            .0
            .overflowing_shl(LOG2_ITEMS_PER_CACHE_LINE)
            .0,
        inc_major_level,
    );
    result.cmovz(
        &result.overflowing_add(min_idx.overflowing_shl(1).0).0,
        inc_major_level,
    );

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case((0, 0, false), 1)] // 1 Maj idx: 1
    #[case((0, 0, true), 2)] // 2
    #[case((1, 1, false), 3)] // 3
    #[case((1, 1, true), 4)] // 4
    #[case((2, 1, false), 5)] // 5
    #[case((2, 1, true), 6)] // 6
    #[case((3, 2, false), 8)] // 7
    #[case((3, 2, true), 16)] // 8
    #[case((4, 2, false), 24)] // 9
    #[case((4, 2, true), 32)] // 10
    #[case((5, 2, false), 40)] // 11
    #[case((5, 2, true), 48)] // 12
    #[case((6, 2, false), 56)] // 13
    #[case((6, 2, true), 64)] // 14
    #[case((8, 3, false), 9)] // 15 Maj idx: 2
    #[case((8, 3, true), 10)] // 16
    #[case((9, 4, false), 11)] // 17
    #[case((9, 4, true), 12)] // 18
    #[case((10, 4, false), 13)] // 19
    #[case((10, 4, true), 14)] // 20
    #[case((11, 5, false), 72)] // 21
    #[case((11, 5, true), 80)] // 22
    #[case((12, 5, false), 88)] // 23
    #[case((12, 5, true), 96)] // 24
    #[case((13, 5, false), 104)] // 25
    #[case((13, 5, true), 112)] // 26
    #[case((14, 5, false), 120)] // 27
    #[case((14, 5, true), 128)] // 28
    #[case((16, 3, false), 17)] // 29  Maj index: 3
    #[case((16, 3, true), 18)] // 30
    #[case((17, 4, false), 19)] // 31
    #[case((17, 4, true), 20)] // 32
    #[case((18, 4, false), 21)] // 33
    #[case((18, 4, true), 22)] // 34
    #[case((19, 5, false), 136)] // 35
    #[case((19, 5, true), 144)] // 36
    #[case((20, 5, false), 152)] // 37
    #[case((20, 5, true), 160)] // 38
    #[case((21, 5, false), 168)] // 39
    #[case((21, 5, true), 176)] // 40
    #[case((22, 5, false), 184)] // 41
    #[case((22, 5, true), 192)] // 42
    fn mod_v_e_b_get_child_idx_produces_correct_values(
        #[case] input: (u32, u32, bool),
        #[case] expected: u32,
    ) {
        let (curr_idx, level, is_right_child) = input;

        let next_idx = modified_van_emde_boas_get_child_idx_v2(curr_idx, is_right_child, level);

        assert_eq!(next_idx, expected);
    }

    #[rstest]
    #[case((0, 0, false), 1)] // 1 Maj idx: 1
    #[case((0, 0, true), 2)] // 2
    #[case((1, 1, false), 3)] // 3
    #[case((1, 1, true), 4)] // 4
    #[case((2, 1, false), 5)] // 5
    #[case((2, 1, true), 6)] // 6
    #[case((3, 2, false), 8)] // 7
    #[case((3, 2, true), 16)] // 8
    #[case((4, 2, false), 24)] // 9
    #[case((4, 2, true), 32)] // 10
    #[case((5, 2, false), 40)] // 11
    #[case((5, 2, true), 48)] // 12
    #[case((6, 2, false), 56)] // 13
    #[case((6, 2, true), 64)] // 14
    #[case((8, 0, false), 9)] // 15 Maj idx: 2
    #[case((8, 0, true), 10)] // 16
    #[case((9, 1, false), 11)] // 17
    #[case((9, 1, true), 12)] // 18
    #[case((10, 1, false), 13)] // 19
    #[case((10, 1, true), 14)] // 20
    #[case((11, 2, false), 72)] // 21
    #[case((11, 2, true), 80)] // 22
    #[case((12, 2, false), 88)] // 23
    #[case((12, 2, true), 96)] // 24
    #[case((13, 2, false), 104)] // 25
    #[case((13, 2, true), 112)] // 26
    #[case((14, 2, false), 120)] // 27
    #[case((14, 2, true), 128)] // 28
    #[case((16, 0, false), 17)] // 29  Maj index: 3
    #[case((16, 0, true), 18)] // 30
    #[case((17, 1, false), 19)] // 31
    #[case((17, 1, true), 20)] // 32
    #[case((18, 1, false), 21)] // 33
    #[case((18, 1, true), 22)] // 34
    #[case((19, 2, false), 136)] // 35
    #[case((19, 2, true), 144)] // 36
    #[case((20, 2, false), 152)] // 37
    #[case((20, 2, true), 160)] // 38
    #[case((21, 2, false), 168)] // 39
    #[case((21, 2, true), 176)] // 40
    #[case((22, 2, false), 184)] // 41
    #[case((22, 2, true), 192)] // 42
    fn mod_v_e_b_get_child_idx_branchless_produces_correct_values(
        #[case] input: (u32, u32, bool),
        #[case] expected: u32,
    ) {
        let (curr_idx, minor_level, is_right_child) = input;

        let next_idx = modified_van_emde_boas_get_child_idx_v2_branchless(
            curr_idx,
            is_right_child,
            minor_level,
        );

        assert_eq!(next_idx, expected);
    }
}
