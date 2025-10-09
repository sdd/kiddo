use cmov::Cmov;
use num_traits::ops::overflowing::OverflowingAdd;

pub const CACHE_LINE_WIDTH: u32 = 64; // Intel and AMD x86-64 have 64 byte cache lines. Apple M2 has 128
pub const FLOAT_WIDTH: u32 = 8; // f64 = 8 bytes; f32 = 4 bytes
pub const ITEMS_PER_CACHE_LINE: u32 = CACHE_LINE_WIDTH / FLOAT_WIDTH; // f64 = 8 items; f32 = 16 items
pub const ITEMS_PER_CACHE_LINE_MASK: u32 = ITEMS_PER_CACHE_LINE - 1;
pub const ITEMS_PER_CACHE_LINE_MASK_INV: u32 = !ITEMS_PER_CACHE_LINE_MASK;
pub const LOG2_ITEMS_PER_CACHE_LINE: u32 = ITEMS_PER_CACHE_LINE.ilog2(); // f64 = 3 levels; f32 = 4 levels

#[allow(dead_code)]
#[cfg_attr(not(feature = "no_inline"), inline)]
#[cfg_attr(feature = "no_inline", inline(never))]
pub fn donnelly_get_idx_v2(curr_idx: u32, is_right_child: bool, level: u32) -> u32 {
    let minor_level = level % LOG2_ITEMS_PER_CACHE_LINE;
    let maj_idx = curr_idx >> LOG2_ITEMS_PER_CACHE_LINE;
    let min_idx = curr_idx & ITEMS_PER_CACHE_LINE_MASK;

    let is_right_child = u32::from(is_right_child);

    if (minor_level + 1) == LOG2_ITEMS_PER_CACHE_LINE {
        // next level is in new cacheline / minor triangle
        let min_row_idx = min_idx - minor_level - 2;
        (((maj_idx << LOG2_ITEMS_PER_CACHE_LINE) + (min_row_idx << 1) + 1 + is_right_child)
            << LOG2_ITEMS_PER_CACHE_LINE) + 1
    } else {
        // next level is in same cacheline / minor triangle
        (maj_idx << LOG2_ITEMS_PER_CACHE_LINE) + (min_idx << 1) + is_right_child
    }
}

#[allow(dead_code)]
// #[cfg_attr(not(feature = "no_inline"), inline)]
#[cfg_attr(feature = "no_inline", inline(never))]
#[inline(never)]
pub fn donnelly_get_idx_v2_branchless(
    curr_idx: u32,
    is_right_child: bool,
    minor_level: u32,
) -> u32 {
    let min_idx = curr_idx & ITEMS_PER_CACHE_LINE_MASK;

    let min_row_idx = min_idx.overflowing_sub(minor_level).0.overflowing_sub(2).0;

    let is_right_child = u32::from(is_right_child);
    let inc_major_level = u8::from((minor_level.overflowing_add(1).0) == LOG2_ITEMS_PER_CACHE_LINE);

    let mut result: u32 = (curr_idx & ITEMS_PER_CACHE_LINE_MASK_INV)
        // .overflowing_add(1)
        // .0
        .overflowing_add(is_right_child)
        .0;
    result.cmovnz(
        &result
            .overflowing_add(min_row_idx.overflowing_shl(1).0)
            .0
            .overflowing_add(1)
            .0
            .overflowing_shl(LOG2_ITEMS_PER_CACHE_LINE)
            .0
            .overflowing_add(1)
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
    #[case((1, 0, false), 2)] // 1 Maj idx: 1
    #[case((1, 0, true), 3)] // 2
    #[case((2, 1, false), 4)] // 3
    #[case((2, 1, true), 5)] // 4
    #[case((3, 1, false), 6)] // 5
    #[case((3, 1, true), 7)] // 6
    #[case((4, 2, false), 9)] // 7
    #[case((4, 2, true), 17)] // 8
    #[case((5, 2, false), 25)] // 9
    #[case((5, 2, true), 33)] // 10
    #[case((6, 2, false), 41)] // 11
    #[case((6, 2, true), 49)] // 12
    #[case((7, 2, false), 57)] // 13
    #[case((7, 2, true), 65)] // 14
    #[case((9, 3, false), 10)] // 15 Maj idx: 2
    #[case((9, 3, true), 11)] // 16
    #[case((10, 4, false), 12)] // 17
    #[case((10, 4, true), 13)] // 18
    #[case((11, 4, false), 14)] // 19
    #[case((11, 4, true), 15)] // 20
    #[case((12, 5, false), 73)] // 21
    #[case((12, 5, true), 81)] // 22
    #[case((13, 5, false), 89)] // 23
    #[case((13, 5, true), 97)] // 24
    #[case((14, 5, false), 105)] // 25
    #[case((14, 5, true), 113)] // 26
    #[case((15, 5, false), 121)] // 27
    #[case((15, 5, true), 129)] // 28
    #[case((17, 3, false), 18)] // 29  Maj index: 3
    #[case((17, 3, true), 19)] // 30
    #[case((18, 4, false), 20)] // 31
    #[case((18, 4, true), 21)] // 32
    #[case((19, 4, false), 22)] // 33
    #[case((19, 4, true), 23)] // 34
    #[case((20, 5, false), 137)] // 35
    #[case((20, 5, true), 145)] // 36
    #[case((21, 5, false), 153)] // 37
    #[case((21, 5, true), 161)] // 38
    #[case((22, 5, false), 169)] // 39
    #[case((22, 5, true), 177)] // 40
    #[case((23, 5, false), 185)] // 41
    #[case((23, 5, true), 193)] // 42
    fn donnelly_v2_get_child_idx_produces_correct_values(
        #[case] input: (u32, u32, bool),
        #[case] expected: u32,
    ) {
        let (curr_idx, level, is_right_child) = input;

        let next_idx = donnelly_get_idx_v2(curr_idx, is_right_child, level);

        assert_eq!(next_idx, expected);
    }

    #[rstest]
    #[case((1, 0, false), 2)] // 1 Maj idx: 1
    #[case((1, 0, true), 3)] // 2
    #[case((2, 1, false), 4)] // 3
    #[case((2, 1, true), 5)] // 4
    #[case((3, 1, false), 6)] // 5
    #[case((3, 1, true), 7)] // 6
    #[case((4, 2, false), 9)] // 7
    #[case((4, 2, true), 17)] // 8
    #[case((5, 2, false), 25)] // 9
    #[case((5, 2, true), 33)] // 10
    #[case((6, 2, false), 41)] // 11
    #[case((6, 2, true), 49)] // 12
    #[case((7, 2, false), 57)] // 13
    #[case((7, 2, true), 65)] // 14
    #[case((9, 3, false), 10)] // 15 Maj idx: 2
    #[case((9, 3, true), 11)] // 16
    #[case((10, 4, false), 12)] // 17
    #[case((10, 4, true), 13)] // 18
    #[case((11, 4, false), 14)] // 19
    #[case((11, 4, true), 15)] // 20
    #[case((12, 5, false), 73)] // 21
    #[case((12, 5, true), 81)] // 22
    #[case((13, 5, false), 89)] // 23
    #[case((13, 5, true), 97)] // 24
    #[case((14, 5, false), 105)] // 25
    #[case((14, 5, true), 113)] // 26
    #[case((15, 5, false), 121)] // 27
    #[case((15, 5, true), 129)] // 28
    #[case((17, 3, false), 18)] // 29  Maj index: 3
    #[case((17, 3, true), 19)] // 30
    #[case((18, 4, false), 20)] // 31
    #[case((18, 4, true), 21)] // 32
    #[case((19, 4, false), 22)] // 33
    #[case((19, 4, true), 23)] // 34
    #[case((20, 5, false), 137)] // 35
    #[case((20, 5, true), 145)] // 36
    #[case((21, 5, false), 153)] // 37
    #[case((21, 5, true), 161)] // 38
    #[case((22, 5, false), 169)] // 39
    #[case((22, 5, true), 177)] // 40
    #[case((23, 5, false), 185)] // 41
    #[case((23, 5, true), 193)] // 42
    fn donnelly_v2_branchless_get_child_idx_branchless_produces_correct_values(
        #[case] input: (u32, u32, bool),
        #[case] expected: u32,
    ) {
        let (curr_idx, minor_level, is_right_child) = input;

        let next_idx = donnelly_get_idx_v2_branchless(curr_idx, is_right_child, minor_level);

        assert_eq!(next_idx, expected);
    }
}
