use std::ops::Not;

use cmov::Cmov;

const CACHE_LINE_WIDTH: u32 = 64; // Intel and AMD x86-64 have 64 byte cache lines. Apple M2 has 128
const FLOAT_WIDTH: u32 = 8; // f64 = 8 bytes; f32 = 4 bytes
const ITEMS_PER_CACHE_LINE: u32 = CACHE_LINE_WIDTH / FLOAT_WIDTH;  // f64 = 8 items; f32 = 16 items
const LOG2_ITEMS_PER_CACHE_LINE: u32 = ITEMS_PER_CACHE_LINE.ilog2(); // f64 = 3 levels; f32 = 4 levels
const MINOR_TRIANGLE_CHILDREN_COUNT: u32 = 1 << LOG2_ITEMS_PER_CACHE_LINE; // f64 = 8 children; f32 = 16 children

#[allow(dead_code)]
pub(crate) fn modified_van_emde_boas_get_child_idx(curr_idx: u32, is_right_child: bool, level: &mut u32, minor_level: &mut u32, major_level: &mut u32, major_level_base_idx: &mut u32, major_level_base_delta: &mut u32) -> u32 {

    // let minor_triangle_idx = curr_idx % ITEMS_PER_CACHE_LINE;
    let minor_triangle_idx = curr_idx & (ITEMS_PER_CACHE_LINE - 1); // Faster?

    let mut idx;
    if (*minor_level + 1) == LOG2_ITEMS_PER_CACHE_LINE {
        // next level is in new cacheline / minor triangle

        *major_level_base_idx = *major_level_base_idx + *major_level_base_delta; // 8
        *major_level_base_delta = *major_level_base_delta << LOG2_ITEMS_PER_CACHE_LINE; // 64

        let next_minor_triangle_root_idx = (minor_triangle_idx - LOG2_ITEMS_PER_CACHE_LINE) * MINOR_TRIANGLE_CHILDREN_COUNT << 1; // (3 - 3) * 16

        idx = *major_level_base_idx + next_minor_triangle_root_idx;
        idx = idx + u32::from(is_right_child) * ITEMS_PER_CACHE_LINE;

        *minor_level = 0;
        *major_level = *major_level + 1;
    } else {
        // next level is in same cacheline / minor triangle

        // Eytzinger layout-style, adjusted for starting at index 0 within the minor triangle rather than 1
        idx = curr_idx + (minor_triangle_idx + 1) + u32::from(is_right_child);

        *minor_level = *minor_level + 1;
    }

    *level = *level + 1;

    idx
}

#[allow(dead_code)]
#[inline]
pub(crate) fn modified_van_emde_boas_branchless_get_child_idx(mut curr_idx: u32, is_right_child: bool, level: &mut u32, minor_level: &mut u32, major_level_base_idx: &mut u32, major_level_base_delta: &mut u32) -> u32 {
    let minor_triangle_idx = curr_idx & ITEMS_PER_CACHE_LINE.overflowing_sub(1).0;

    *minor_level = minor_level.overflowing_add(1).0;
    let incrementing_major_level = u8::from(*minor_level == LOG2_ITEMS_PER_CACHE_LINE);

    // Switching to next van Emde Boas triangle if incrementing major level
    major_level_base_idx.cmovnz(&(*major_level_base_idx + *major_level_base_delta), incrementing_major_level);
    major_level_base_delta.cmovnz(&(*major_level_base_delta << LOG2_ITEMS_PER_CACHE_LINE), incrementing_major_level);

    let next_minor_triangle_root_idx = minor_triangle_idx.overflowing_sub(LOG2_ITEMS_PER_CACHE_LINE).0 << 4; // (3 - 3) * 16

    let curr_idx = &mut curr_idx;

    let next_idx_if_inc_major = (*major_level_base_idx).overflowing_add(next_minor_triangle_root_idx).0.overflowing_add(u32::from(is_right_child) << 3).0;
    curr_idx.cmovnz(&next_idx_if_inc_major, incrementing_major_level);

    minor_level.cmovnz(&0, incrementing_major_level);

    // Eytzinger layout-style, adjusted for starting at index 0 within the minor triangle rather than 1
    curr_idx.cmovz(&(*curr_idx + minor_triangle_idx + 1 + u32::from(is_right_child)), incrementing_major_level);

    *level = *level + 1;

    *curr_idx
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case((0, 0, 0, 0, 8, false), ( 1, 1, 1, 0, 8))] // 1
    #[case((0, 0, 0, 0, 8,  true), ( 2, 1, 1, 0, 8))] // 2

    #[case((1, 1, 1, 0, 8, false), ( 3, 2, 2, 0, 8))] // 3
    #[case((1, 1, 1, 0, 8,  true), ( 4, 2, 2, 0, 8))] // 4
    #[case((2, 1, 1, 0, 8, false), ( 5, 2, 2, 0, 8))] // 5
    #[case((2, 1, 1, 0, 8,  true), ( 6, 2, 2, 0, 8))] // 6

    #[case((3, 2, 2, 0, 8, false), ( 8, 3, 0, 8, 64))] // 7
    #[case((3, 2, 2, 0, 8,  true), (16, 3, 0, 8, 64))] // 8
    #[case((4, 2, 2, 0, 8, false), (24, 3, 0, 8, 64))] // 9
    #[case((4, 2, 2, 0, 8,  true), (32, 3, 0, 8, 64))] // 10
    #[case((5, 2, 2, 0, 8, false), (40, 3, 0, 8, 64))] // 11
    #[case((5, 2, 2, 0, 8,  true), (48, 3, 0, 8, 64))] // 12
    #[case((6, 2, 2, 0, 8, false), (56, 3, 0, 8, 64))] // 13
    #[case((6, 2, 2, 0, 8,  true), (64, 3, 0, 8, 64))] // 14


    #[case(( 8, 3, 0, 8, 64, false), ( 9, 4, 1, 8, 64))] // 15
    #[case(( 8, 3, 0, 8, 64,  true), (10, 4, 1, 8, 64))] // 16

    #[case(( 9, 4, 1, 8, 64, false), (11, 5, 2, 8, 64))] // 17
    #[case(( 9, 4, 1, 8, 64,  true), (12, 5, 2, 8, 64))] // 18
    #[case((10, 4, 1, 8, 64, false), (13, 5, 2, 8, 64))] // 19
    #[case((10, 4, 1, 8, 64,  true), (14, 5, 2, 8, 64))] // 20

    #[case((11, 5, 2, 8, 64, false), ( 72, 6, 0, 72, 512))] // 21
    #[case((11, 5, 2, 8, 64,  true), ( 80, 6, 0, 72, 512))] // 22
    #[case((12, 5, 2, 8, 64, false), ( 88, 6, 0, 72, 512))] // 23
    #[case((12, 5, 2, 8, 64,  true), ( 96, 6, 0, 72, 512))] // 24
    #[case((13, 5, 2, 8, 64, false), (104, 6, 0, 72, 512))] // 25
    #[case((13, 5, 2, 8, 64,  true), (112, 6, 0, 72, 512))] // 26
    #[case((14, 5, 2, 8, 64, false), (120, 6, 0, 72, 512))] // 27
    #[case((14, 5, 2, 8, 64,  true), (128, 6, 0, 72, 512))] // 28
    fn mod_v_e_b_get_child_idx_produces_correct_values(#[case] input: (u32, u32, u32, u32, u32, bool), #[case] expected: (u32, u32, u32, u32, u32)) {
        let _ = env_logger::builder().is_test(false).try_init();

        let (mut curr_idx, mut level, mut minor_level, mut major_level_base_idx, mut major_level_base_delta, is_right_child) = input;

        curr_idx = modified_van_emde_boas_branchless_get_child_idx(curr_idx, is_right_child, &mut level, &mut minor_level, &mut major_level_base_idx, &mut major_level_base_delta);

        assert_eq!(level, expected.1, "level");
        assert_eq!(minor_level, expected.2, "minor_level");
        assert_eq!(major_level_base_idx, expected.3, "major_level_base_idx");
        assert_eq!(major_level_base_delta, expected.4, "major_level_base_delta");
        assert_eq!(curr_idx, expected.0, "curr_idx");
    }
}