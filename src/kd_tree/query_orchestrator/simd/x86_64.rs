//! x86_64 AVX2 SIMD implementations for query orchestration pruning

/// SIMD prune block for f64 values (AVX2)
///
/// Compares 8 rd_values against max_dist in parallel and returns a bitmask
/// indicating which siblings should be explored.
#[cfg(target_feature = "avx2")]
#[inline(always)]
pub(crate) fn simd_prune_block_f64(rd_values: &[f64; 8], max_dist: f64, sibling_mask: u8) -> u8 {
    unsafe {
        use std::arch::x86_64::*;

        let max_dist_vec = _mm256_set1_pd(max_dist);
        let rd_low = _mm256_loadu_pd(rd_values.as_ptr());
        let rd_high = _mm256_loadu_pd(rd_values.as_ptr().add(4));

        let cmp_low = _mm256_cmp_pd(rd_low, max_dist_vec, _CMP_LE_OQ);
        let cmp_high = _mm256_cmp_pd(rd_high, max_dist_vec, _CMP_LE_OQ);

        let mask_low = _mm256_movemask_pd(cmp_low) as u8;
        let mask_high = _mm256_movemask_pd(cmp_high) as u8;

        let mask = mask_low | (mask_high << 4);

        // We need to account for the fact that the ordering of stem pivots within
        // a 3-block is triangular. e.g.:
        //
        //                               #0 (0.5)
        //            #1 (0.25)                             #2 (0.75)
        // #3 (0.125)          #4 (0.375)        #5 (0.625)          #6 (0.875)
        //
        //  Child Idx |      Val Range      |  Pivot Idx
        //     0      |           x < 0.125 |      3
        //     1      |  0.125 <= x < 0.250 |      1
        //     2      |  0.250 <= x < 0.375 |      4
        //     3      |  0.375 <= x < 0.500 |      0
        //     4      |  0.500 <= x < 0.625 |      5
        //     5      |  0.625 <= x < 0.750 |      2
        //     6      |  0.750 <= x < 0.875 |      6
        //     7      |  0.875 <= x         |
        //
        // Map pivot idx to child idx by permuting the mask
        //
        // Source: https://programming.sirrida.de/calcperm.php
        // Config: LSB First, Origin 0, Base 10, indices refer to source bits
        // Input: "7 3 1 4 0 5 2 6  # bswap"
        // allow all
        // Method used: Bit Group Moving
        // let permuted_mask = (mask & 0x20)
        //     | ((mask & 0x42) << 1)
        //     | ((mask & 0x05) << 4)
        //     | ((mask & 0x80) >> 7)
        //     | ((mask & 0x08) >> 2)
        //     | ((mask & 0x10) >> 1);
        //
        // let masked_permuted_mask = permuted_mask & sibling_mask;
        //
        // masked_permuted_mask

        mask & sibling_mask
    }
}

/// SIMD prune block for f32 values (AVX2)
///
/// Compares 8 rd_values against max_dist in parallel and returns a bitmask
/// indicating which siblings should be explored.
#[cfg(target_feature = "avx2")]
#[inline(always)]
pub(crate) fn simd_prune_block_f32(rd_values: &[f32; 8], max_dist: f32, sibling_mask: u8) -> u8 {
    unsafe {
        use std::arch::x86_64::*;

        let max_dist_vec = _mm256_set1_ps(max_dist);
        let rd_vec = _mm256_loadu_ps(rd_values.as_ptr());

        let cmp = _mm256_cmp_ps(rd_vec, max_dist_vec, _CMP_LE_OQ);
        let mask = _mm256_movemask_ps(cmp) as u8;

        mask & sibling_mask
    }
}
