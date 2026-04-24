//! Autovectorization-friendly SIMD pruning implementations.

/// Autovec prune block for f64 values.
#[inline(always)]
#[allow(unused)]
pub(crate) fn simd_prune_block_f64(rd_values: &[f64; 8], max_dist: f64, sibling_mask: u8) -> u8 {
    let mut mask = 0u8;
    for i in 0..8 {
        if rd_values[i] <= max_dist {
            mask |= 1 << i;
        }
    }
    mask & sibling_mask
}

/// Autovec prune block for f32 values.
#[inline(always)]
#[allow(unused)]
pub(crate) fn simd_prune_block_f32(rd_values: &[f32; 8], max_dist: f32, sibling_mask: u8) -> u8 {
    let mut mask = 0u8;
    for i in 0..8 {
        if rd_values[i] <= max_dist {
            mask |= 1 << i;
        }
    }
    mask & sibling_mask
}
