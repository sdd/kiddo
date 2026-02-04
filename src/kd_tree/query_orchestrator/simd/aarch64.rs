//! ARM aarch64 NEON SIMD implementations for query orchestration pruning.

/// SIMD prune block for f64 values (NEON)
///
/// Compares 8 rd_values against max_dist in parallel and returns a bitmask.
#[inline(always)]
#[allow(dead_code)]
pub(crate) fn simd_prune_block_f64(rd_values: &[f64; 8], max_dist: f64, sibling_mask: u8) -> u8 {
    unsafe {
        use core::arch::aarch64::*;

        let max_vec = vdupq_n_f64(max_dist);
        let rd_0 = vld1q_f64(rd_values.as_ptr());
        let rd_1 = vld1q_f64(rd_values.as_ptr().add(2));
        let rd_2 = vld1q_f64(rd_values.as_ptr().add(4));
        let rd_3 = vld1q_f64(rd_values.as_ptr().add(6));

        let cmp_0 = vcleq_f64(rd_0, max_vec);
        let cmp_1 = vcleq_f64(rd_1, max_vec);
        let cmp_2 = vcleq_f64(rd_2, max_vec);
        let cmp_3 = vcleq_f64(rd_3, max_vec);

        let weights_0 = [1u64, 2u64];
        let weights_1 = [4u64, 8u64];
        let weights_2 = [16u64, 32u64];
        let weights_3 = [64u64, 128u64];

        let mask_0 = vaddvq_u64(vandq_u64(cmp_0, vld1q_u64(weights_0.as_ptr())));
        let mask_1 = vaddvq_u64(vandq_u64(cmp_1, vld1q_u64(weights_1.as_ptr())));
        let mask_2 = vaddvq_u64(vandq_u64(cmp_2, vld1q_u64(weights_2.as_ptr())));
        let mask_3 = vaddvq_u64(vandq_u64(cmp_3, vld1q_u64(weights_3.as_ptr())));

        let mask = (mask_0 | mask_1 | mask_2 | mask_3) as u8;

        mask & sibling_mask
    }
}

/// SIMD prune block for f32 values (NEON)
///
/// Compares 8 rd_values against max_dist in parallel and returns a bitmask.
#[inline(always)]
#[allow(dead_code)]
pub(crate) fn simd_prune_block_f32(rd_values: &[f32; 8], max_dist: f32, sibling_mask: u8) -> u8 {
    unsafe {
        use core::arch::aarch64::*;

        let max_vec = vdupq_n_f32(max_dist);
        let rd_0 = vld1q_f32(rd_values.as_ptr());
        let rd_1 = vld1q_f32(rd_values.as_ptr().add(4));

        let cmp_0 = vcleq_f32(rd_0, max_vec);
        let cmp_1 = vcleq_f32(rd_1, max_vec);

        let weights_0 = [1u32, 2u32, 4u32, 8u32];
        let weights_1 = [16u32, 32u32, 64u32, 128u32];

        let mask_0 = vaddvq_u32(vandq_u32(cmp_0, vld1q_u32(weights_0.as_ptr())));
        let mask_1 = vaddvq_u32(vandq_u32(cmp_1, vld1q_u32(weights_1.as_ptr())));

        let mask = (mask_0 | mask_1) as u8;

        mask & sibling_mask
    }
}
