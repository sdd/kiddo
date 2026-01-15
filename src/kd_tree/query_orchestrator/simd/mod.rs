//! SIMD operations for query orchestration
//!
//! This module provides SIMD-accelerated pruning operations used during
//! backtracking query traversal. Architecture-specific implementations
//! are in submodules.

use crate::traits_unified_2::AxisUnified;

// Architecture-specific modules
#[cfg(target_arch = "x86_64")]
mod x86_64;

#[cfg(target_arch = "aarch64")]
mod aarch64;

/// SIMD prune block helper
///
/// Compares 8 rd_values against max_dist in parallel and returns a bitmask
/// indicating which siblings pass the backtracking distance test.
///
/// This is a type-dispatching wrapper that calls the appropriate SIMD
/// implementation based on the size of O (f32 or f64).
#[inline(always)]
pub(crate) fn simd_prune_block<O>(rd_values: &[O; 8], max_dist: O, sibling_mask: u8) -> u8
where
    O: AxisUnified<Coord = O>,
{
    if std::mem::size_of::<O>() == 8 {
        // f64 path
        let max_dist_f64: f64 = unsafe { std::mem::transmute_copy(&max_dist) };
        let rd_f64: [f64; 8] = unsafe { std::mem::transmute_copy(rd_values) };

        #[cfg(target_arch = "x86_64")]
        {
            x86_64::simd_prune_block_f64(&rd_f64, max_dist_f64, sibling_mask)
        }

        #[cfg(target_arch = "aarch64")]
        {
            aarch64::simd_prune_block_f64(&rd_f64, max_dist_f64, sibling_mask)
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            compile_error!("SIMD pruning only supports x86_64 and aarch64")
        }
    } else if std::mem::size_of::<O>() == 4 {
        // f32 path
        let max_dist_f32: f32 = unsafe { std::mem::transmute_copy(&max_dist) };
        let rd_f32: [f32; 8] = unsafe { std::mem::transmute_copy(rd_values) };

        #[cfg(target_arch = "x86_64")]
        {
            x86_64::simd_prune_block_f32(&rd_f32, max_dist_f32, sibling_mask)
        }

        #[cfg(target_arch = "aarch64")]
        {
            aarch64::simd_prune_block_f32(&rd_f32, max_dist_f32, sibling_mask)
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            compile_error!("SIMD pruning only supports x86_64 and aarch64")
        }
    } else {
        panic!("Unsupported output type size for SIMD pruning");
    }
}
