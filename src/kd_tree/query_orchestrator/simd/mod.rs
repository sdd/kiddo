//! SIMD operations for query orchestration
//!
//! This module provides SIMD-accelerated pruning operations used during
//! backtracking query traversal. Architecture-specific implementations
//! are in submodules.

// Architecture-specific modules
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
mod x86_64;

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
mod aarch64;

mod autovec;

/// SIMD prune block helper
///
/// Compares 8 rd_values against max_dist in parallel and returns a bitmask
/// indicating which siblings pass the backtracking distance test.
///
/// This delegates to the type-specific implementation via the SimdPrune trait,
/// which provides architecture-specific SIMD or autovec fallback implementations.
#[inline(always)]
pub(crate) fn simd_prune_block<O>(rd_values: &[O; 8], max_dist: O, sibling_mask: u8) -> u8
where
    O: crate::stem_strategies::SimdPrune,
{
    O::simd_prune_block3(rd_values, max_dist, sibling_mask)
}
