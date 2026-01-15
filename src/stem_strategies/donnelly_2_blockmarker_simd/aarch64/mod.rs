//! ARM aarch64 SIMD implementations (NEON) for Donnelly block marker strategy
//!
//! TODO: Implement NEON-based SIMD comparisons for ARM processors.
//! For now, these are stubs that panic to allow the code to compile on aarch64.

use std::ptr::NonNull;

/// Compare query value against 7 pivots in a Block3 (f64, NEON)
///
/// TODO: Implement using NEON intrinsics
#[inline(always)]
pub unsafe fn compare_block3_f64_neon(
    _stems_ptr: NonNull<u8>,
    _cache_line_base: usize,
    _query_val: f64,
) -> u8 {
    unimplemented!("NEON implementation for Block3 f64 comparison not yet implemented")
}

/// Compare query value against 7 pivots in a Block3 (f32, NEON)
///
/// TODO: Implement using NEON intrinsics
#[inline(always)]
pub unsafe fn compare_block3_f32_neon(
    _stems_ptr: NonNull<u8>,
    _cache_line_base: usize,
    _query_val: f32,
) -> u8 {
    unimplemented!("NEON implementation for Block3 f32 comparison not yet implemented")
}

/// Compare query value against 15 pivots in a Block4 (f32, NEON)
///
/// TODO: Implement using NEON intrinsics
#[inline(always)]
pub unsafe fn compare_block4_f32_neon(
    _stems_ptr: NonNull<u8>,
    _cache_line_base: usize,
    _query_val: f32,
) -> u8 {
    unimplemented!("NEON implementation for Block4 f32 comparison not yet implemented")
}
