//! Autovectorization-friendly block marker comparisons.

use std::ptr::NonNull;

/// Compare query value against 7 pivots in a Block3 (f64, autovec).
#[inline(always)]
#[allow(unused)]
pub unsafe fn compare_block3_f64_autovec(
    stems_ptr: NonNull<u8>,
    cache_line_base: usize,
    query_val: f64,
) -> u8 {
    let ptr = stems_ptr.as_ptr().add(cache_line_base * 8) as *const f64;
    let mut count = 0u8;
    for i in 0..8 {
        if query_val >= *ptr.add(i) {
            count += 1;
        }
    }
    count
}

/// Compare query value against 7 pivots in a Block3 (f32, autovec).
#[inline(always)]
#[allow(unused)]
pub unsafe fn compare_block3_f32_autovec(
    stems_ptr: NonNull<u8>,
    cache_line_base: usize,
    query_val: f32,
) -> u8 {
    let ptr = stems_ptr.as_ptr().add(cache_line_base * 4) as *const f32;
    let mut count = 0u8;
    for i in 0..8 {
        if query_val >= *ptr.add(i) {
            count += 1;
        }
    }
    count
}

/// Compare query value against 15 pivots in a Block4 (f32, autovec).
#[inline(always)]
#[allow(unused)]
pub unsafe fn compare_block4_f32_autovec(
    stems_ptr: NonNull<u8>,
    cache_line_base: usize,
    query_val: f32,
) -> u8 {
    let ptr = stems_ptr.as_ptr().add(cache_line_base * 4) as *const f32;
    let mut count = 0u8;
    for i in 0..16 {
        if query_val >= *ptr.add(i) {
            count += 1;
        }
    }
    count
}
