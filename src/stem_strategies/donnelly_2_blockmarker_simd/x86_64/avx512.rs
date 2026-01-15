//! AVX-512 SIMD implementations for Block3 and Block4 comparisons

use std::ptr::NonNull;

/// Compare query value against 7 pivots in a Block3 (f64, AVX-512)
///
/// Returns the child index (0-7) based on how many pivots the query exceeds.
#[target_feature(enable = "avx512f,avx512vl,popcnt")]
#[inline(always)]
pub unsafe fn compare_block3_f64_avx512(
    stems_ptr: NonNull<u8>,
    cache_line_base: usize,
    query_val: f64,
) -> u8 {
    use std::arch::x86_64::*;

    let ptr = stems_ptr.as_ptr().add(cache_line_base * 8) as *const f64;
    let pivots = _mm512_loadu_pd(ptr);
    let query_vec = _mm512_set1_pd(query_val);

    let mask = _mm512_cmp_pd_mask(query_vec, pivots, _CMP_GE_OQ);
    _popcnt32(mask as i32) as u8
}

/// Compare query value against 15 pivots in a Block4 (f32, AVX-512)
///
/// Returns the child index (0-15) based on how many pivots the query exceeds.
#[target_feature(enable = "avx512f,avx512vl,popcnt")]
#[inline(always)]
pub unsafe fn compare_block4_f32_avx512(
    stems_ptr: NonNull<u8>,
    cache_line_base: usize,
    query_val: f32,
) -> u8 {
    use std::arch::x86_64::*;

    let ptr = stems_ptr.as_ptr().add(cache_line_base * 8) as *const f32;
    let pivots = _mm512_loadu_ps(ptr);
    let query_vec = _mm512_set1_ps(query_val);

    let mask = _mm512_cmp_ps_mask(query_vec, pivots, _CMP_GE_OQ);
    _popcnt32(mask as i32) as u8
}
