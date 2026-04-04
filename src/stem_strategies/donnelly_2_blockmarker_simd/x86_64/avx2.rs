//! AVX2 SIMD implementations for Block3 and Block4 comparisons

use std::ptr::NonNull;

/// Compare query value against 7 pivots in a Block3 (f64, AVX2)
///
/// Returns the child index (0-7) based on how many pivots the query exceeds.
#[target_feature(enable = "avx2,popcnt")]
#[inline(always)]
pub unsafe fn compare_block3_f64_avx2(
    stems_ptr: NonNull<u8>,
    cache_line_base: usize,
    query_val: f64,
) -> u8 {
    use std::arch::x86_64::*;

    let ptr = stems_ptr.as_ptr().add(cache_line_base * 8) as *const f64;

    let pivots_low = _mm256_loadu_pd(ptr);
    let pivots_high = _mm256_loadu_pd(ptr.add(4));
    let query_vec = _mm256_set1_pd(query_val);

    let cmp_low = _mm256_cmp_pd(query_vec, pivots_low, _CMP_GE_OQ);
    let cmp_high = _mm256_cmp_pd(query_vec, pivots_high, _CMP_GE_OQ);

    let mask_low = _mm256_movemask_pd(cmp_low) as u32;
    let mask_high = _mm256_movemask_pd(cmp_high) as u32;
    let mask = mask_low | (mask_high << 4);

    _popcnt32(mask as i32) as u8
}

/// Compare query value against 7 pivots in a Block3 (f32, AVX2)
///
/// Returns the child index (0-7) based on how many pivots the query exceeds.
#[target_feature(enable = "avx2,popcnt")]
#[inline(always)]
pub unsafe fn compare_block3_f32_avx2(
    stems_ptr: NonNull<u8>,
    cache_line_base: usize,
    query_val: f32,
) -> u8 {
    use std::arch::x86_64::*;

    let ptr = stems_ptr.as_ptr().add(cache_line_base * 4) as *const f32;

    let pivots = _mm256_loadu_ps(ptr);
    let query_vec = _mm256_set1_ps(query_val);

    let cmp = _mm256_cmp_ps(query_vec, pivots, _CMP_GE_OQ);

    let mask = _mm256_movemask_ps(cmp) as u32;

    _popcnt32(mask as i32) as u8
}

/// Compare query value against 15 pivots in a Block4 (f32, AVX2)
///
/// Returns the child index (0-15) based on how many pivots the query exceeds.
#[target_feature(enable = "avx2,popcnt")]
#[inline(always)]
pub unsafe fn compare_block4_f32_avx2(
    stems_ptr: NonNull<u8>,
    cache_line_base: usize,
    query_val: f32,
) -> u8 {
    use std::arch::x86_64::*;

    let ptr = stems_ptr.as_ptr().add(cache_line_base * 4) as *const f32;

    let pivots_low = _mm256_loadu_ps(ptr);
    let pivots_high = _mm256_loadu_ps(ptr.add(8));
    let query_vec = _mm256_set1_ps(query_val);

    let cmp_low = _mm256_cmp_ps(query_vec, pivots_low, _CMP_GE_OQ);
    let cmp_high = _mm256_cmp_ps(query_vec, pivots_high, _CMP_GE_OQ);

    let mask_low = _mm256_movemask_ps(cmp_low) as u32;
    let mask_high = _mm256_movemask_ps(cmp_high) as u32;
    let mask = mask_low | (mask_high << 8);

    _popcnt32(mask as i32) as u8
}
