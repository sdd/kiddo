#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use core::arch::x86_64::{_mm_add_epi16, _mm_set1_epi16, _mm_setzero_si128, _CMP_NLT_UQ};

// #[cfg(target_feature = "avx2")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use core::arch::x86_64::{_mm256_cmp_pd, _mm256_loadu_pd, _mm256_min_pd, _mm256_storeu_pd};
use std::ptr;

use crate::types::Content;

// #[cfg(target_feature = "avx2")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(crate) unsafe fn get_best_from_dists_f64_avx2<T: Content, const B: usize>(
    acc: &[f64; B],
    items: &[T; B],
    best_dist: &mut f64,
    best_item: &mut T,
) {
    let mut index_v = _mm_setzero_si128();
    let mut min_dist_indexes_v = _mm_set1_epi16(-1);
    let all_ones = _mm_set1_epi16(1);
    let mut min_dists = [*best_dist; 8];
    let mut min_dists_v = _mm256_loadu_pd(ptr::addr_of!(min_dists[0]));

    let mut any_is_better = false;

    // AVX, 128 iterations, unrolled x2
    for chunk in acc.as_chunks_unchecked::<4>().iter() {
        let chunk_v = _mm256_loadu_pd(ptr::addr_of!(chunk[0]));

        // TODO: is there a better way to determine if is_better_mask != all zeros?
        let is_better_mask = _mm256_cmp_pd(min_dists_v, chunk_v, _CMP_NLT_UQ);

        let mut is_better_exploded = [0f64; 4];
        _mm256_storeu_pd(ptr::addr_of_mut!(is_better_exploded[0]), is_better_mask);
        any_is_better |= (is_better_exploded[0] != 0f64
            || is_better_exploded[1] != 0f64
            || is_better_exploded[2] != 0f64
            || is_better_exploded[3] != 0f64);

        min_dists_v = _mm256_min_pd(min_dists_v, chunk_v);

        min_dist_indexes_v = _mm_mask_mov_epi16(min_dist_indexes_v, is_better_mask, index_v);

        index_v = _mm_add_epi16(index_v, all_ones);
    }

    if !any_is_better {
        return;
    }

    let mut min_dist_indexes = [0i16; 4];
    _mm_storeu_epi16(ptr::addr_of_mut!(min_dist_indexes[0]), min_dist_indexes_v);
    _mm256_storeu_pd(ptr::addr_of_mut!(min_dists[0]), min_dists_v);

    for (i, dist) in min_dists.iter().enumerate() {
        if *dist < *best_dist {
            *best_dist = *dist;
            *best_item = items[min_dist_indexes[i] as usize + i];
        }
    }
}
