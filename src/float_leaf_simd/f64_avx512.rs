use core::arch::x86_64::{
    _mm_add_epi16, _mm_mask_mov_epi16, _mm_set1_epi16, _mm_setzero_si128, _mm_storeu_epi16,
    _CMP_NLT_UQ,
};

use core::arch::x86_64::{_mm512_cmp_pd_mask, _mm512_loadu_pd, _mm512_min_pd, _mm512_storeu_pd};
use std::ptr;

use crate::types::Content;

pub(crate) unsafe fn get_best_from_dists_f64_avx512<T: Content, const B: usize>(
    acc: &[f64; B],
    items: &[T; B],
    best_dist: &mut f64,
    best_item: &mut T,
) {
    // SSE2 (_mm_setzero_si128 & _mm_set1_epi16)
    let mut index_v = _mm_setzero_si128();
    let mut min_dist_indexes_v = _mm_set1_epi16(-1);
    let all_ones = _mm_set1_epi16(1);

    let mut min_dists = [*best_dist; 8];
    let mut min_dists_v = _mm512_loadu_pd(ptr::addr_of!(min_dists[0]));

    let mut any_is_better = false;

    // AVX512: 64 iterations
    for chunk in acc.as_chunks_unchecked::<8>().iter() {
        let chunk_v = _mm512_loadu_pd(ptr::addr_of!(chunk[0]));

        let is_better_mask = _mm512_cmp_pd_mask(min_dists_v, chunk_v, _CMP_NLT_UQ);
        any_is_better |= is_better_mask != 0;

        min_dists_v = _mm512_min_pd(min_dists_v, chunk_v);

        // AVX512BW + AVX512VL
        min_dist_indexes_v = _mm_mask_mov_epi16(min_dist_indexes_v, is_better_mask, index_v);

        // SSE2
        index_v = _mm_add_epi16(index_v, all_ones);
    }

    if !any_is_better {
        return;
    }

    let mut min_dist_indexes = [0i16; 8];
    // AVX512BW + AVX512VL
    _mm_storeu_epi16(ptr::addr_of_mut!(min_dist_indexes[0]), min_dist_indexes_v);
    _mm512_storeu_pd(ptr::addr_of_mut!(min_dists[0]), min_dists_v);

    for (i, dist) in min_dists.iter().enumerate() {
        if *dist < *best_dist {
            *best_dist = *dist;
            *best_item = items[min_dist_indexes[i] as usize + i];
        }
    }
}
