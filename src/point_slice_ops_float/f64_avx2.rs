use core::arch::x86_64::{
    __m256i, _mm256_castpd256_pd128, _mm256_cmp_pd, _mm256_loadu_pd, _mm256_min_pd,
    _mm256_permutevar8x32_epi32, _mm256_set_epi32, _mm256_storeu_pd, _mm256_testz_si256,
    _mm_add_epi32, _mm_blendv_ps, _mm_maskstore_epi32, _mm_set1_epi32, _mm_set_epi32, _CMP_LT_OQ,
};

const NUM_LANES: usize = 4;

pub(crate) unsafe fn get_best_from_dists_f64_avx2<T: crate::types::Content>(
    acc: &[f64],
    items: &[T],
    best_dist: &mut f64,
    best_item: &mut T,
) {
    let is_better_shuffle_pattern: __m256i = unsafe { _mm256_set_epi32(6, 4, 2, 0, 7, 5, 3, 1) };

    // SSE2 (_mm_set_epi32 & _mm_set1_epi32)
    let mut index_v = _mm_set_epi32(3, 2, 1, 0);
    let mut min_dist_indexes_v = _mm_set1_epi32(-1);
    let all_fours = _mm_set1_epi32(4);
    let mask_128_all = _mm_set1_epi32(-1);

    let mut min_dists = [*best_dist; NUM_LANES];
    let mut min_dists_v = _mm256_loadu_pd(std::ptr::addr_of!(min_dists[0]));

    let mut any_is_better = false;
    for chunk in acc.as_chunks_unchecked::<NUM_LANES>().iter() {
        let chunk_v = _mm256_loadu_pd(std::ptr::addr_of!(chunk[0]));

        let is_better = _mm256_cmp_pd(chunk_v, min_dists_v, _CMP_LT_OQ);

        let these_better = _mm256_testz_si256(
            std::mem::transmute(is_better),
            std::mem::transmute(is_better),
        );

        any_is_better |= these_better == 0;

        min_dists_v = _mm256_min_pd(min_dists_v, chunk_v);

        let is_better_shuffled =
            _mm256_permutevar8x32_epi32(std::mem::transmute(is_better), is_better_shuffle_pattern);

        let is_better_mask = _mm256_castpd256_pd128(std::mem::transmute(is_better_shuffled));

        // SSE4.1
        min_dist_indexes_v = std::mem::transmute(_mm_blendv_ps(
            std::mem::transmute(min_dist_indexes_v),
            std::mem::transmute(index_v),
            std::mem::transmute(is_better_mask),
        ));

        // SSE2
        index_v = _mm_add_epi32(index_v, all_fours);
    }

    if !any_is_better {
        return;
    }

    let mut min_dist_indexes = [0i32; NUM_LANES];

    _mm_maskstore_epi32(
        std::ptr::addr_of_mut!(min_dist_indexes[0]),
        mask_128_all,
        min_dist_indexes_v,
    );

    _mm256_storeu_pd(std::ptr::addr_of_mut!(min_dists[0]), min_dists_v);

    for (i, dist) in min_dists.iter().enumerate() {
        if *dist < *best_dist {
            *best_dist = *dist;
            *best_item = items[min_dist_indexes[i] as usize];
        }
    }
}
