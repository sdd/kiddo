use std::error::Error;

#[cfg(all(
    feature = "simd",
    target_feature = "avx2",
    any(target_arch = "x86", target_arch = "x86_64")
))]
use std::arch::x86_64::{
    __m128, __m128d, __m128i, __m256d, __m256i, _mm256_castpd256_pd128, _mm256_cmp_pd,
    _mm256_loadu_pd, _mm256_min_pd, _mm256_permutevar8x32_epi32, _mm256_set_epi32,
    _mm256_storeu_pd, _mm256_testz_si256, _mm_add_epi32, _mm_blendv_ps, _mm_maskstore_epi32,
    _mm_set1_epi32, _mm_set_epi32, _CMP_LT_OQ,
};

fn main() -> Result<(), Box<dyn Error>> {
    #[cfg(any(
        not(target_feature = "avx2"),
        all(not(target_arch = "x86"), not(target_arch = "x86_64"))
    ))]
    {
        println!("Not running on x86 or x86_64. Exiting");
    }

    #[cfg(all(
        feature = "simd",
        target_feature = "avx2",
        any(target_arch = "x86", target_arch = "x86_64")
    ))]
    {
        if is_x86_feature_detected!("avx512f") {
            println!("AVX512 Detected!");
        } else if is_x86_feature_detected!("avx2") {
            println!("AVX2 Detected!");
        } else {
            println!("No AVX2 or AVX512!");
            return Ok(());
        }

        let mut best_dist_val = 97f64;
        let mut best_item_val = 1usize;

        let best_dist = &mut best_dist_val;
        let best_item = &mut best_item_val;

        let items = [101, 102, 103, 104];

        let chunk_vals = [100f64, 90f64, 95f64, 80f64];
        let chunk = &chunk_vals;

        println!("Chunk = {:?}", chunk);

        unsafe {
            let is_better_shuffle_pattern: __m256i = _mm256_set_epi32(6, 4, 2, 0, 7, 5, 3, 1);
            println!(
                "is_better_shuffle_pattern = {:?}",
                &is_better_shuffle_pattern
            );

            let mut index_v = _mm_set_epi32(3, 2, 1, 0);
            println!("index_v = {:?}", &index_v);

            let mut min_dist_indexes_v = _mm_set1_epi32(-1);
            println!("min_dist_indexes_v = {:?}", &min_dist_indexes_v);

            let all_fours = _mm_set1_epi32(4);
            println!("all_fours = {:?}", &all_fours);

            let mask_128_all = _mm_set1_epi32(-1);
            println!("mask_128_all = {:?}", &mask_128_all);

            let mut min_dists = [*best_dist; 4];
            println!("min_dists = {:?}", &min_dists);

            let mut min_dists_v = _mm256_loadu_pd(std::ptr::addr_of!(min_dists[0]));
            println!("min_dists_v = {:?}", &min_dists_v);

            let mut any_is_better = false;
            let chunk_v = _mm256_loadu_pd(std::ptr::addr_of!(chunk[0]));
            println!("chunk_v = {:?}", &chunk_v);

            let is_better = _mm256_cmp_pd::<_CMP_LT_OQ>(chunk_v, min_dists_v);
            println!("is_better = {:?}", &is_better);

            let these_better = _mm256_testz_si256(
                std::mem::transmute::<__m256d, __m256i>(is_better),
                std::mem::transmute::<__m256d, __m256i>(is_better),
            );
            println!("these_better = {:?}", &these_better);

            any_is_better |= these_better == 0;
            println!("any_is_better = {:?}", &any_is_better);

            min_dists_v = _mm256_min_pd(min_dists_v, chunk_v);
            println!("min_dists_v = {:?}", &min_dists_v);

            let is_better_shuffled = _mm256_permutevar8x32_epi32(
                std::mem::transmute::<__m256d, __m256i>(is_better),
                is_better_shuffle_pattern,
            );
            println!("min_dists_v = {:?}", &min_dists_v);

            let is_better_mask =
                _mm256_castpd256_pd128(std::mem::transmute::<__m256i, __m256d>(is_better_shuffled));
            println!("is_better_shuffled = {:?}", &is_better_shuffled);

            min_dist_indexes_v = std::mem::transmute::<__m128, __m128i>(_mm_blendv_ps(
                std::mem::transmute::<__m128i, __m128>(min_dist_indexes_v),
                std::mem::transmute::<__m128i, __m128>(index_v),
                std::mem::transmute::<__m128d, __m128>(is_better_mask),
            ));
            println!("min_dist_indexes_v = {:?}", &min_dist_indexes_v);

            index_v = _mm_add_epi32(index_v, all_fours);
            println!("index_v = {:?}", &index_v);

            if !any_is_better {
                println!("None better!");
                return Ok(());
            }

            let mut min_dist_indexes = [0i32; 4];
            println!("min_dist_indexes = {:?}", &min_dist_indexes);

            _mm_maskstore_epi32(
                std::ptr::addr_of_mut!(min_dist_indexes[0]),
                mask_128_all,
                min_dist_indexes_v,
            );
            println!("min_dist_indexes = {:?}", &min_dist_indexes);

            _mm256_storeu_pd(std::ptr::addr_of_mut!(min_dists[0]), min_dists_v);
            println!("min_dists = {:?}", &min_dists);

            for (i, dist) in min_dists.iter().enumerate() {
                if *dist < *best_dist {
                    *best_dist = *dist;
                    println!("best_dist = {:?}", &best_dist);

                    *best_item = items[min_dist_indexes[i] as usize];
                    println!("best_item = {:?}", &best_item);
                } else {
                    println!("No change in best dist");
                }
            }
        }
    }

    Ok(())
}
