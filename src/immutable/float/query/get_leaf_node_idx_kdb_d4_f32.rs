use crate::immutable::float::kdbtree::ArchivedR8ImmutableKdBTree;
use crate::stem_strategies::Donnelly;
use crate::traits::StemStrategy;

impl ArchivedR8ImmutableKdBTree<f32, usize, 4, 2> {
    pub fn get_leaf_node_idx(&self, query: &[f32; 4]) -> usize {
        let mut level = 0;
        let mut dim = 0;
        let mut stem_idx = 0;

        while level <= self.max_stem_level {
            let vals: [f32; 8] = *unsafe { self.stems.get_unchecked(stem_idx) };
            let query_val = *unsafe { query.get_unchecked(dim) };

            // simd comp for all 8 vals at once
            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::x86_64::*;
                unsafe {
                    let query_vec = _mm256_set1_ps(query_val);
                    let vals_vec = _mm256_loadu_ps(vals.as_ptr());
                    let cmp_result = _mm256_cmp_ps(query_vec, vals_vec, _CMP_GE_OQ);
                    let cmp_mask = _mm256_movemask_ps(cmp_result);

                    // popcnt to determine child node index (0-7, since 8th is always infinity)
                    let child_offset = cmp_mask.count_ones() as usize;
                    stem_idx = (stem_idx << 3) + child_offset;
                }
            }

            #[cfg(not(target_arch = "x86_64"))]
            {
                // Only check first 7 values, 8th is always infinity
                let mut child_offset = 0;
                for i in 0..7 {
                    if query_val >= vals[i] {
                        child_offset += 1;
                    } else {
                        break;
                    }
                }

                stem_idx = (stem_idx << 3) + child_offset;
            }

            level += 1;

            let wrap_dim_mask = 0usize.wrapping_sub((dim == (4 - 1)) as usize);
            dim = (dim + 1) & !wrap_dim_mask;
        }

        stem_idx
    }
}
