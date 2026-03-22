#![allow(clippy::missing_safety_doc)]
#![allow(clippy::too_many_arguments)]

use std::arch::x86_64::*;
use std::mem::MaybeUninit;

pub const CHUNK_SIZE: usize = 32;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct BestResult {
    pub dist: f64,
    pub item: u64,
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn leaf_nearest_one_chunked_nozero_f64_k3(
    axis0: *const f64,
    axis1: *const f64,
    axis2: *const f64,
    items: *const u64,
    len: usize,
    q0: f64,
    q1: f64,
    q2: f64,
    best_dist_in: f64,
    best_item_in: u64,
) -> BestResult {
    let axis0 = std::slice::from_raw_parts(axis0, len);
    let axis1 = std::slice::from_raw_parts(axis1, len);
    let axis2 = std::slice::from_raw_parts(axis2, len);
    let items = std::slice::from_raw_parts(items, len);

    let mut best_dist = best_dist_in;
    let mut best_item = best_item_in;

    let full_chunks_len = len & !(CHUNK_SIZE - 1);
    let mut base = 0usize;

    // Chunk loop
    while base < full_chunks_len {
        let q0v = _mm512_set1_pd(q0);
        let q1v = _mm512_set1_pd(q1);
        let q2v = _mm512_set1_pd(q2);

        // return 4 ZMM regs here to ensure that we don't spill
        // to stack when marshalling the function call here
        let (d0, d1, d2, d3) =
            dists_for_chunk_nozero_fma_x4(axis0, axis1, axis2, base, q0v, q1v, q2v);

        update_best_chunk_avx512(d0, d1, d2, d3, items, base, &mut best_dist, &mut best_item);

        base += CHUNK_SIZE;
    }

    // scalar trailer loop for remainder beyond last full chunk
    for idx in base..len {
        let d = dist1_sqe(*axis0.get_unchecked(idx), q0)
            + dist1_sqe(*axis1.get_unchecked(idx), q1)
            + dist1_sqe(*axis2.get_unchecked(idx), q2);
        if d < best_dist {
            best_dist = d;
            best_item = *items.get_unchecked(idx);
        }
    }

    BestResult {
        dist: best_dist,
        item: best_item,
    }
}

#[inline(always)]
unsafe fn dists_for_chunk_nozero_fma_x4(
    axis0: &[f64],
    axis1: &[f64],
    axis2: &[f64],
    base: usize,
    q0v: __m512d,
    q1v: __m512d,
    q2v: __m512d,
) -> (__m512d, __m512d, __m512d, __m512d) {
    macro_rules! chunk {
        ($off:expr) => {{
            #[cfg(feature = "leaf_nta_prefetch")]
            {
                // non-temporal (NTA) prefetches for these addresses
                // may help reduce L2 pollution, letting us keep more of the
                // stem data cache-resident, rather than replacing them in
                // the cache with leaf data that is unlikely to be re-loaded
                // any time soon
                _mm_prefetch(axis0.as_ptr().add(p) as *const i8, _MM_HINT_NTA);
                _mm_prefetch(axis1.as_ptr().add(p) as *const i8, _MM_HINT_NTA);
                _mm_prefetch(axis2.as_ptr().add(p) as *const i8, _MM_HINT_NTA);
            }

            let a0 = _mm512_loadu_pd(axis0.as_ptr().add(base + $off * 8));
            let a1 = _mm512_loadu_pd(axis1.as_ptr().add(base + $off * 8));
            let a2 = _mm512_loadu_pd(axis2.as_ptr().add(base + $off * 8));

            let d0 = _mm512_sub_pd(a0, q0v);
            let d1 = _mm512_sub_pd(a1, q1v);
            let d2 = _mm512_sub_pd(a2, q2v);

            let acc = _mm512_mul_pd(d0, d0);

            // use FMAs for combining 2nd dimension onwards ino the acc.
            // These need to be intrinsics - unless the entire binary is
            // compiled with `-C llvm-args=-fp-contract=fast`,
            // the compiler won't output FMAs, and consumers
            // may not want to use fast math everywhere
            let acc = _mm512_fmadd_pd(d1, d1, acc);
            _mm512_fmadd_pd(d2, d2, acc)
        }};
    }

    (chunk!(0), chunk!(1), chunk!(2), chunk!(3))
}

#[inline(always)]
unsafe fn update_best_chunk_avx512(
    d0: __m512d,
    d1: __m512d,
    d2: __m512d,
    d3: __m512d,
    items: &[u64],
    base: usize,
    best_dist: &mut f64,
    best_item: &mut u64,
) {
    // first, we do a very quick check to see if any
    // of the distances are a new best. Since we
    // already have these in ZMM regs and we have multiple
    // CPU ports, this is a very quick check and we can
    // exit early if we know that none of the entries
    // in the leaf is the new best. ~6 cycles.
    let bb = _mm512_set1_pd(*best_dist);
    let m0 = _mm512_cmp_pd_mask(d0, bb, _CMP_LT_OQ);
    let m1 = _mm512_cmp_pd_mask(d1, bb, _CMP_LT_OQ);
    let m2 = _mm512_cmp_pd_mask(d2, bb, _CMP_LT_OQ);
    let m3 = _mm512_cmp_pd_mask(d3, bb, _CMP_LT_OQ);

    if (m0 | m1 | m2 | m3) == 0 {
        return;
    }

    // Cold path: find min and its index. Finding
    // the min is much quicker to do than finding the
    // index

    // first find the min value per lane across
    // the 8 lanes in all four regs in the chunk
    let min01 = _mm512_min_pd(d0, d1);
    let min23 = _mm512_min_pd(d2, d3);
    let min0123 = _mm512_min_pd(min01, min23);

    // extract the hi and lo halves into 256 bit regs
    // and find the min value per lane across the 4 lanes
    let hi256 = _mm512_extractf64x4_pd(min0123, 1);
    let lo256 = _mm512_castpd512_pd256(min0123);
    let min256 = _mm256_min_pd(lo256, hi256);

    // extract the hi and lo halves into 128 bit regs
    // and find the min value per lane across the 2 lanes
    let hi128 = _mm256_extractf128_pd(min256, 1);
    let lo128 = _mm256_castpd256_pd128(min256);
    let min128 = _mm_min_pd(lo128, hi128);

    // find the min across those two lanes by
    // copying hi to lo of hi64 and comparing lo vs lo,
    // storing min in lo of min_scalar
    let hi64 = _mm_unpackhi_pd(min128, min128);
    let min_scalar = _mm_min_sd(min128, hi64);

    // extract lo into an f64 GP reg
    let chunk_min = _mm_cvtsd_f64(min_scalar);

    *best_dist = chunk_min;

    // broadcast min val to all 8 lanes
    let min_bcast = _mm512_set1_pd(chunk_min);

    // test for equality for all 32 dists in in chunk
    let eq0 = _mm512_cmp_pd_mask(d0, min_bcast, _CMP_EQ_OQ);
    let eq1 = _mm512_cmp_pd_mask(d1, min_bcast, _CMP_EQ_OQ);
    let eq2 = _mm512_cmp_pd_mask(d2, min_bcast, _CMP_EQ_OQ);
    let eq3 = _mm512_cmp_pd_mask(d3, min_bcast, _CMP_EQ_OQ);

    // horizontally merge all 8-bit masks
    let combined = (eq0 as u32) | ((eq1 as u32) << 8) | ((eq2 as u32) << 16) | ((eq3 as u32) << 24);

    // set idx to the index of the first set bit from the left
    let idx = combined.trailing_zeros() as usize;
    *best_item = *items.get_unchecked(base + idx);
}

#[inline(always)]
fn dist1_sqe(a: f64, b: f64) -> f64 {
    let d = a - b;
    d * d
}
