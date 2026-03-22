#![allow(clippy::missing_safety_doc)]
#![allow(clippy::too_many_arguments)]
#![allow(unsafe_op_in_unsafe_fn)]

use std::arch::x86_64::*;

pub const CHUNK_SIZE: usize = 32;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct BestResult {
    pub dist: f64,
    pub item: u64,
}

#[inline(always)]
fn dist1_sqe(a: f64, b: f64) -> f64 {
    let d = a - b;
    d * d
}

#[inline(always)]
unsafe fn update_best_chunk_avx512_raw(
    d0: __m512d,
    d1: __m512d,
    d2: __m512d,
    d3: __m512d,
    items: *const u64,
    base: usize,
    best_dist: f64,
    best_item: u64,
) -> (f64, u64) {
    let bb = _mm512_set1_pd(best_dist);
    let m0 = _mm512_cmp_pd_mask(d0, bb, _CMP_LT_OQ);
    let m1 = _mm512_cmp_pd_mask(d1, bb, _CMP_LT_OQ);
    let m2 = _mm512_cmp_pd_mask(d2, bb, _CMP_LT_OQ);
    let m3 = _mm512_cmp_pd_mask(d3, bb, _CMP_LT_OQ);

    if (m0 | m1 | m2 | m3) == 0 {
        return (best_dist, best_item);
    }

    let min01 = _mm512_min_pd(d0, d1);
    let min23 = _mm512_min_pd(d2, d3);
    let min0123 = _mm512_min_pd(min01, min23);

    let hi256 = _mm512_extractf64x4_pd(min0123, 1);
    let lo256 = _mm512_castpd512_pd256(min0123);
    let min256 = _mm256_min_pd(lo256, hi256);

    let hi128 = _mm256_extractf128_pd(min256, 1);
    let lo128 = _mm256_castpd256_pd128(min256);
    let min128 = _mm_min_pd(lo128, hi128);

    let hi64 = _mm_unpackhi_pd(min128, min128);
    let min_scalar = _mm_min_sd(min128, hi64);
    let chunk_min = _mm_cvtsd_f64(min_scalar);

    let min_bcast = _mm512_set1_pd(chunk_min);
    let eq0 = _mm512_cmp_pd_mask(d0, min_bcast, _CMP_EQ_OQ);
    let eq1 = _mm512_cmp_pd_mask(d1, min_bcast, _CMP_EQ_OQ);
    let eq2 = _mm512_cmp_pd_mask(d2, min_bcast, _CMP_EQ_OQ);
    let eq3 = _mm512_cmp_pd_mask(d3, min_bcast, _CMP_EQ_OQ);

    let combined = (eq0 as u32) | ((eq1 as u32) << 8) | ((eq2 as u32) << 16) | ((eq3 as u32) << 24);
    let idx = combined.trailing_zeros() as usize;

    (chunk_min, *items.add(base + idx))
}

macro_rules! impl_leaf_kernel_k {
  ($extern_name:ident, $k:expr, [$(($dim:literal, $p:ident, $q:ident)),*]) => {
      #[target_feature(enable = "avx512f,avx512vl,fma")]
      unsafe fn $extern_name(
          points: *const *const f64,
          items: *const u64,
          len: usize,
          query: *const f64,
          mut best_dist: f64,
          mut best_item: u64,
      ) -> BestResult {
          let p0 = *points.add(0);
          let q0 = *query.add(0);
          $(
              let $p = *points.add($dim);
              let $q = *query.add($dim);
          )*

          let qv0 = _mm512_set1_pd(q0);
          $(
              let $q = _mm512_set1_pd($q);
          )*

          let full_chunks_len = len & !(CHUNK_SIZE - 1);
          let mut base = 0usize;
            while base < full_chunks_len {
              macro_rules! chunk {
                  ($off:expr) => {{
                      let a0 = _mm512_loadu_pd(p0.add(base + $off * 8));
                      let d0 = _mm512_sub_pd(a0, qv0);
                      let mut acc = _mm512_mul_pd(d0, d0);
                      $(
                          let a = _mm512_loadu_pd($p.add(base + $off * 8));
                          let d = _mm512_sub_pd(a, $q);
                          acc = _mm512_fmadd_pd(d, d, acc);
                      )*
                      acc
                  }};
              }

              let d0 = chunk!(0);
              let d1 = chunk!(1);
              let d2 = chunk!(2);
              let d3 = chunk!(3);

              (best_dist, best_item) =
                  update_best_chunk_avx512_raw(d0, d1, d2, d3, items, base, best_dist, best_item);

              base += CHUNK_SIZE;
          }

          for idx in base..len {
              let mut d = dist1_sqe(*p0.add(idx), q0);
              $(
                  d += dist1_sqe(*$p.add(idx), _mm_cvtsd_f64(_mm512_castpd512_pd128($q)));
              )*
              if d < best_dist {
                  best_dist = d;
                  best_item = *items.add(idx);
              }
          }

          BestResult { dist: best_dist, item: best_item }
      }
  };
}

impl_leaf_kernel_k!(
    leaf_nearest_one_chunked_nozero_f64_k1,
    1,
    []
);
impl_leaf_kernel_k!(
    leaf_nearest_one_chunked_nozero_f64_k2,
    2,
    [(1, p1, q1)]
);
impl_leaf_kernel_k!(
    leaf_nearest_one_chunked_nozero_f64_k3,
    3,
    [(1, p1, q1), (2, p2, q2)]
);
impl_leaf_kernel_k!(
    leaf_nearest_one_chunked_nozero_f64_k4,
    4,
    [(1, p1, q1), (2, p2, q2), (3, p3, q3)]
);
impl_leaf_kernel_k!(
    leaf_nearest_one_chunked_nozero_f64_k5,
    5,
    [(1, p1, q1), (2, p2, q2), (3, p3, q3),  (4, p4, q4)]
);
impl_leaf_kernel_k!(
    leaf_nearest_one_chunked_nozero_f64_k6,
    6,
    [(1, p1, q1), (2, p2, q2), (3, p3, q3),  (4, p4, q4), (5, p5, q5)]
);
impl_leaf_kernel_k!(
    leaf_nearest_one_chunked_nozero_f64_k7,
    7,
    [(1, p1, q1), (2, p2, q2), (3, p3, q3),  (4, p4, q4), (5, p5, q5), (6, p6, q6)]
);
impl_leaf_kernel_k!(
    leaf_nearest_one_chunked_nozero_f64_k8,
    8,
    [(1, p1, q1), (2, p2, q2), (3, p3, q3),  (4, p4, q4), (5, p5, q5), (6, p6, q6), (7, p7, q7)]
);

#[inline(always)]
unsafe fn scalar_fallback_dynamic(
    points: *const *const f64,
    items: *const u64,
    len: usize,
    k: usize,
    query: *const f64,
    mut best_dist: f64,
    mut best_item: u64,
) -> BestResult {
    let points = std::slice::from_raw_parts(points, k);
    let items = std::slice::from_raw_parts(items, len);
    let query = std::slice::from_raw_parts(query, k);

    for idx in 0..len {
        let mut d = 0.0f64;
        for dim in 0..k {
            d += dist1_sqe(*(*points.get_unchecked(dim)).add(idx), *query.get_unchecked(dim));
        }
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
unsafe fn leaf_nearest_one_chunked_nozero_f64_selector(
    k: usize,
    points: *const *const f64,
    items: *const u64,
    len: usize,
    query: *const f64,
    best_dist_in: f64,
    best_item_in: u64,
) -> BestResult {
    match k {
        1 => leaf_nearest_one_chunked_nozero_f64_k1(points, items, len, query, best_dist_in, best_item_in),
        2 => leaf_nearest_one_chunked_nozero_f64_k2(points, items, len, query, best_dist_in, best_item_in),
        3 => leaf_nearest_one_chunked_nozero_f64_k3(points, items, len, query, best_dist_in, best_item_in),
        4 => leaf_nearest_one_chunked_nozero_f64_k4(points, items, len, query, best_dist_in, best_item_in),
        5 => leaf_nearest_one_chunked_nozero_f64_k5(points, items, len, query, best_dist_in, best_item_in),
        6 => leaf_nearest_one_chunked_nozero_f64_k6(points, items, len, query, best_dist_in, best_item_in),
        7 => leaf_nearest_one_chunked_nozero_f64_k7(points, items, len, query, best_dist_in, best_item_in),
        8 => leaf_nearest_one_chunked_nozero_f64_k8(points, items, len, query, best_dist_in, best_item_in),
        _ => scalar_fallback_dynamic(points, items, len, k, query, best_dist_in, best_item_in),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn leaf_nearest_one_chunked_nozero_f64(
    points: *const *const f64,
    items: *const u64,
    len: usize,
    query: *const f64,
    best_dist_in: f64,
    best_item_in: u64,
) -> BestResult {
    leaf_nearest_one_chunked_nozero_f64_k3(points, items, len, query, best_dist_in, best_item_in)
}
