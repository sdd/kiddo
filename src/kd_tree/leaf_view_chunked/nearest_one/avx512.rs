#![allow(clippy::missing_safety_doc)]
#![allow(clippy::too_many_arguments)]

use std::arch::x86_64::*;

use crate::dist::distance_metric_avx512::Avx512F64LeafOps;
use crate::kd_tree::leaf_view::LeafView;
use crate::traits_unified_2::{AxisUnified, Basics};

const CHUNK_SIZE: usize = 32;
const LINE_SIZE: usize = 8;
const AVX2_LINE_SIZE: usize = 4;

#[repr(C)]
#[derive(Copy, Clone)]
struct BestResult<T: Basics> {
    dist: f64,
    item: T,
}

#[cfg(feature = "leaf_nta_prefetch")]
#[inline(always)]
unsafe fn prefetch_nta(ptr: *const u8) {
    _mm_prefetch::<{ _MM_HINT_NTA }>(ptr as *const i8);
}

#[inline(always)]
unsafe fn update_best_chunk_avx512_raw<T: Basics>(
    d0: __m512d,
    d1: __m512d,
    d2: __m512d,
    d3: __m512d,
    items: *const T,
    base: usize,
    best_dist: f64,
    best_item: T,
) -> (f64, T) {
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
    core::hint::assert_unchecked(combined != 0);
    let idx = combined.trailing_zeros() as usize;

    (chunk_min, std::ptr::read_unaligned(items.add(base + idx)))
}

#[inline(always)]
unsafe fn update_best_line_avx512_raw<T: Basics>(
    d0: __m512d,
    items: *const T,
    base: usize,
    best_dist: f64,
    best_item: T,
) -> (f64, T) {
    let bb = _mm512_set1_pd(best_dist);
    let m0 = _mm512_cmp_pd_mask(d0, bb, _CMP_LT_OQ);

    if m0 == 0 {
        return (best_dist, best_item);
    }

    let hi256 = _mm512_extractf64x4_pd(d0, 1);
    let lo256 = _mm512_castpd512_pd256(d0);
    let min256 = _mm256_min_pd(lo256, hi256);

    let hi128 = _mm256_extractf128_pd(min256, 1);
    let lo128 = _mm256_castpd256_pd128(min256);
    let min128 = _mm_min_pd(lo128, hi128);

    let hi64 = _mm_unpackhi_pd(min128, min128);
    let min_scalar = _mm_min_sd(min128, hi64);
    let chunk_min = _mm_cvtsd_f64(min_scalar);

    let min_bcast = _mm512_set1_pd(chunk_min);
    let eq0 = _mm512_cmp_pd_mask(d0, min_bcast, _CMP_EQ_OQ);

    core::hint::assert_unchecked(eq0 != 0);
    let idx = eq0.trailing_zeros() as usize;

    (chunk_min, std::ptr::read_unaligned(items.add(base + idx)))
}

#[inline(always)]
unsafe fn update_best_line_avx2_raw<T: Basics>(
    d0: __m256d,
    items: *const T,
    base: usize,
    best_dist: f64,
    best_item: T,
) -> (f64, T) {
    let bb = _mm256_set1_pd(best_dist);
    let m0 = _mm256_movemask_pd(_mm256_cmp_pd(d0, bb, _CMP_LT_OQ)) as u32;

    if m0 == 0 {
        return (best_dist, best_item);
    }

    let hi128 = _mm256_extractf128_pd(d0, 1);
    let lo128 = _mm256_castpd256_pd128(d0);
    let min128 = _mm_min_pd(lo128, hi128);

    let hi64 = _mm_unpackhi_pd(min128, min128);
    let min_scalar = _mm_min_sd(min128, hi64);
    let chunk_min = _mm_cvtsd_f64(min_scalar);

    let bcast = _mm256_set1_pd(chunk_min);
    let eq0 = _mm256_movemask_pd(_mm256_cmp_pd(d0, bcast, _CMP_EQ_OQ)) as u32;
    core::hint::assert_unchecked(eq0 != 0);
    let idx = eq0.trailing_zeros() as usize;

    (chunk_min, std::ptr::read_unaligned(items.add(base + idx)))
}

#[inline(always)]
unsafe fn update_best_line_avx128_raw<T: Basics>(
    d0: __m128d,
    items: *const T,
    base: usize,
    best_dist: f64,
    best_item: T,
) -> (f64, T) {
    let bb = _mm_set1_pd(best_dist);
    let m0 = _mm_movemask_pd(_mm_cmplt_pd(d0, bb)) as u32;

    if m0 == 0 {
        return (best_dist, best_item);
    }

    let hi64 = _mm_unpackhi_pd(d0, d0);
    let min_scalar = _mm_min_sd(d0, hi64);
    let chunk_min = _mm_cvtsd_f64(min_scalar);

    let bcast = _mm_set1_pd(chunk_min);
    let eq0 = _mm_movemask_pd(_mm_cmpeq_pd(d0, bcast)) as u32;
    core::hint::assert_unchecked(eq0 != 0);
    let idx = eq0.trailing_zeros() as usize;

    (chunk_min, std::ptr::read_unaligned(items.add(base + idx)))
}

macro_rules! impl_leaf_kernel_k {
    ($extern_name:ident, $k:expr, [$(($dim:literal, $p:ident, $qs:ident, $qv:ident)),*]) => {
        #[target_feature(enable = "avx512f,avx512vl,fma")]
        unsafe fn $extern_name<AX, L, T>(
            points: *const *const f64,
            items: *const T,
            len: usize,
            query: *const f64,
            mut best_dist: f64,
            mut best_item: T,
        ) -> BestResult<T>
        where
            AX: AxisUnified<Coord = AX>,
            L: Avx512F64LeafOps,
            T: Basics,
        {
            let p0 = *points.add(0);
            let q0s = *query.add(0);
            $(
                let $p = *points.add($dim);
                let $qs = *query.add($dim);
            )*

            let qv0 = _mm512_set1_pd(q0s);
            $(
                let $qv = _mm512_set1_pd($qs);
            )*

            let full_chunks_len = len & !(CHUNK_SIZE - 1);
            let mut base = 0usize;
            while base != full_chunks_len {
                macro_rules! chunk {
                    ($off:expr) => {{
                        let a0 = _mm512_loadu_pd(p0.add(base + $off * 8));
                        let d0 = _mm512_sub_pd(a0, qv0);
                        #[allow(unused_mut)]
                        let mut acc = L::dist_k0_f64x8(d0);
                        $(
                            let a = _mm512_loadu_pd($p.add(base + $off * 8));
                            let d = _mm512_sub_pd(a, $qv);
                            acc = L::dist_kn_f64x8(acc, d);
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

            let full_lines_len = full_chunks_len + ((len - full_chunks_len) & !(LINE_SIZE - 1));
            while base != full_lines_len {
                let a0 = _mm512_loadu_pd(p0.add(base));
                let d0 = _mm512_sub_pd(a0, qv0);
                #[allow(unused_mut)]
                let mut acc = L::dist_k0_f64x8(d0);

                $(
                    let a = _mm512_loadu_pd($p.add(base));
                    let d = _mm512_sub_pd(a, $qv);
                    acc = L::dist_kn_f64x8(acc, d);
                )*

                (best_dist, best_item) =
                    update_best_line_avx512_raw(acc, items, base, best_dist, best_item);

                base += LINE_SIZE;
            }

            if base + AVX2_LINE_SIZE <= len {
                let qy0 = _mm512_castpd512_pd256(qv0);
                let a0 = _mm256_loadu_pd(p0.add(base));
                let d0 = _mm256_sub_pd(a0, qy0);
                #[allow(unused_mut)]
                let mut acc = L::dist_k0_f64x4(d0);

                $(
                    let qy = _mm512_castpd512_pd256($qv);
                    let a = _mm256_loadu_pd($p.add(base));
                    let d = _mm256_sub_pd(a, qy);
                    acc = L::dist_kn_f64x4(acc, d);
                )*

                (best_dist, best_item) =
                    update_best_line_avx2_raw(acc, items, base, best_dist, best_item);

                base += AVX2_LINE_SIZE;
            }

            for idx in base..len {
                #[allow(unused_mut)]
                let mut d = L::dist_k0_f64x1(*p0.add(idx) - q0s);
                $(
                    d = L::dist_kn_f64x1(d, *$p.add(idx) - $qs);
                )*
                if d < best_dist {
                    best_dist = d;
                    best_item = std::ptr::read_unaligned(items.add(idx));
                }
            }

            BestResult { dist: best_dist, item: best_item }
        }
    };
}

impl_leaf_kernel_k!(leaf_nearest_one_chunked_nozero_f64_k1, 1, []);
impl_leaf_kernel_k!(
    leaf_nearest_one_chunked_nozero_f64_k2,
    2,
    [(1, p1, q1s, qv1)]
);
impl_leaf_kernel_k!(
    leaf_nearest_one_chunked_nozero_f64_k3,
    3,
    [(1, p1, q1s, qv1), (2, p2, q2s, qv2)]
);
impl_leaf_kernel_k!(
    leaf_nearest_one_chunked_nozero_f64_k4,
    4,
    [(1, p1, q1s, qv1), (2, p2, q2s, qv2), (3, p3, q3s, qv3)]
);
impl_leaf_kernel_k!(
    leaf_nearest_one_chunked_nozero_f64_k5,
    5,
    [
        (1, p1, q1s, qv1),
        (2, p2, q2s, qv2),
        (3, p3, q3s, qv3),
        (4, p4, q4s, qv4)
    ]
);
impl_leaf_kernel_k!(
    leaf_nearest_one_chunked_nozero_f64_k6,
    6,
    [
        (1, p1, q1s, qv1),
        (2, p2, q2s, qv2),
        (3, p3, q3s, qv3),
        (4, p4, q4s, qv4),
        (5, p5, q5s, qv5)
    ]
);
impl_leaf_kernel_k!(
    leaf_nearest_one_chunked_nozero_f64_k7,
    7,
    [
        (1, p1, q1s, qv1),
        (2, p2, q2s, qv2),
        (3, p3, q3s, qv3),
        (4, p4, q4s, qv4),
        (5, p5, q5s, qv5),
        (6, p6, q6s, qv6)
    ]
);
impl_leaf_kernel_k!(
    leaf_nearest_one_chunked_nozero_f64_k8,
    8,
    [
        (1, p1, q1s, qv1),
        (2, p2, q2s, qv2),
        (3, p3, q3s, qv3),
        (4, p4, q4s, qv4),
        (5, p5, q5s, qv5),
        (6, p6, q6s, qv6),
        (7, p7, q7s, qv7)
    ]
);

macro_rules! impl_leaf_arena_kernel_k {
    ($extern_name:ident, $k:expr, [$(($dim:literal, $p:ident, $qs:ident, $qv:ident, $qy:ident, $qx:ident)),*]) => {
        #[target_feature(enable = "avx512f,avx512vl,fma")]
        pub(crate) unsafe fn $extern_name<AX, L, T>(
            tile_base: *const u8,
            len: usize,
            query: *const f64,
            best_dist: &mut f64,
            best_item: &mut T,
        )
        where
            AX: AxisUnified<Coord = AX>,
            L: Avx512F64LeafOps,
            T: Basics,
        {
            if len == 0 {
                return;
            }

            let p0 = tile_base as *const f64;
            let items = tile_base.add($k * len * std::mem::size_of::<f64>()) as *const T;
            let q0s = *query.add(0);
            $(
                let $p = p0.add($dim * len);
                let $qs = *query.add($dim);
            )*

            let qv0 = _mm512_set1_pd(q0s);
            let qy0 = _mm512_castpd512_pd256(qv0);
            let qx0 = _mm256_castpd256_pd128(qy0);
            $(
                let $qv = _mm512_set1_pd($qs);
                let $qy = _mm512_castpd512_pd256($qv);
                let $qx = _mm256_castpd256_pd128($qy);
            )*

            let mut best_dist_val = *best_dist;
            let mut best_item_val = *best_item;
            let mut base = 0usize;

            if base + CHUNK_SIZE <= len {
                #[cfg(feature = "leaf_nta_prefetch")]
                {
                    // perf shows that the "base + 0" prefetches are
                    // too late to be useful but still seem to have some
                    // benefit.
                    // TODO: revisit this once we can ZC-deserealize a tree
                    //       via rkyv so that we get less noise in the perf output

                    // prefetch_nta(p0.add(base) as *const u8);
                    prefetch_nta(p0.add(base + 8) as *const u8);
                    prefetch_nta(p0.add(base + 16) as *const u8);
                    prefetch_nta(p0.add(base + 24) as *const u8);
                    $(
                        // prefetch_nta($p.add(base) as *const u8);
                        prefetch_nta($p.add(base + 8) as *const u8);
                        prefetch_nta($p.add(base + 16) as *const u8);
                        prefetch_nta($p.add(base + 24) as *const u8);
                    )*
                }

                let x0 = _mm512_loadu_pd(p0.add(base));
                let x1 = _mm512_loadu_pd(p0.add(base + 8));
                let x2 = _mm512_loadu_pd(p0.add(base + 16));
                let x3 = _mm512_loadu_pd(p0.add(base + 24));

                let dx0 = _mm512_sub_pd(x0, qv0);
                let dx1 = _mm512_sub_pd(x1, qv0);
                let dx2 = _mm512_sub_pd(x2, qv0);
                let dx3 = _mm512_sub_pd(x3, qv0);

                #[allow(unused_mut)]
                let mut d0 = L::dist_k0_f64x8(dx0);
                #[allow(unused_mut)]
                let mut d1 = L::dist_k0_f64x8(dx1);
                #[allow(unused_mut)]
                let mut d2 = L::dist_k0_f64x8(dx2);
                #[allow(unused_mut)]
                let mut d3 = L::dist_k0_f64x8(dx3);

                $(
                    let y0 = _mm512_loadu_pd($p.add(base));
                    let y1 = _mm512_loadu_pd($p.add(base + 8));
                    let y2 = _mm512_loadu_pd($p.add(base + 16));
                    let y3 = _mm512_loadu_pd($p.add(base + 24));

                    let dy0 = _mm512_sub_pd(y0, $qv);
                    let dy1 = _mm512_sub_pd(y1, $qv);
                    let dy2 = _mm512_sub_pd(y2, $qv);
                    let dy3 = _mm512_sub_pd(y3, $qv);

                    d0 = L::dist_kn_f64x8(d0, dy0);
                    d1 = L::dist_kn_f64x8(d1, dy1);
                    d2 = L::dist_kn_f64x8(d2, dy2);
                    d3 = L::dist_kn_f64x8(d3, dy3);
                )*

                (best_dist_val, best_item_val) = update_best_chunk_avx512_raw(
                    d0,
                    d1,
                    d2,
                    d3,
                    items,
                    base,
                    best_dist_val,
                    best_item_val,
                );

                base += CHUNK_SIZE;
            }

            if base + LINE_SIZE <= len {
                #[cfg(feature = "leaf_nta_prefetch")]
                {
                    prefetch_nta(p0.add(base) as *const u8);
                    $(
                        prefetch_nta($p.add(base) as *const u8);
                    )*
                }

                let a0 = _mm512_loadu_pd(p0.add(base));
                let d0 = _mm512_sub_pd(a0, qv0);
                #[allow(unused_mut)]
                let mut acc = L::dist_k0_f64x8(d0);

                $(
                    let a = _mm512_loadu_pd($p.add(base));
                    let d = _mm512_sub_pd(a, $qv);
                    acc = L::dist_kn_f64x8(acc, d);
                )*

                (best_dist_val, best_item_val) = update_best_line_avx512_raw(
                    acc,
                    items,
                    base,
                    best_dist_val,
                    best_item_val,
                );

                base += LINE_SIZE;
            }

            if base + AVX2_LINE_SIZE <= len {
                let a0 = _mm256_loadu_pd(p0.add(base));
                let d0 = _mm256_sub_pd(a0, qy0);
                #[allow(unused_mut)]
                let mut acc = L::dist_k0_f64x4(d0);

                $(
                    let a = _mm256_loadu_pd($p.add(base));
                    let d = _mm256_sub_pd(a, $qy);
                    acc = L::dist_kn_f64x4(acc, d);
                )*

                (best_dist_val, best_item_val) = update_best_line_avx2_raw(
                    acc,
                    items,
                    base,
                    best_dist_val,
                    best_item_val,
                );

                base += AVX2_LINE_SIZE;
            }

            if base + 2 <= len {
                let a0 = _mm_loadu_pd(p0.add(base));
                let d0 = _mm_sub_pd(a0, qx0);
                #[allow(unused_mut)]
                let mut acc = L::dist_k0_f64x2(d0);

                $(
                    let a = _mm_loadu_pd($p.add(base));
                    let d = _mm_sub_pd(a, $qx);
                    acc = L::dist_kn_f64x2(acc, d);
                )*

                (best_dist_val, best_item_val) = update_best_line_avx128_raw(
                    acc,
                    items,
                    base,
                    best_dist_val,
                    best_item_val,
                );

                base += 2;
            }

            while base < len {
                #[allow(unused_mut)]
                let mut d = L::dist_k0_f64x1(*p0.add(base) - q0s);
                $(
                    d = L::dist_kn_f64x1(d, *$p.add(base) - $qs);
                )*

                if d < best_dist_val {
                    best_dist_val = d;
                    best_item_val = std::ptr::read_unaligned(items.add(base));
                }

                base += 1;
            }

            *best_dist = best_dist_val;
            *best_item = best_item_val;
        }
    };
}

impl_leaf_arena_kernel_k!(leaf_nearest_one_arena_nozero_f64_k1, 1, []);
impl_leaf_arena_kernel_k!(
    leaf_nearest_one_arena_nozero_f64_k2,
    2,
    [(1, p1, q1s, qv1, qy1, qx1)]
);
impl_leaf_arena_kernel_k!(
    leaf_nearest_one_arena_nozero_f64_k3,
    3,
    [(1, p1, q1s, qv1, qy1, qx1), (2, p2, q2s, qv2, qy2, qx2)]
);
impl_leaf_arena_kernel_k!(
    leaf_nearest_one_arena_nozero_f64_k4,
    4,
    [
        (1, p1, q1s, qv1, qy1, qx1),
        (2, p2, q2s, qv2, qy2, qx2),
        (3, p3, q3s, qv3, qy3, qx3)
    ]
);

#[inline(always)]
unsafe fn scalar_fallback_dynamic<AX, L, T>(
    points: *const *const f64,
    items: *const T,
    len: usize,
    k: usize,
    query: *const f64,
    mut best_dist: f64,
    mut best_item: T,
) -> BestResult<T>
where
    AX: AxisUnified<Coord = AX>,
    L: Avx512F64LeafOps,
    T: Basics,
{
    let points = std::slice::from_raw_parts(points, k);
    let items = std::slice::from_raw_parts(items, len);
    let query = std::slice::from_raw_parts(query, k);

    for idx in 0..len {
        let mut d =
            L::dist_k0_f64x1(*(*points.get_unchecked(0)).add(idx) - *query.get_unchecked(0));
        for dim in 1..k {
            d = L::dist_kn_f64x1(
                d,
                *(*points.get_unchecked(dim)).add(idx) - *query.get_unchecked(dim),
            );
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
unsafe fn scalar_fallback_arena_dynamic<AX, L, T>(
    tile_base: *const u8,
    len: usize,
    k: usize,
    query: *const f64,
    best_dist: &mut f64,
    best_item: &mut T,
) where
    AX: AxisUnified<Coord = AX>,
    L: Avx512F64LeafOps,
    T: Basics,
{
    let points = tile_base as *const f64;
    let items = tile_base.add(k * len * std::mem::size_of::<f64>()) as *const T;
    let mut best_dist_val = *best_dist;
    let mut best_item_val = *best_item;

    for idx in 0..len {
        let mut d = L::dist_k0_f64x1(*points.add(idx) - *query.add(0));
        for dim in 1..k {
            let axis = points.add(dim * len);
            d = L::dist_kn_f64x1(d, *axis.add(idx) - *query.add(dim));
        }
        if d < best_dist_val {
            best_dist_val = d;
            best_item_val = std::ptr::read_unaligned(items.add(idx));
        }
    }

    *best_dist = best_dist_val;
    *best_item = best_item_val;
}

#[inline(always)]
unsafe fn leaf_nearest_one_chunked_nozero_f64_selector<AX, L, T>(
    k: usize,
    points: *const *const f64,
    items: *const T,
    len: usize,
    query: *const f64,
    best_dist_in: f64,
    best_item_in: T,
) -> BestResult<T>
where
    AX: AxisUnified<Coord = AX>,
    L: Avx512F64LeafOps,
    T: Basics,
{
    match k {
        1 => leaf_nearest_one_chunked_nozero_f64_k1::<AX, L, T>(
            points,
            items,
            len,
            query,
            best_dist_in,
            best_item_in,
        ),
        2 => leaf_nearest_one_chunked_nozero_f64_k2::<AX, L, T>(
            points,
            items,
            len,
            query,
            best_dist_in,
            best_item_in,
        ),
        3 => leaf_nearest_one_chunked_nozero_f64_k3::<AX, L, T>(
            points,
            items,
            len,
            query,
            best_dist_in,
            best_item_in,
        ),
        4 => leaf_nearest_one_chunked_nozero_f64_k4::<AX, L, T>(
            points,
            items,
            len,
            query,
            best_dist_in,
            best_item_in,
        ),
        5 => leaf_nearest_one_chunked_nozero_f64_k5::<AX, L, T>(
            points,
            items,
            len,
            query,
            best_dist_in,
            best_item_in,
        ),
        6 => leaf_nearest_one_chunked_nozero_f64_k6::<AX, L, T>(
            points,
            items,
            len,
            query,
            best_dist_in,
            best_item_in,
        ),
        7 => leaf_nearest_one_chunked_nozero_f64_k7::<AX, L, T>(
            points,
            items,
            len,
            query,
            best_dist_in,
            best_item_in,
        ),
        8 => leaf_nearest_one_chunked_nozero_f64_k8::<AX, L, T>(
            points,
            items,
            len,
            query,
            best_dist_in,
            best_item_in,
        ),
        _ => scalar_fallback_dynamic::<AX, L, T>(
            points,
            items,
            len,
            k,
            query,
            best_dist_in,
            best_item_in,
        ),
    }
}

#[inline(always)]
unsafe fn leaf_nearest_one_arena_nozero_f64_selector<AX, L, T>(
    k: usize,
    tile_base: *const u8,
    len: usize,
    query: *const f64,
    best_dist: &mut f64,
    best_item: &mut T,
) where
    AX: AxisUnified<Coord = AX>,
    L: Avx512F64LeafOps,
    T: Basics,
{
    match k {
        1 => leaf_nearest_one_arena_nozero_f64_k1::<AX, L, T>(
            tile_base, len, query, best_dist, best_item,
        ),
        2 => leaf_nearest_one_arena_nozero_f64_k2::<AX, L, T>(
            tile_base, len, query, best_dist, best_item,
        ),
        3 => leaf_nearest_one_arena_nozero_f64_k3::<AX, L, T>(
            tile_base, len, query, best_dist, best_item,
        ),
        4 => leaf_nearest_one_arena_nozero_f64_k4::<AX, L, T>(
            tile_base, len, query, best_dist, best_item,
        ),
        _ => scalar_fallback_arena_dynamic::<AX, L, T>(
            tile_base, len, k, query, best_dist, best_item,
        ),
    }
}

#[inline(always)]
pub(crate) unsafe fn nearest_one_avx512_raw_unchecked<AX, L, T, const K: usize>(
    points: [*const AX; K],
    items: *const T,
    len: usize,
    query_wide: &[AX; K],
    best_dist: &mut AX,
    best_item: &mut T,
) where
    AX: AxisUnified<Coord = AX>,
    L: Avx512F64LeafOps,
    T: Basics,
{
    if len == 0 {
        return;
    }

    let points_ptrs: [*const f64; K] = std::array::from_fn(|dim| points[dim] as *const f64);
    let query_ptr = query_wide.as_ptr() as *const f64;
    let best_dist_ptr = best_dist as *mut AX as *mut f64;

    let result = leaf_nearest_one_chunked_nozero_f64_selector::<AX, L, T>(
        K,
        points_ptrs.as_ptr(),
        items,
        len,
        query_ptr,
        *best_dist_ptr,
        *best_item,
    );

    *best_dist_ptr = result.dist;
    *best_item = result.item;
}

#[inline(always)]
pub(crate) unsafe fn nearest_one_avx512_arena_unchecked<AX, L, T, const K: usize>(
    tile_base: *const u8,
    len: usize,
    query: *const f64,
    best_dist: &mut f64,
    best_item: &mut T,
) where
    AX: AxisUnified<Coord = AX>,
    L: Avx512F64LeafOps,
    T: Basics,
{
    leaf_nearest_one_arena_nozero_f64_selector::<AX, L, T>(
        K, tile_base, len, query, best_dist, best_item,
    );
}

#[inline(always)]
pub(crate) unsafe fn nearest_one_avx512_unchecked<AX, L, T, const K: usize, const B: usize>(
    leaf: &LeafView<'_, AX, T, K, B>,
    query_wide: &[AX; K],
    best_dist: &mut AX,
    best_item: &mut T,
) where
    AX: AxisUnified<Coord = AX>,
    L: Avx512F64LeafOps,
    T: Basics,
{
    let points = leaf.points();
    let point_ptrs = std::array::from_fn(|dim| points[dim].as_ptr());

    nearest_one_avx512_raw_unchecked::<AX, L, T, K>(
        point_ptrs,
        leaf.items().as_ptr(),
        leaf.items().len(),
        query_wide,
        best_dist,
        best_item,
    );
}
