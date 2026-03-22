#![allow(clippy::missing_safety_doc)]
#![allow(clippy::too_many_arguments)]
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(dead_code)]

use std::arch::x86_64::*;

pub const CHUNK_SIZE: usize = 16;
pub const LINE_SIZE: usize = 4;
pub const XMM_LINE_SIZE: usize = 2;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct BestResult {
    pub dist: f64,
    pub item: u64,
}

#[inline(always)]
fn abs_f64(x: f64) -> f64 {
    x.abs()
}

#[inline(always)]
unsafe fn abs_pd_256(x: __m256d) -> __m256d {
    // Clear sign bit: abs(x) == andnot(-0.0, x)
    let sign = _mm256_set1_pd(-0.0);
    _mm256_andnot_pd(sign, x)
}

#[inline(always)]
unsafe fn abs_pd_128(x: __m128d) -> __m128d {
    // Clear sign bit: abs(x) == andnot(-0.0, x)
    let sign = _mm_set1_pd(-0.0);
    _mm_andnot_pd(sign, x)
}

pub trait Avx2F64LeafOps {
    unsafe fn init_256(delta: __m256d) -> __m256d;

    unsafe fn accum_256(acc: __m256d, delta: __m256d) -> __m256d;

    unsafe fn init_128(delta: __m128d) -> __m128d;

    unsafe fn accum_128(acc: __m128d, delta: __m128d) -> __m128d;

    fn init_scalar(delta: f64) -> f64;

    fn accum_scalar(acc: f64, delta: f64) -> f64;
}

pub trait DistanceMetricUnifiedF64 {
    type Avx2F64Ops: Avx2F64LeafOps;
}

pub struct SquaredEuclidean;
pub struct Manhattan;

pub struct SquaredEuclideanAvx2Ops;
pub struct ManhattanAvx2Ops;

impl Avx2F64LeafOps for SquaredEuclideanAvx2Ops {
    #[inline(always)]
    unsafe fn init_256(delta: __m256d) -> __m256d {
        _mm256_mul_pd(delta, delta)
    }

    #[inline(always)]
    unsafe fn accum_256(acc: __m256d, delta: __m256d) -> __m256d {
        _mm256_fmadd_pd(delta, delta, acc)
    }

    #[inline(always)]
    unsafe fn init_128(delta: __m128d) -> __m128d {
        _mm_mul_pd(delta, delta)
    }

    #[inline(always)]
    unsafe fn accum_128(acc: __m128d, delta: __m128d) -> __m128d {
        _mm_add_pd(acc, _mm_mul_pd(delta, delta))
    }

    #[inline(always)]
    fn init_scalar(delta: f64) -> f64 {
        delta * delta
    }

    #[inline(always)]
    fn accum_scalar(acc: f64, delta: f64) -> f64 {
        acc + delta * delta
    }
}

impl Avx2F64LeafOps for ManhattanAvx2Ops {
    #[inline(always)]
    unsafe fn init_256(delta: __m256d) -> __m256d {
        abs_pd_256(delta)
    }

    #[inline(always)]
    unsafe fn accum_256(acc: __m256d, delta: __m256d) -> __m256d {
        _mm256_add_pd(acc, abs_pd_256(delta))
    }

    #[inline(always)]
    unsafe fn init_128(delta: __m128d) -> __m128d {
        abs_pd_128(delta)
    }

    #[inline(always)]
    unsafe fn accum_128(acc: __m128d, delta: __m128d) -> __m128d {
        _mm_add_pd(acc, abs_pd_128(delta))
    }

    #[inline(always)]
    fn init_scalar(delta: f64) -> f64 {
        abs_f64(delta)
    }

    #[inline(always)]
    fn accum_scalar(acc: f64, delta: f64) -> f64 {
        acc + abs_f64(delta)
    }
}

impl DistanceMetricUnifiedF64 for SquaredEuclidean {
    type Avx2F64Ops = SquaredEuclideanAvx2Ops;
}

impl DistanceMetricUnifiedF64 for Manhattan {
    type Avx2F64Ops = ManhattanAvx2Ops;
}

#[inline(always)]
unsafe fn update_best_chunk_avx2_raw(
    d0: __m256d,
    d1: __m256d,
    d2: __m256d,
    d3: __m256d,
    items: *const u64,
    base: usize,
    best_dist: f64,
    best_item: u64,
) -> (f64, u64) {
    let bb = _mm256_set1_pd(best_dist);
    let m0 = _mm256_movemask_pd(_mm256_cmp_pd(d0, bb, _CMP_LT_OQ)) as u32;
    let m1 = _mm256_movemask_pd(_mm256_cmp_pd(d1, bb, _CMP_LT_OQ)) as u32;
    let m2 = _mm256_movemask_pd(_mm256_cmp_pd(d2, bb, _CMP_LT_OQ)) as u32;
    let m3 = _mm256_movemask_pd(_mm256_cmp_pd(d3, bb, _CMP_LT_OQ)) as u32;

    if (m0 | m1 | m2 | m3) == 0 {
        return (best_dist, best_item);
    }

    let min01 = _mm256_min_pd(d0, d1);
    let min23 = _mm256_min_pd(d2, d3);
    let min0123 = _mm256_min_pd(min01, min23);

    let hi128 = _mm256_extractf128_pd(min0123, 1);
    let lo128 = _mm256_castpd256_pd128(min0123);
    let min128 = _mm_min_pd(lo128, hi128);

    let hi64 = _mm_unpackhi_pd(min128, min128);
    let min_scalar = _mm_min_sd(min128, hi64);
    let chunk_min = _mm_cvtsd_f64(min_scalar);

    let bcast = _mm256_set1_pd(chunk_min);
    let eq0 = _mm256_movemask_pd(_mm256_cmp_pd(d0, bcast, _CMP_EQ_OQ)) as u32;
    let eq1 = _mm256_movemask_pd(_mm256_cmp_pd(d1, bcast, _CMP_EQ_OQ)) as u32;
    let eq2 = _mm256_movemask_pd(_mm256_cmp_pd(d2, bcast, _CMP_EQ_OQ)) as u32;
    let eq3 = _mm256_movemask_pd(_mm256_cmp_pd(d3, bcast, _CMP_EQ_OQ)) as u32;

    let combined = eq0 | (eq1 << 4) | (eq2 << 8) | (eq3 << 12);
    core::hint::assert_unchecked(combined != 0);
    let idx = combined.trailing_zeros() as usize;

    (chunk_min, *items.add(base + idx))
}

#[inline(always)]
unsafe fn update_best_line_avx2_raw(
    d0: __m256d,
    items: *const u64,
    base: usize,
    best_dist: f64,
    best_item: u64,
) -> (f64, u64) {
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

    (chunk_min, *items.add(base + idx))
}

#[inline(always)]
unsafe fn update_best_line_sse2_raw(
    d0: __m128d,
    items: *const u64,
    base: usize,
    best_dist: f64,
    best_item: u64,
) -> (f64, u64) {
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

    (chunk_min, *items.add(base + idx))
}

macro_rules! impl_leaf_kernel_k {
    ($extern_name:ident, $k:expr, [$(($dim:literal, $p:ident, $qs:ident, $qv:ident)),*]) => {
        #[target_feature(enable = "avx2,fma")]
        unsafe fn $extern_name<M: DistanceMetricUnifiedF64>(
            points: *const *const f64,
            items: *const u64,
            len: usize,
            query: *const f64,
            mut best_dist: f64,
            mut best_item: u64,
        ) -> BestResult {
            let p0 = *points.add(0);
            let q0s = *query.add(0);
            $(
                let $p = *points.add($dim);
                let $qs = *query.add($dim);
            )*

            let qv0 = _mm256_set1_pd(q0s);
            $(
                let $qv = _mm256_set1_pd($qs);
            )*

            let full_chunks_len = len & !(CHUNK_SIZE - 1);
            let mut base = 0usize;
            while base != full_chunks_len {
                macro_rules! chunk {
                    ($off:expr) => {{
                        let a0 = _mm256_loadu_pd(p0.add(base + $off * 4));
                        let d0 = _mm256_sub_pd(a0, qv0);
                        #[allow(unused_mut)]
                        let mut acc = <M as DistanceMetricUnifiedF64>::Avx2F64Ops::init_256(d0);
                        $(
                            let a = _mm256_loadu_pd($p.add(base + $off * 4));
                            let d = _mm256_sub_pd(a, $qv);
                            acc = <M as DistanceMetricUnifiedF64>::Avx2F64Ops::accum_256(acc, d);
                        )*
                        acc
                    }};
                }

                let d0 = chunk!(0);
                let d1 = chunk!(1);
                let d2 = chunk!(2);
                let d3 = chunk!(3);

                (best_dist, best_item) =
                    update_best_chunk_avx2_raw(d0, d1, d2, d3, items, base, best_dist, best_item);

                base += CHUNK_SIZE;
            }

            let full_lines_len = full_chunks_len + ((len - full_chunks_len) & !(LINE_SIZE - 1));
            while base != full_lines_len {
                let a0 = _mm256_loadu_pd(p0.add(base));
                let d0 = _mm256_sub_pd(a0, qv0);
                #[allow(unused_mut)]
                let mut acc = <M as DistanceMetricUnifiedF64>::Avx2F64Ops::init_256(d0);

                $(
                    let a = _mm256_loadu_pd($p.add(base));
                    let d = _mm256_sub_pd(a, $qv);
                    acc = <M as DistanceMetricUnifiedF64>::Avx2F64Ops::accum_256(acc, d);
                )*

                (best_dist, best_item) =
                    update_best_line_avx2_raw(acc, items, base, best_dist, best_item);

                base += LINE_SIZE;
            }

            if base + XMM_LINE_SIZE <= len {
                let qx0 = _mm256_castpd256_pd128(qv0);
                let a0 = _mm_loadu_pd(p0.add(base));
                let d0 = _mm_sub_pd(a0, qx0);
                #[allow(unused_mut)]
                let mut acc = <M as DistanceMetricUnifiedF64>::Avx2F64Ops::init_128(d0);

                $(
                    let qx = _mm256_castpd256_pd128($qv);
                    let a = _mm_loadu_pd($p.add(base));
                    let d = _mm_sub_pd(a, qx);
                    acc = <M as DistanceMetricUnifiedF64>::Avx2F64Ops::accum_128(acc, d);
                )*

                (best_dist, best_item) =
                    update_best_line_sse2_raw(acc, items, base, best_dist, best_item);

                base += XMM_LINE_SIZE;
            }

            for idx in base..len {
                #[allow(unused_mut)]
                let mut d = <M as DistanceMetricUnifiedF64>::Avx2F64Ops::init_scalar(*p0.add(idx) - q0s);
                $(
                    d = <M as DistanceMetricUnifiedF64>::Avx2F64Ops::accum_scalar(d, *$p.add(idx) - $qs);
                )*
                if d < best_dist {
                    best_dist = d;
                    best_item = *items.add(idx);
                }
            }

            BestResult {
                dist: best_dist,
                item: best_item,
            }
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

#[inline(always)]
unsafe fn scalar_fallback_dynamic<M: DistanceMetricUnifiedF64>(
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
        let mut d = <M as DistanceMetricUnifiedF64>::Avx2F64Ops::init_scalar(
            *(*points.get_unchecked(0)).add(idx) - *query.get_unchecked(0),
        );
        for dim in 1..k {
            d = <M as DistanceMetricUnifiedF64>::Avx2F64Ops::accum_scalar(
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
unsafe fn leaf_nearest_one_chunked_nozero_f64_selector<M: DistanceMetricUnifiedF64>(
    k: usize,
    points: *const *const f64,
    items: *const u64,
    len: usize,
    query: *const f64,
    best_dist_in: f64,
    best_item_in: u64,
) -> BestResult {
    match k {
        1 => leaf_nearest_one_chunked_nozero_f64_k1::<M>(
            points,
            items,
            len,
            query,
            best_dist_in,
            best_item_in,
        ),
        2 => leaf_nearest_one_chunked_nozero_f64_k2::<M>(
            points,
            items,
            len,
            query,
            best_dist_in,
            best_item_in,
        ),
        3 => leaf_nearest_one_chunked_nozero_f64_k3::<M>(
            points,
            items,
            len,
            query,
            best_dist_in,
            best_item_in,
        ),
        4 => leaf_nearest_one_chunked_nozero_f64_k4::<M>(
            points,
            items,
            len,
            query,
            best_dist_in,
            best_item_in,
        ),
        5 => leaf_nearest_one_chunked_nozero_f64_k5::<M>(
            points,
            items,
            len,
            query,
            best_dist_in,
            best_item_in,
        ),
        6 => leaf_nearest_one_chunked_nozero_f64_k6::<M>(
            points,
            items,
            len,
            query,
            best_dist_in,
            best_item_in,
        ),
        7 => leaf_nearest_one_chunked_nozero_f64_k7::<M>(
            points,
            items,
            len,
            query,
            best_dist_in,
            best_item_in,
        ),
        8 => leaf_nearest_one_chunked_nozero_f64_k8::<M>(
            points,
            items,
            len,
            query,
            best_dist_in,
            best_item_in,
        ),
        _ => scalar_fallback_dynamic::<M>(points, items, len, k, query, best_dist_in, best_item_in),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn leaf_nearest_one_chunked_nozero_f64_sq_euc(
    points: *const *const f64,
    items: *const u64,
    len: usize,
    query: *const f64,
    best_dist_in: f64,
    best_item_in: u64,
) -> BestResult {
    leaf_nearest_one_chunked_nozero_f64_k3::<SquaredEuclidean>(
        points,
        items,
        len,
        query,
        best_dist_in,
        best_item_in,
    )
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
    leaf_nearest_one_chunked_nozero_f64_sq_euc(
        points,
        items,
        len,
        query,
        best_dist_in,
        best_item_in,
    )
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn leaf_nearest_one_chunked_nozero_f64_manhattan(
    points: *const *const f64,
    items: *const u64,
    len: usize,
    query: *const f64,
    best_dist_in: f64,
    best_item_in: u64,
) -> BestResult {
    leaf_nearest_one_chunked_nozero_f64_k3::<Manhattan>(
        points,
        items,
        len,
        query,
        best_dist_in,
        best_item_in,
    )
}
