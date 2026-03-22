#![allow(clippy::missing_safety_doc)]
#![allow(clippy::too_many_arguments)]
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(dead_code)]

use std::arch::x86_64::*;

pub const CHUNK_SIZE: usize = 32;
pub const LINE_SIZE: usize = 8;
pub const AVX2_LINE_SIZE: usize = 4;

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
unsafe fn abs_pd_512(x: __m512d) -> __m512d {
    // Clear sign bit: abs(x) == andnot(-0.0, x)
    let sign = _mm512_set1_pd(-0.0);
    _mm512_andnot_pd(sign, x)
}

#[inline(always)]
unsafe fn abs_pd_256(x: __m256d) -> __m256d {
    // Clear sign bit: abs(x) == andnot(-0.0, x)
    let sign = _mm256_set1_pd(-0.0);
    _mm256_andnot_pd(sign, x)
}

pub trait Avx512F64LeafOps {
    unsafe fn init_512(delta: __m512d) -> __m512d;

    unsafe fn accum_512(acc: __m512d, delta: __m512d) -> __m512d;

    unsafe fn init_256(delta: __m256d) -> __m256d;

    unsafe fn accum_256(acc: __m256d, delta: __m256d) -> __m256d;

    fn init_scalar(delta: f64) -> f64;

    fn accum_scalar(acc: f64, delta: f64) -> f64;
}

pub trait DistanceMetricUnifiedF64 {
    type Avx512F64Ops: Avx512F64LeafOps;
}

pub struct SquaredEuclidean;
pub struct Manhattan;

pub struct SquaredEuclideanAvx512Ops;
pub struct ManhattanAvx512Ops;

impl Avx512F64LeafOps for SquaredEuclideanAvx512Ops {
    #[inline(always)]
    unsafe fn init_512(delta: __m512d) -> __m512d {
        _mm512_mul_pd(delta, delta)
    }

    #[inline(always)]
    unsafe fn accum_512(acc: __m512d, delta: __m512d) -> __m512d {
        _mm512_fmadd_pd(delta, delta, acc)
    }

    #[inline(always)]
    unsafe fn init_256(delta: __m256d) -> __m256d {
        _mm256_mul_pd(delta, delta)
    }

    #[inline(always)]
    unsafe fn accum_256(acc: __m256d, delta: __m256d) -> __m256d {
        _mm256_fmadd_pd(delta, delta, acc)
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

impl Avx512F64LeafOps for ManhattanAvx512Ops {
    #[inline(always)]
    unsafe fn init_512(delta: __m512d) -> __m512d {
        abs_pd_512(delta)
    }

    #[inline(always)]
    unsafe fn accum_512(acc: __m512d, delta: __m512d) -> __m512d {
        _mm512_add_pd(acc, abs_pd_512(delta))
    }

    #[inline(always)]
    unsafe fn init_256(delta: __m256d) -> __m256d {
        abs_pd_256(delta)
    }

    #[inline(always)]
    unsafe fn accum_256(acc: __m256d, delta: __m256d) -> __m256d {
        _mm256_add_pd(acc, abs_pd_256(delta))
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
    type Avx512F64Ops = SquaredEuclideanAvx512Ops;
}

impl DistanceMetricUnifiedF64 for Manhattan {
    type Avx512F64Ops = ManhattanAvx512Ops;
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

    // we know if we get here then at least 1 of the bits of combined is set
    // otherwise we would have returned at the early <= check.
    // Hinting this to the compiler will allow us to elide an
    // unnecessary defensive check in the resulting asm output
    core::hint::assert_unchecked(combined != 0);
    let idx = combined.trailing_zeros() as usize;

    (chunk_min, *items.add(base + idx))
}

#[inline(always)]
unsafe fn update_best_line_avx512_raw(
    d0: __m512d,
    items: *const u64,
    base: usize,
    best_dist: f64,
    best_item: u64,
) -> (f64, u64) {
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

    // Hint to compiler that eq0 will never be zero here
    // without this, the compiler will add an extra instruction
    // to set the MSB of eq0 as a branchless fallback
    // so that trailing_zeros (tzcnt) can't
    // return an undefined / large value.
    core::hint::assert_unchecked(eq0 != 0);
    let idx = eq0.trailing_zeros() as usize;

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
    let idx = eq0.trailing_zeros() as usize;

    (chunk_min, *items.add(base + idx))
}

macro_rules! impl_leaf_kernel_k {
    ($extern_name:ident, $k:expr, [$(($dim:literal, $p:ident, $qs:ident, $qv:ident)),*]) => {
        #[target_feature(enable = "avx512f,avx512vl,fma")]
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
                        let mut acc = <M as
  DistanceMetricUnifiedF64>::Avx512F64Ops::init_512(d0);
                        $(
                            let a = _mm512_loadu_pd($p.add(base + $off * 8));
                            let d = _mm512_sub_pd(a, $qv);
                            acc = <M as
  DistanceMetricUnifiedF64>::Avx512F64Ops::accum_512(acc, d);
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
                let mut acc = <M as DistanceMetricUnifiedF64>::Avx512F64Ops::init_512(d0);

                $(
                    let a = _mm512_loadu_pd($p.add(base));
                    let d = _mm512_sub_pd(a, $qv);
                    acc = <M as DistanceMetricUnifiedF64>::Avx512F64Ops::accum_512(acc, d);
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
                let mut acc = <M as DistanceMetricUnifiedF64>::Avx512F64Ops::init_256(d0);

                $(
                    let qy = _mm512_castpd512_pd256($qv);
                    let a = _mm256_loadu_pd($p.add(base));
                    let d = _mm256_sub_pd(a, qy);
                    acc = <M as DistanceMetricUnifiedF64>::Avx512F64Ops::accum_256(acc, d);
                )*

                (best_dist, best_item) =
                    update_best_line_avx2_raw(acc, items, base, best_dist, best_item);

                base += AVX2_LINE_SIZE;
            }

            for idx in base..len {
                #[allow(unused_mut)]
                let mut d = <M as
  DistanceMetricUnifiedF64>::Avx512F64Ops::init_scalar(*p0.add(idx) - q0s);
                $(
                    d = <M as
  DistanceMetricUnifiedF64>::Avx512F64Ops::accum_scalar(d, *$p.add(idx) - $qs);
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
    type Ops<M> = <M as DistanceMetricUnifiedF64>::Avx512F64Ops;

    let points = std::slice::from_raw_parts(points, k);
    let items = std::slice::from_raw_parts(items, len);
    let query = std::slice::from_raw_parts(query, k);

    for idx in 0..len {
        let mut d = <M as DistanceMetricUnifiedF64>::Avx512F64Ops::init_scalar(
            *(*points.get_unchecked(0)).add(idx) - *query.get_unchecked(0),
        );
        for dim in 1..k {
            d = <M as DistanceMetricUnifiedF64>::Avx512F64Ops::accum_scalar(
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
