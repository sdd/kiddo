#![allow(clippy::missing_safety_doc)]
#![allow(clippy::too_many_arguments)]
#![allow(unsafe_op_in_unsafe_fn)]

use std::arch::aarch64::*;

pub const CHUNK_SIZE: usize = 8;

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

pub trait NeonF64LeafOps {
    unsafe fn init(delta: float64x2_t) -> float64x2_t;

    unsafe fn accum(acc: float64x2_t, delta: float64x2_t) -> float64x2_t;

    fn init_scalar(delta: f64) -> f64;

    fn accum_scalar(acc: f64, delta: f64) -> f64;
}

pub trait DistanceMetricUnifiedF64 {
    type NeonF64Ops: NeonF64LeafOps;
}

pub struct SquaredEuclidean;
pub struct Manhattan;

pub struct SquaredEuclideanNeonOps;
pub struct ManhattanNeonOps;

impl NeonF64LeafOps for SquaredEuclideanNeonOps {
    #[inline(always)]
    unsafe fn init(delta: float64x2_t) -> float64x2_t {
        vmulq_f64(delta, delta)
    }

    #[inline(always)]
    unsafe fn accum(acc: float64x2_t, delta: float64x2_t) -> float64x2_t {
        // Keep this as mul+add for portability in Godbolt snippets.
        vaddq_f64(acc, vmulq_f64(delta, delta))
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

impl NeonF64LeafOps for ManhattanNeonOps {
    #[inline(always)]
    unsafe fn init(delta: float64x2_t) -> float64x2_t {
        vabsq_f64(delta)
    }

    #[inline(always)]
    unsafe fn accum(acc: float64x2_t, delta: float64x2_t) -> float64x2_t {
        vaddq_f64(acc, vabsq_f64(delta))
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
    type NeonF64Ops = SquaredEuclideanNeonOps;
}

impl DistanceMetricUnifiedF64 for Manhattan {
    type NeonF64Ops = ManhattanNeonOps;
}

#[inline(always)]
unsafe fn mask2_u32(mask: uint64x2_t) -> u32 {
    let b0 = (vgetq_lane_u64(mask, 0) != 0) as u32;
    let b1 = (vgetq_lane_u64(mask, 1) != 0) as u32;
    b0 | (b1 << 1)
}

#[inline(always)]
unsafe fn update_best_chunk_neon_raw(
    d0: float64x2_t,
    d1: float64x2_t,
    d2: float64x2_t,
    d3: float64x2_t,
    items: *const u64,
    base: usize,
    best_dist: f64,
    best_item: u64,
) -> (f64, u64) {
    let bb = vdupq_n_f64(best_dist);
    let m0 = vcltq_f64(d0, bb);
    let m1 = vcltq_f64(d1, bb);
    let m2 = vcltq_f64(d2, bb);
    let m3 = vcltq_f64(d3, bb);

    let any = vorrq_u64(vorrq_u64(m0, m1), vorrq_u64(m2, m3));
    if (vgetq_lane_u64(any, 0) | vgetq_lane_u64(any, 1)) == 0 {
        return (best_dist, best_item);
    }

    let min01 = vminq_f64(d0, d1);
    let min23 = vminq_f64(d2, d3);
    let min0123 = vminq_f64(min01, min23);
    let chunk_min = f64::min(vgetq_lane_f64(min0123, 0), vgetq_lane_f64(min0123, 1));

    let bcast = vdupq_n_f64(chunk_min);
    let eq0 = mask2_u32(vceqq_f64(d0, bcast));
    let eq1 = mask2_u32(vceqq_f64(d1, bcast));
    let eq2 = mask2_u32(vceqq_f64(d2, bcast));
    let eq3 = mask2_u32(vceqq_f64(d3, bcast));

    let combined = eq0 | (eq1 << 2) | (eq2 << 4) | (eq3 << 6);
    let idx = combined.trailing_zeros() as usize;

    (chunk_min, *items.add(base + idx))
}

macro_rules! impl_leaf_kernel_k {
    ($extern_name:ident, $k:expr, [$(($dim:literal, $p:ident, $qs:ident, $qv:ident)),*]) => {
        #[target_feature(enable = "neon")]
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

            let qv0 = vdupq_n_f64(q0s);
            $(
                let $qv = vdupq_n_f64($qs);
            )*

            let full_chunks_len = len & !(CHUNK_SIZE - 1);
            let mut base = 0usize;
            while base < full_chunks_len {
                macro_rules! chunk {
                    ($off:expr) => {{
                        let a0 = vld1q_f64(p0.add(base + $off * 2));
                        let d0 = vsubq_f64(a0, qv0);
                        let mut acc = <M as DistanceMetricUnifiedF64>::NeonF64Ops::init(d0);
                        $(
                            let a = vld1q_f64($p.add(base + $off * 2));
                            let d = vsubq_f64(a, $qv);
                            acc = <M as DistanceMetricUnifiedF64>::NeonF64Ops::accum(acc, d);
                        )*
                        acc
                    }};
                }

                let d0 = chunk!(0);
                let d1 = chunk!(1);
                let d2 = chunk!(2);
                let d3 = chunk!(3);

                (best_dist, best_item) =
                    update_best_chunk_neon_raw(d0, d1, d2, d3, items, base, best_dist, best_item);

                base += CHUNK_SIZE;
            }

            for idx in base..len {
                let mut d = <M as DistanceMetricUnifiedF64>::NeonF64Ops::init_scalar(*p0.add(idx) - q0s);
                $(
                    d = <M as DistanceMetricUnifiedF64>::NeonF64Ops::accum_scalar(d, *$p.add(idx) - $qs);
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
    let points = std::slice::from_raw_parts(points, k);
    let items = std::slice::from_raw_parts(items, len);
    let query = std::slice::from_raw_parts(query, k);

    for idx in 0..len {
        let mut d = <M as DistanceMetricUnifiedF64>::NeonF64Ops::init_scalar(
            *(*points.get_unchecked(0)).add(idx) - *query.get_unchecked(0),
        );
        for dim in 1..k {
            d = <M as DistanceMetricUnifiedF64>::NeonF64Ops::accum_scalar(
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
pub unsafe extern "C" fn leaf_nearest_one_chunked_nozero_f64(
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
