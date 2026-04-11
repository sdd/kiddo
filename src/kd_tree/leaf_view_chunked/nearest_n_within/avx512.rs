#![allow(clippy::missing_safety_doc)]

use std::arch::x86_64::*;

use array_init::array_init;

use crate::dist::distance_metric_avx512::Avx512F64LeafOps;
use crate::kd_tree::leaf_view::LeafView;
use crate::traits_unified_2::Basics;

const CHUNK_SIZE: usize = 32;
const LINE_SIZE: usize = 8;
const AVX2_LINE_SIZE: usize = 4;

#[inline(always)]
unsafe fn emit_results_avx512<T, F>(
    dists: __m512d,
    items: *const T,
    base: usize,
    max_dist: f64,
    emit: &mut F,
) where
    T: Basics,
    F: FnMut(f64, T),
{
    let mask = _mm512_cmp_pd_mask(dists, _mm512_set1_pd(max_dist), _CMP_LE_OQ) as u32;
    if mask == 0 {
        return;
    }

    let mut dist_values = [0.0f64; LINE_SIZE];
    _mm512_storeu_pd(dist_values.as_mut_ptr(), dists);

    let mut remaining = mask;
    while remaining != 0 {
        let lane = remaining.trailing_zeros() as usize;
        emit(
            *dist_values.get_unchecked(lane),
            std::ptr::read_unaligned(items.add(base + lane)),
        );
        remaining &= remaining - 1;
    }
}

#[inline(always)]
unsafe fn emit_results_avx2<T, F>(
    dists: __m256d,
    items: *const T,
    base: usize,
    max_dist: f64,
    emit: &mut F,
) where
    T: Basics,
    F: FnMut(f64, T),
{
    let mask =
        _mm256_movemask_pd(_mm256_cmp_pd(dists, _mm256_set1_pd(max_dist), _CMP_LE_OQ)) as u32;
    if mask == 0 {
        return;
    }

    let mut dist_values = [0.0f64; AVX2_LINE_SIZE];
    _mm256_storeu_pd(dist_values.as_mut_ptr(), dists);

    let mut remaining = mask;
    while remaining != 0 {
        let lane = remaining.trailing_zeros() as usize;
        emit(
            *dist_values.get_unchecked(lane),
            std::ptr::read_unaligned(items.add(base + lane)),
        );
        remaining &= remaining - 1;
    }
}

#[inline(always)]
unsafe fn emit_results_avx128<T, F>(
    dists: __m128d,
    items: *const T,
    base: usize,
    max_dist: f64,
    emit: &mut F,
) where
    T: Basics,
    F: FnMut(f64, T),
{
    let mask = _mm_movemask_pd(_mm_cmple_pd(dists, _mm_set1_pd(max_dist))) as u32;
    if mask == 0 {
        return;
    }

    let mut dist_values = [0.0f64; 2];
    _mm_storeu_pd(dist_values.as_mut_ptr(), dists);

    let mut remaining = mask;
    while remaining != 0 {
        let lane = remaining.trailing_zeros() as usize;
        emit(
            *dist_values.get_unchecked(lane),
            std::ptr::read_unaligned(items.add(base + lane)),
        );
        remaining &= remaining - 1;
    }
}

#[inline(always)]
unsafe fn line_dists_avx512<L, const K: usize>(
    points: &[*const f64; K],
    query: &[__m512d; K],
    base: usize,
) -> __m512d
where
    L: Avx512F64LeafOps,
{
    let a0 = _mm512_loadu_pd(points[0].add(base));
    let d0 = _mm512_sub_pd(a0, query[0]);
    let mut acc = L::dist_k0_f64x8(d0);

    for dim in 1..K {
        let a = _mm512_loadu_pd(points[dim].add(base));
        let d = _mm512_sub_pd(a, query[dim]);
        acc = L::dist_kn_f64x8(acc, d);
    }

    acc
}

#[inline(always)]
unsafe fn line_dists_avx2<L, const K: usize>(
    points: &[*const f64; K],
    query: &[__m256d; K],
    base: usize,
) -> __m256d
where
    L: Avx512F64LeafOps,
{
    let a0 = _mm256_loadu_pd(points[0].add(base));
    let d0 = _mm256_sub_pd(a0, query[0]);
    let mut acc = L::dist_k0_f64x4(d0);

    for dim in 1..K {
        let a = _mm256_loadu_pd(points[dim].add(base));
        let d = _mm256_sub_pd(a, query[dim]);
        acc = L::dist_kn_f64x4(acc, d);
    }

    acc
}

#[inline(always)]
unsafe fn line_dists_avx128<L, const K: usize>(
    points: &[*const f64; K],
    query: &[__m128d; K],
    base: usize,
) -> __m128d
where
    L: Avx512F64LeafOps,
{
    let a0 = _mm_loadu_pd(points[0].add(base));
    let d0 = _mm_sub_pd(a0, query[0]);
    let mut acc = L::dist_k0_f64x2(d0);

    for dim in 1..K {
        let a = _mm_loadu_pd(points[dim].add(base));
        let d = _mm_sub_pd(a, query[dim]);
        acc = L::dist_kn_f64x2(acc, d);
    }

    acc
}

#[inline(always)]
unsafe fn dist_scalar<L, const K: usize>(
    points: &[*const f64; K],
    query: &[f64; K],
    idx: usize,
) -> f64
where
    L: Avx512F64LeafOps,
{
    let mut dist = L::dist_k0_f64x1(*points[0].add(idx) - query[0]);
    for dim in 1..K {
        dist = L::dist_kn_f64x1(dist, *points[dim].add(idx) - query[dim]);
    }
    dist
}

#[target_feature(enable = "avx512f,avx512vl,fma")]
unsafe fn nearest_n_within_avx512_raw<L, T, F, const K: usize>(
    points: [*const f64; K],
    items: *const T,
    len: usize,
    query: &[f64; K],
    max_dist: f64,
    emit: &mut F,
) where
    L: Avx512F64LeafOps,
    T: Basics,
    F: FnMut(f64, T),
{
    if len == 0 {
        return;
    }

    let query_512 = array_init(|dim| _mm512_set1_pd(query[dim]));
    let query_256 = array_init(|dim| _mm256_set1_pd(query[dim]));
    let query_128 = array_init(|dim| _mm_set1_pd(query[dim]));

    let full_chunks_len = len & !(CHUNK_SIZE - 1);
    let mut base = 0usize;

    while base != full_chunks_len {
        let d0 = line_dists_avx512::<L, K>(&points, &query_512, base);
        let d1 = line_dists_avx512::<L, K>(&points, &query_512, base + LINE_SIZE);
        let d2 = line_dists_avx512::<L, K>(&points, &query_512, base + 2 * LINE_SIZE);
        let d3 = line_dists_avx512::<L, K>(&points, &query_512, base + 3 * LINE_SIZE);

        emit_results_avx512(d0, items, base, max_dist, emit);
        emit_results_avx512(d1, items, base + LINE_SIZE, max_dist, emit);
        emit_results_avx512(d2, items, base + 2 * LINE_SIZE, max_dist, emit);
        emit_results_avx512(d3, items, base + 3 * LINE_SIZE, max_dist, emit);

        base += CHUNK_SIZE;
    }

    let full_lines_len = full_chunks_len + ((len - full_chunks_len) & !(LINE_SIZE - 1));
    while base != full_lines_len {
        let d0 = line_dists_avx512::<L, K>(&points, &query_512, base);
        emit_results_avx512(d0, items, base, max_dist, emit);
        base += LINE_SIZE;
    }

    if base + AVX2_LINE_SIZE <= len {
        let d0 = line_dists_avx2::<L, K>(&points, &query_256, base);
        emit_results_avx2(d0, items, base, max_dist, emit);
        base += AVX2_LINE_SIZE;
    }

    if base + 2 <= len {
        let d0 = line_dists_avx128::<L, K>(&points, &query_128, base);
        emit_results_avx128(d0, items, base, max_dist, emit);
        base += 2;
    }

    for idx in base..len {
        let dist = dist_scalar::<L, K>(&points, query, idx);
        if dist <= max_dist {
            emit(dist, std::ptr::read_unaligned(items.add(idx)));
        }
    }
}

#[target_feature(enable = "avx512f,avx512vl,fma")]
pub(crate) unsafe fn nearest_n_within_avx512_unchecked<L, T, F, const K: usize, const B: usize>(
    leaf: &LeafView<'_, f64, T, K, B>,
    query: &[f64; K],
    max_dist: f64,
    emit: &mut F,
) where
    L: Avx512F64LeafOps,
    T: Basics,
    F: FnMut(f64, T),
{
    let points = leaf.points();
    let point_ptrs = array_init(|dim| points[dim].as_ptr());
    nearest_n_within_avx512_raw::<L, T, F, K>(
        point_ptrs,
        leaf.items().as_ptr(),
        leaf.items().len(),
        query,
        max_dist,
        emit,
    );
}

#[target_feature(enable = "avx512f,avx512vl,fma")]
pub(crate) unsafe fn nearest_n_within_avx512_arena_unchecked<L, T, F, const K: usize>(
    tile_base: *const u8,
    len: usize,
    query: &[f64; K],
    max_dist: f64,
    emit: &mut F,
) where
    L: Avx512F64LeafOps,
    T: Basics,
    F: FnMut(f64, T),
{
    let point_base = tile_base as *const f64;
    let point_ptrs = array_init(|dim| point_base.add(dim * len));
    let items = tile_base.add(K * len * std::mem::size_of::<f64>()) as *const T;

    nearest_n_within_avx512_raw::<L, T, F, K>(point_ptrs, items, len, query, max_dist, emit);
}
