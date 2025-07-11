#![allow(clippy::type_complexity, clippy::unit_arg)]
use az::Cast;
use fixed::types::extra::Unsigned;
use fixed::FixedU16;
use rand::distr::{Distribution, StandardUniform};
use rayon::iter::ParallelIterator;
use rayon::prelude::IntoParallelRefIterator;
use std::array;
use std::hint::black_box;

use crate::fixed::kdtree::{Axis as AxisFixed, KdTree as FixedKdTree};
use crate::float::kdtree::{Axis, KdTree};
use crate::float_leaf_slice::leaf_slice::{LeafSliceFloat, LeafSliceFloatChunk};
use crate::immutable::float::kdtree::ImmutableKdTree;
use crate::traits::{Content, Index};

// use rand_distr::UnitSphere as SPHERE;

/*fn rand_unit_sphere_point_f64() -> [f64; 3] {
    SPHERE.sample(&mut rand::rng())
}

fn rand_sphere_data() -> ([f64; 3], usize) {
    (rand_unit_sphere_point_f64(), rand::random())
}*/

pub fn rand_data_fixed_u16<A: Unsigned>() -> FixedU16<A>
where
    FixedU16<A>: AxisFixed,
{
    let val: u16 = rand::random();
    unsafe { std::mem::transmute(val) }
}

pub fn rand_data_fixed_u16_point<A: Unsigned, const K: usize>() -> [FixedU16<A>; K]
where
    FixedU16<A>: AxisFixed,
{
    array::from_fn(|_| rand_data_fixed_u16::<A>())
}

pub fn rand_data_fixed_u16_entry<A: Unsigned, T: Content, const K: usize>() -> ([FixedU16<A>; K], T)
where
    StandardUniform: Distribution<T>,
    FixedU16<A>: AxisFixed,
{
    (rand_data_fixed_u16_point::<A, K>(), rand::random())
}

#[doc(hidden)]
#[macro_export]
macro_rules! size_t_idx {
    ( $group:ident; $callee:ident; $a:ty|$k:tt; [$(($size:tt,$t:ty,$idx:ty)),+] ) => {
        { $($callee!($group, $a, $t, $k, $idx, $size, concat!($k, "D ", stringify!($a)));)* }
    }
}

#[doc(hidden)]
#[macro_export]
macro_rules! batch_benches {
    ($group:ident, $callee:ident, [$(($a:ty, $k:tt)),+], $s_t_idx_list:tt ) => {
        { $($crate::size_t_idx!($group; $callee; $a|$k; $s_t_idx_list );)* }
    }
}

#[doc(hidden)]
#[macro_export]
macro_rules! size_t_idx_parameterized {
    ( $group:ident; $callee:ident; $param:tt;  $a:ty|$k:tt; [$(($size:tt,$t:ty,$idx:ty)),+] ) => {
        { $($callee!($group, $a, $t, $k, $idx, $size, $param, concat!($k, "D ", stringify!($a)));)* }
    }
}

#[doc(hidden)]
#[macro_export]
macro_rules! batch_benches_parameterized {
    ($group:ident, $callee:ident, $param:tt,  [$(($a:ty, $k:tt)),+], $s_t_idx_list:tt ) => {
        { $($crate::size_t_idx_parameterized!($group; $callee; $param; $a|$k; $s_t_idx_list );)* }
    }
}

pub fn build_populated_tree_fixed<
    A: Unsigned,
    T: Content,
    const K: usize,
    const B: usize,
    IDX: Index<T = IDX>,
>(
    size: usize,
    spare_capacity: usize,
) -> FixedKdTree<FixedU16<A>, T, K, B, IDX>
where
    usize: Cast<IDX>,
    StandardUniform: Distribution<T>,
    FixedU16<A>: AxisFixed,
{
    let mut kdtree = FixedKdTree::<FixedU16<A>, T, K, B, IDX>::with_capacity(size + spare_capacity);

    for _ in 0..size {
        let entry = rand_data_fixed_u16_entry::<A, T, K>();
        kdtree.add(&entry.0, entry.1);
    }

    kdtree
}

pub fn build_populated_tree_float<
    A: Axis,
    T: Content,
    const K: usize,
    const B: usize,
    IDX: Index<T = IDX>,
>(
    size: usize,
    spare_capacity: usize,
) -> KdTree<A, T, K, B, IDX>
where
    usize: Cast<IDX>,
    StandardUniform: Distribution<T>,
    StandardUniform: Distribution<([A; K], T)>,
    StandardUniform: Distribution<[A; K]>,
{
    let mut kdtree = KdTree::<A, T, K, B, IDX>::with_capacity(size + spare_capacity);

    for _ in 0..size {
        let entry = rand::random::<([A; K], T)>();
        kdtree.add(&entry.0, entry.1);
    }

    kdtree
}

pub fn build_populated_tree_immutable_float<A, T: Content, const K: usize, const B: usize>(
    size: usize,
) -> ImmutableKdTree<A, T, K, B>
where
    usize: Cast<T>,
    StandardUniform: Distribution<T>,
    StandardUniform: Distribution<([A; K], T)>,
    StandardUniform: Distribution<[A; K]>,
    A: Axis + LeafSliceFloat<T> + LeafSliceFloatChunk<T, K>,
{
    let mut points = vec![];
    points.resize_with(size, rand::random::<[A; K]>);

    ImmutableKdTree::<A, T, K, B>::new_from_slice(&points)
}

/*
pub fn build_populated_tree_float_sss<
    A: AxisSSS,
    T: Content,
    const K: usize,
    const B: usize,
    IDX: Index<T = IDX>,
>(
    size: usize,
    spare_capacity: usize,
) -> KdTreeSSS<A, T, K, B, IDX>
where
    usize: Cast<IDX>,
    StandardUniform: Distribution<T>,
    StandardUniform: Distribution<([A; K], T)>,
    StandardUniform: Distribution<[A; K]>,
{
    let mut kdtree = KdTreeSSS::<A, T, K, B, IDX>::with_capacity(size + spare_capacity);

    for _ in 0..size {
        let entry = rand::random::<([A; K], T)>();
        kdtree.add(&entry.0, entry.1);
    }

    kdtree
}
*/

pub fn build_query_points_fixed<A: Unsigned, const K: usize>(
    points_qty: usize,
) -> Vec<[FixedU16<A>; K]>
where
    FixedU16<A>: AxisFixed,
{
    (0..points_qty)
        .map(|_| rand_data_fixed_u16_point::<A, K>())
        .collect()
}

pub fn build_populated_tree_and_query_points_fixed<
    A: Unsigned,
    T: Content,
    const K: usize,
    const B: usize,
    IDX: Index<T = IDX>,
>(
    size: usize,
    query_point_qty: usize,
) -> (
    FixedKdTree<FixedU16<A>, T, K, B, IDX>,
    Vec<[FixedU16<A>; K]>,
)
where
    usize: Cast<IDX>,
    StandardUniform: Distribution<T>,
    FixedU16<A>: AxisFixed,
{
    (
        build_populated_tree_fixed(size, 0),
        build_query_points_fixed(query_point_qty),
    )
}

pub fn process_queries_fixed<
    A: Unsigned,
    T: Content,
    const K: usize,
    const B: usize,
    IDX: Index<T = IDX>,
    F,
>(
    query: F,
) -> Box<
    dyn Fn(
        (
            FixedKdTree<FixedU16<A>, T, K, B, IDX>,
            Vec<[FixedU16<A>; K]>,
        ),
    ),
>
where
    usize: Cast<IDX>,
    FixedU16<A>: AxisFixed,
    F: Fn(&FixedKdTree<FixedU16<A>, T, K, B, IDX>, &[FixedU16<A>; K]) + 'static,
{
    Box::new(
        move |(kdtree, points_to_query): (
            FixedKdTree<FixedU16<A>, T, K, B, IDX>,
            Vec<[FixedU16<A>; K]>,
        )| {
            black_box(
                points_to_query
                    .iter()
                    .for_each(|point| black_box(query(&kdtree, point))),
            )
        },
    )
}

pub fn process_queries_fixed_parameterized<
    A: Unsigned,
    T: Content,
    const K: usize,
    const B: usize,
    IDX: Index<T = IDX>,
    F,
>(
    query: F,
    param: f64,
) -> Box<
    dyn Fn(
        (
            FixedKdTree<FixedU16<A>, T, K, B, IDX>,
            Vec<[FixedU16<A>; K]>,
        ),
    ),
>
where
    usize: Cast<IDX>,
    FixedU16<A>: AxisFixed,
    F: Fn(&FixedKdTree<FixedU16<A>, T, K, B, IDX>, &[FixedU16<A>; K], f64) + 'static,
{
    Box::new(
        move |(kdtree, points_to_query): (
            FixedKdTree<FixedU16<A>, T, K, B, IDX>,
            Vec<[FixedU16<A>; K]>,
        )| {
            black_box(
                points_to_query
                    .iter()
                    .for_each(|point| black_box(query(&kdtree, point, param))),
            )
        },
    )
}

pub fn build_query_points_float<A: Axis, const K: usize>(points_qty: usize) -> Vec<[A; K]>
where
    StandardUniform: Distribution<[A; K]>,
{
    (0..points_qty).map(|_| rand::random::<[A; K]>()).collect()
}

pub fn build_populated_tree_and_query_points_float<
    A: Axis,
    T: Content,
    const K: usize,
    const B: usize,
    IDX: Index<T = IDX>,
>(
    size: usize,
    query_point_qty: usize,
) -> (KdTree<A, T, K, B, IDX>, Vec<[A; K]>)
where
    usize: Cast<IDX>,
    StandardUniform: Distribution<T>,
    StandardUniform: Distribution<[A; K]>,
{
    (
        build_populated_tree_float(size, 0),
        build_query_points_float(query_point_qty),
    )
}

pub fn build_populated_tree_and_query_points_immutable_float<
    A,
    T: Content,
    const K: usize,
    const B: usize,
>(
    size: usize,
    query_point_qty: usize,
) -> (ImmutableKdTree<A, T, K, B>, Vec<[A; K]>)
where
    A: Axis + LeafSliceFloat<T> + LeafSliceFloatChunk<T, K>,
    usize: Cast<T>,
    StandardUniform: Distribution<T>,
    StandardUniform: Distribution<[A; K]>,
{
    (
        build_populated_tree_immutable_float(size),
        build_query_points_float(query_point_qty),
    )
}

/*
pub fn build_populated_tree_and_query_points_float_sss<
    A: AxisSSS,
    T: Content,
    const K: usize,
    const B: usize,
    IDX: Index<T = IDX>,
>(
    size: usize,
    query_point_qty: usize,
) -> (KdTreeSSS<A, T, K, B, IDX>, Vec<[A; K]>)
where
    usize: Cast<IDX>,
    StandardUniform: Distribution<T>,
    StandardUniform: Distribution<[A; K]>,
{
    (
        build_populated_tree_float_sss(size, 0),
        build_query_points_float(query_point_qty),
    )
}
*/

#[inline]
pub fn process_queries_float<
    A: Axis + 'static,
    T: Content,
    const K: usize,
    const B: usize,
    IDX: Index<T = IDX>,
    F,
>(
    query: F,
) -> Box<dyn Fn((KdTree<A, T, K, B, IDX>, Vec<[A; K]>))>
where
    usize: Cast<IDX>,
    StandardUniform: Distribution<T>,
    StandardUniform: Distribution<[A; K]>,
    F: Fn(&KdTree<A, T, K, B, IDX>, &[A; K]) + 'static + Sync,
{
    Box::new(
        move |(kdtree, points_to_query): (KdTree<A, T, K, B, IDX>, Vec<[A; K]>)| {
            black_box(
                points_to_query
                    .par_iter()
                    .for_each(|point| black_box(query(&kdtree, point))),
            )
        },
    )
}

#[inline]
pub fn process_queries_immutable_float<
    A: Axis + 'static,
    T: Content,
    const K: usize,
    const B: usize,
    F,
>(
    query: F,
) -> Box<dyn Fn((ImmutableKdTree<A, T, K, B>, Vec<[A; K]>))>
where
    usize: Cast<T>,
    StandardUniform: Distribution<T>,
    StandardUniform: Distribution<[A; K]>,
    F: Fn(&ImmutableKdTree<A, T, K, B>, &[A; K]) + 'static + Sync,
{
    Box::new(
        move |(kdtree, points_to_query): (ImmutableKdTree<A, T, K, B>, Vec<[A; K]>)| {
            black_box(
                points_to_query
                    .par_iter()
                    .for_each(|point| black_box(query(&kdtree, point))),
            )
        },
    )
}

/*
#[inline]
pub fn process_queries_float_sss<
    A: AxisSSS + 'static,
    T: Content,
    const K: usize,
    const B: usize,
    IDX: Index<T = IDX>,
    F,
>(
    query: F,
) -> Box<dyn Fn((KdTreeSSS<A, T, K, B, IDX>, Vec<[A; K]>))>
where
    usize: Cast<IDX>,
    StandardUniform: Distribution<T>,
    StandardUniform: Distribution<[A; K]>,
    F: Fn(&KdTreeSSS<A, T, K, B, IDX>, &[A; K]) + 'static,
{
    Box::new(
        move |(kdtree, points_to_query): (KdTreeSSS<A, T, K, B, IDX>, Vec<[A; K]>)| {
            black_box(
                points_to_query
                    .iter()
                    .for_each(|point| black_box(query(&kdtree, point))),
            )
        },
    )
}
*/

pub fn process_queries_float_parameterized<
    A: Axis + 'static,
    T: Content,
    const K: usize,
    const B: usize,
    IDX: Index<T = IDX>,
    F,
>(
    query: F,
    param: f64,
) -> Box<dyn Fn((KdTree<A, T, K, B, IDX>, Vec<[A; K]>))>
where
    usize: Cast<IDX>,
    StandardUniform: Distribution<T>,
    StandardUniform: Distribution<[A; K]>,
    F: Fn(&KdTree<A, T, K, B, IDX>, &[A; K], f64) + 'static,
{
    Box::new(
        move |(kdtree, points_to_query): (KdTree<A, T, K, B, IDX>, Vec<[A; K]>)| {
            black_box(
                points_to_query
                    .iter()
                    .for_each(|point| black_box(query(&kdtree, point, param))),
            )
        },
    )
}
