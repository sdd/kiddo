#![allow(clippy::type_complexity, clippy::unit_arg)]
use fixed::types::extra::Unsigned;
use fixed::FixedU16;
use rand::distr::{Distribution, StandardUniform};
use std::array;

use crate::traits::{Axis, AxisFixed, Content};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// Pick a fixed seed for all benches so the same points/queries are reused.
const RNG_SEED: u64 = 42;

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

pub fn build_query_points_float<A: Axis, const K: usize>(points_qty: usize) -> Vec<[A; K]>
where
    StandardUniform: Distribution<[A; K]>,
{
    let mut rng = StdRng::seed_from_u64(RNG_SEED);

    (0..points_qty).map(|_| rng.random::<[A; K]>()).collect()
}
