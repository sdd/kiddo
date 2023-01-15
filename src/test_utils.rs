use std::array;
use fixed::FixedU16;
use fixed::types::extra::Unsigned;
use rand::distributions::{Distribution, Standard};

use crate::types::Content;
use crate::fixed::kdtree::{Axis as AxisFixed};

// use rand_distr::UnitSphere as SPHERE;

/*fn rand_unit_sphere_point_f64() -> [f64; 3] {
    SPHERE.sample(&mut rand::thread_rng())
}

fn rand_sphere_data() -> ([f64; 3], usize) {
    (rand_unit_sphere_point_f64(), rand::random())
}*/

pub fn rand_data_fixed_u16<A: Unsigned>() -> FixedU16<A> where FixedU16<A>: AxisFixed {
    let val: u16 = rand::random();
    unsafe { std::mem::transmute(val) }
}

pub fn rand_data_fixed_u16_point<A: Unsigned, const K: usize>() -> [FixedU16<A>; K] where FixedU16<A>: AxisFixed {
    array::from_fn(|_| rand_data_fixed_u16::<A>())
}

pub fn rand_data_fixed_u16_entry<A: Unsigned, T: Content, const K: usize>() -> ([FixedU16<A>; K], T) where Standard: Distribution<T>, FixedU16<A>: AxisFixed {
    (rand_data_fixed_u16_point::<A, K>(), rand::random())
}

#[macro_export]
macro_rules! size_t_idx {
    ( $group:ident; $callee:ident; $a:ty|$k:tt; [$(($size:tt,$t:ty,$idx:ty)),+] ) => {
        { $($callee!($group, $a, $t, $k, $idx, $size, concat!($k, "D ", stringify!($a)));)* }
    }
}

#[macro_export]
macro_rules! batch_benches {
    ($group:ident, $callee:ident, [$(($a:ty, $k:tt)),+], $s_t_idx_list:tt ) => {
        { $($crate::size_t_idx!($group; $callee; $a|$k; $s_t_idx_list );)* }
    }
}
