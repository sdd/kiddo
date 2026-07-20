#![allow(clippy::type_complexity, clippy::unit_arg)]
#[cfg(feature = "simulator")]
pub mod cache_simulator;
pub mod exact_query_stats;
pub mod exact_query_trace;

/// Result collector forced by nearest-n threshold crossover benchmarks.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NearestNBenchmarkCollector {
    BinaryHeap,
    ThresholdVecFused,
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
