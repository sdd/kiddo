use az::Cast;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput, BenchmarkGroup, PlotConfiguration, AxisScale, BatchSize};
use criterion::measurement::WallTime;
use fixed::FixedU16;
use fixed::types::extra::U16;
use rand::distributions::Standard;

use rand_distr::Distribution;
// use rand_distr::UnitSphere as SPHERE;

use sok::distance::squared_euclidean;
use sok::fixed::distance::squared_euclidean as squared_euclidean_fixedpoint;
use sok::float::kdtree::{Axis, Content, Index, KdTree};
use sok::fixed::kdtree::{Content as ContentFixed, Index as IndexFixed, KdTree as FixedKdTree};

const BUCKET_SIZE: usize = 32;
const QUERY_POINTS_PER_LOOP: usize = 1000;

type FXP = FixedU16<U16>;

/*fn rand_unit_sphere_point_f64() -> [f64; 3] {
    SPHERE.sample(&mut rand::thread_rng())
}

fn rand_sphere_data() -> ([f64; 3], usize) {
    (rand_unit_sphere_point_f64(), rand::random())
}*/

fn rand_data_fxp() -> FXP {
    let val: u16 = rand::random();
    unsafe { std::mem::transmute(val) }
}

fn rand_data_fixed_u16_3d_point<T: ContentFixed>() -> [FXP; 3] {
    [rand_data_fxp(), rand_data_fxp(), rand_data_fxp()]
}

fn rand_data_fixed_u16_3d_entry<T: ContentFixed>() -> ([FXP; 3], T) where Standard: Distribution<T> {
    (rand_data_fixed_u16_3d_point::<T>(), rand::random())
}

macro_rules! bench_float {
    ($group:ident, $a:ty, $t:ty, $k:tt, $idx: ty, $size:tt, $subtype: expr) => {
        bench_query_nearest_one_float::<$a, $t, $k, $idx>(&mut $group, $size, QUERY_POINTS_PER_LOOP, $subtype);
    }
}

macro_rules! bench_fixed {
    ($group:ident, $a:ty, $t:ty, $k:tt, $idx:ty, $size:tt, $subtype: expr) => {
        bench_query_nearest_one_fixed_3d::<$t, $idx>(&mut $group, $size, QUERY_POINTS_PER_LOOP, $subtype);
    }
}

macro_rules! size_t_idx {
    ( $group:ident; $callee:ident; $a:ty|$k:tt; [$(($size:tt,$t:ty,$idx:ty)),+] ) => {
        { $($callee!($group, $a, $t, $k, $idx, $size, concat!($k, "D ", stringify!($a)));)* }
    }
}

macro_rules! batch_benches {
    ($group:ident, $callee:ident, [$(($a:ty, $k:tt)),+], $s_t_idx_list:tt ) => {
        { $(size_t_idx!($group; $callee; $a|$k; $s_t_idx_list );)* }
    }
}

pub fn nearest_one(c: &mut Criterion) {
    let mut group = c.benchmark_group("Query Nearest 1");
    group.throughput(Throughput::Elements(QUERY_POINTS_PER_LOOP as u64));

    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    group.plot_config(plot_config);

    batch_benches!(group, bench_float,
        [(f64, 2), (f64, 3), (f64, 4), (f32, 3)],
        [(100, u16, u16), (1_000, u16, u16), (10_000, u16, u16), (100_000, u32, u16), (1_000_000, u32, u32)]
    );
    batch_benches!(group, bench_fixed,
        [(FXP, 3)],
        [(100, u16, u16), (1_000, u16, u16), (10_000, u16, u16), (100_000, u32, u16), (1_000_000, u32, u32)]
    );

    group.finish();
}

fn bench_query_nearest_one_float<A: Axis, T: Content, const K: usize, IDX: Index<T = IDX>>(group: &mut BenchmarkGroup<WallTime>, initial_size: usize, query_pts_per_loop: usize, subtype: &str) where usize: Cast<IDX>, Standard: Distribution<([A; K], T)>, Standard: Distribution<[A; K]> {
    group.bench_with_input(BenchmarkId::new(subtype, initial_size), &initial_size, |b, &size| {
        b.iter_batched(|| {
            let points_to_query: Vec<[A; K]> =
                (0..query_pts_per_loop).into_iter().map(|_| rand::random::<[A; K]>()).collect();

            let mut initial_points = vec![];
            for _ in 0..size {
                initial_points.push(rand::random::<([A; K], T)>());
            }
            let mut kdtree =
                KdTree::<A, T, K, BUCKET_SIZE, IDX>::with_capacity(size);

            for i in 0..initial_points.len() {
                kdtree.add(&initial_points[i].0, initial_points[i].1);
            }

            (kdtree, points_to_query)
        }, |(kdtree, points_to_query)| {
            black_box(points_to_query
                .iter()
                .for_each(|point| {
                    kdtree.nearest_one(&point, &squared_euclidean);
                }))
        }, BatchSize::SmallInput);
    });
}

fn bench_query_nearest_one_fixed_3d<T: ContentFixed, IDX: IndexFixed<T = IDX>>(group: &mut BenchmarkGroup<WallTime>, initial_size: usize, query_pts_per_loop: usize, subtype: &str) where usize: Cast<IDX>, Standard: Distribution<T> {
    group.bench_with_input(BenchmarkId::new(subtype, initial_size), &initial_size, |b, &size| {
        b.iter_batched(|| {
            let points_to_query: Vec<[_; 3]> =
                (0..query_pts_per_loop).into_iter().map(|_| rand_data_fixed_u16_3d_point::<T>()).collect();

            let mut initial_points = vec![];
            for _ in 0..size {
                initial_points.push(rand_data_fixed_u16_3d_entry::<T>());
            }
            let mut kdtree =
                FixedKdTree::<FXP, T, 3, BUCKET_SIZE, IDX>::with_capacity(size);

            for i in 0..initial_points.len() {
                kdtree.add(&initial_points[i].0, initial_points[i].1);
            }

            (kdtree, points_to_query)
        }, |(kdtree, points_to_query)| {
            black_box(points_to_query
                .iter()
                .for_each(|point| {
                    kdtree.nearest_one(&point, &squared_euclidean_fixedpoint);
                }))
        }, BatchSize::SmallInput);
    });
}

criterion_group!(benches, nearest_one);
criterion_main!(benches);
