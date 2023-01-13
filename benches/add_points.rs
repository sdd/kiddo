use az::Cast;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput, BenchmarkGroup, BatchSize, PlotConfiguration, AxisScale};
use criterion::measurement::WallTime;
use fixed::FixedU16;
use fixed::types::extra::U16;
use rand::distributions::{Distribution, Standard};

use sok::float::kdtree::{Axis, Content, Index, KdTree};
use sok::fixed::kdtree::{Content as ContentFixed, Index as IndexFixed, KdTree as FixedKdTree};

const BUCKET_SIZE: usize = 32;
const QTY_TO_ADD_TO_POPULATED: u64 = 100;

type FXP = FixedU16<U16>;

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

macro_rules! bench_empty_float {
    ($group:ident, $a:ty, $t:ty, $k:tt, $idx: ty, $size:tt, $subtype: expr) => {
        bench_add_to_empty_float::<$a, $t, $k, $idx>(&mut $group, $size, $subtype);
    }
}

macro_rules! bench_empty_fixed {
    ($group:ident, $a:ty, $t:ty, $k:tt, $idx:ty, $size:tt, $subtype: expr) => {
        bench_add_to_empty_fixed::<$t, $idx>(&mut $group, $size, $subtype);
    }
}

macro_rules! bench_populated_float {
    ($group:ident, $a:ty, $t:ty, $k:tt, $idx: ty, $size:tt, $subtype: expr) => {
        bench_add_to_populated_float::<$a, $t, $k, $idx>(&mut $group, $size, $subtype);
    }
}

macro_rules! bench_populated_fixed {
    ($group:ident, $a:ty, $t:ty, $k:tt, $idx:ty, $size:tt, $subtype: expr) => {
        bench_add_to_populated_fixed::<$t, $idx>(&mut $group, $size, $subtype);
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

pub fn add_to_empty(c: &mut Criterion) {
    let mut group = c.benchmark_group("Add to Empty Tree");

    let plot_config = PlotConfiguration::default()
        .summary_scale(AxisScale::Logarithmic);

    group.plot_config(plot_config);

    batch_benches!(group, bench_empty_fixed,
        [(FXP, 3)],
        [(100, u16, u16), (1_000, u16, u16), (10_000, u16, u16), (100_000, u32, u16), (1_000_000, u32, u32)]
    );

    batch_benches!(group, bench_empty_float,
        [(f64, 2), (f64, 3), (f64, 4), (f32, 3)],
        [(100, u16, u16), (1_000, u16, u16), (10_000, u16, u16), (100_000, u32, u16), (1_000_000, u32, u32)]
    );

    group.finish();
}

pub fn add_to_populated(c: &mut Criterion) {
    let mut group = c.benchmark_group("add to Populated Tree");
    group.throughput(Throughput::Elements(QTY_TO_ADD_TO_POPULATED));

    let plot_config = PlotConfiguration::default()
        .summary_scale(AxisScale::Logarithmic);

    group.plot_config(plot_config);

    batch_benches!(group, bench_populated_fixed,
        [(FXP, 3)],
        [(100, u16, u16), (1_000, u16, u16), (10_000, u16, u16), (100_000, u32, u16), (1_000_000, u32, u32)]
    );

    batch_benches!(group, bench_populated_float,
        [(f64, 2), (f64, 3), (f64, 4), (f32, 3)],
        [(100, u16, u16), (1_000, u16, u16), (10_000, u16, u16), (100_000, u32, u16), (1_000_000, u32, u32)]
    );

    group.finish();
}

fn bench_add_to_empty_float<A: Axis, T: Content, const K: usize, IDX: Index<T = IDX>>(group: &mut BenchmarkGroup<WallTime>, qty_to_add: usize, subtype: &str) where usize: Cast<IDX>, Standard: Distribution<([A; K], T)> {
    group.bench_with_input(BenchmarkId::new(subtype, qty_to_add), &qty_to_add, |b, &size| {
        b.iter_batched(|| {
                let points_to_add: Vec<([A; K], T)> =
                    (0..size).into_iter().map(|_| rand::random::<([A; K], T)>()).collect();

                let kdtree =
                    KdTree::<A, T, K, BUCKET_SIZE, IDX>::with_capacity(points_to_add.len());

                (kdtree, points_to_add)
            }, |(mut kdtree, points_to_add)| {
            black_box(points_to_add
                .iter()
                .for_each(|point| kdtree.add(&point.0, point.1)))
        }, BatchSize::SmallInput);
    });
}

fn bench_add_to_populated_float<A: Axis, T: Content, const K: usize, IDX: Index<T = IDX>>(group: &mut BenchmarkGroup<WallTime>, initial_size: usize, subtype: &str) where usize: Cast<IDX>, Standard: Distribution<([A; K], T)> {
    group.bench_with_input(BenchmarkId::new(subtype, initial_size), &initial_size, |b, &size| {
        b.iter_batched(|| {
            let points_to_add: Vec<([A; K], T)> =
                (0..QTY_TO_ADD_TO_POPULATED).into_iter().map(|_| rand::random::<([A; K], T)>()).collect();

            let mut initial_points = vec![];
            for _ in 0..size {
                initial_points.push(rand::random::<([A; K], T)>());
            }
            let mut kdtree =
                KdTree::<A, T, K, BUCKET_SIZE, IDX>::with_capacity(size + points_to_add.len());

            for i in 0..initial_points.len() {
                kdtree.add(&initial_points[i].0, initial_points[i].1);
            }

            (kdtree, points_to_add)
        }, |(mut kdtree, points_to_add)| {
            black_box(points_to_add
                .iter()
                .for_each(|point| kdtree.add(&point.0, point.1)))
        }, BatchSize::SmallInput);
    });
}

fn bench_add_to_empty_fixed<T: ContentFixed, IDX: IndexFixed<T = IDX>>(group: &mut BenchmarkGroup<WallTime>, qty_to_add: usize, subtype: &str) where usize: Cast<IDX>, Standard: Distribution<T>  {
    group.bench_with_input(BenchmarkId::new(subtype, qty_to_add), &qty_to_add, |b, &size| {
        b.iter_batched(|| {
            let mut points_to_add = vec![];
            for _ in 0..size {
                points_to_add.push(rand_data_fixed_u16_3d_entry::<T>());
            }
            let kdtree =
                FixedKdTree::<FXP, T, 3, BUCKET_SIZE, IDX>::with_capacity(size + points_to_add.len());

            (kdtree, points_to_add)
        }, |(mut kdtree, points_to_add)| {
            points_to_add
                .iter()
                .for_each(|point| black_box(kdtree.add(black_box(&point.0), point.1)))
        }, BatchSize::SmallInput);
    });
}

fn bench_add_to_populated_fixed<T: ContentFixed, IDX: IndexFixed<T = IDX>>(group: &mut BenchmarkGroup<WallTime>, initial_size: usize, subtype: &str) where usize: Cast<IDX>, Standard: Distribution<T>  {
    group.bench_with_input(BenchmarkId::new(subtype, initial_size), &initial_size, |b, &size| {
        b.iter_batched(|| {
            let mut points_to_add = vec![];
            for _ in 0..QTY_TO_ADD_TO_POPULATED {
                points_to_add.push(rand_data_fixed_u16_3d_entry::<T>());
            }

            let mut points = vec![];
            let mut kdtree =
                FixedKdTree::<FXP, T, 3, BUCKET_SIZE, IDX>::with_capacity(size + points_to_add.len());
            for _ in 0..size {
                points.push(rand_data_fixed_u16_3d_entry::<T>());
            }
            for i in 0..points.len() {
                kdtree.add(&points[i].0, points[i].1);
            }

            (kdtree, points_to_add)
        }, |(mut kdtree, points_to_add)| {
            points_to_add
                .iter()
                .for_each(|point| black_box(kdtree.add(black_box(&point.0), point.1)))
        }, BatchSize::SmallInput);
    });
}

criterion_group!(benches, add_to_empty, add_to_populated);
criterion_main!(benches);
