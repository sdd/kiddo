use az::Cast;
use codspeed_criterion_compat::{
    black_box, criterion_group, criterion_main, measurement::WallTime, AxisScale, BatchSize,
    BenchmarkGroup, BenchmarkId, Criterion, PlotConfiguration, Throughput,
};

use fixed::types::extra::{Unsigned, U16};
use fixed::FixedU16;
use rand::distributions::{Distribution, Standard};

use kiddo::batch_benches;
use kiddo::fixed::kdtree::{Axis as AxisFixed, KdTree as FixedKdTree};
use kiddo::float::kdtree::{Axis, KdTree};
use kiddo::test_utils::rand_data_fixed_u16_entry;
use kiddo::traits::{Content, Index};

const BUCKET_SIZE: usize = 32;
const QTY_TO_ADD_TO_POPULATED: u64 = 100;

type Fxd = U16; // FixedU16<U16>;

macro_rules! bench_empty_float {
    ($group:ident, $a:ty, $t:ty, $k:tt, $idx: ty, $size:tt, $subtype: expr) => {
        bench_add_to_empty_float::<$a, $t, $k, $idx>(&mut $group, $size, $subtype);
    };
}

macro_rules! bench_empty_fixed {
    ($group:ident, $a:ty, $t:ty, $k:tt, $idx:ty, $size:tt, $subtype: expr) => {
        bench_add_to_empty_fixed_u16::<$a, $t, $k, $idx>(&mut $group, $size, $subtype);
    };
}

macro_rules! bench_populated_float {
    ($group:ident, $a:ty, $t:ty, $k:tt, $idx: ty, $size:tt, $subtype: expr) => {
        bench_add_to_populated_float::<$a, $t, $k, $idx>(&mut $group, $size, $subtype);
    };
}

macro_rules! bench_populated_fixed {
    ($group:ident, $a:ty, $t:ty, $k:tt, $idx:ty, $size:tt, $subtype: expr) => {
        bench_add_to_populated_fixed_u16::<$a, $t, $k, $idx>(&mut $group, $size, $subtype);
    };
}

pub fn add_to_empty(c: &mut Criterion) {
    let mut group = c.benchmark_group("Add to Empty Tree");

    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    group.plot_config(plot_config);

    batch_benches!(
        group,
        bench_empty_fixed,
        [(Fxd, 3)],
        [
            (100, u16, u16),
            (1_000, u16, u16),
            (10_000, u16, u16),
            (100_000, u32, u16),
            (1_000_000, u32, u32)
        ]
    );

    batch_benches!(
        group,
        bench_empty_float,
        [(f64, 2), (f64, 3), (f64, 4), (f32, 3)],
        [
            (100, u16, u16),
            (1_000, u16, u16),
            (10_000, u16, u16),
            (100_000, u32, u16),
            (1_000_000, u32, u32)
        ]
    );

    group.finish();
}

pub fn add_to_populated(c: &mut Criterion) {
    let mut group = c.benchmark_group("add to Populated Tree");
    group.throughput(Throughput::Elements(QTY_TO_ADD_TO_POPULATED));

    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    group.plot_config(plot_config);

    batch_benches!(
        group,
        bench_populated_fixed,
        [(Fxd, 3)],
        [
            (100, u16, u16),
            (1_000, u16, u16),
            (10_000, u16, u16),
            (100_000, u32, u16),
            (1_000_000, u32, u32)
        ]
    );

    batch_benches!(
        group,
        bench_populated_float,
        [(f64, 2), (f64, 3), (f64, 4), (f32, 3)],
        [
            (100, u16, u16),
            (1_000, u16, u16),
            (10_000, u16, u16),
            (100_000, u32, u16),
            (1_000_000, u32, u32)
        ]
    );

    group.finish();
}

fn bench_add_to_empty_float<A: Axis, T: Content, const K: usize, IDX: Index<T = IDX>>(
    group: &mut BenchmarkGroup<WallTime>,
    qty_to_add: usize,
    subtype: &str,
) where
    usize: Cast<IDX>,
    Standard: Distribution<([A; K], T)>,
{
    group.bench_with_input(
        BenchmarkId::new(subtype, qty_to_add),
        &qty_to_add,
        |b, &size| {
            b.iter_batched(
                || {
                    let points_to_add: Vec<([A; K], T)> =
                        (0..size).map(|_| rand::random::<([A; K], T)>()).collect();

                    let kdtree =
                        KdTree::<A, T, K, BUCKET_SIZE, IDX>::with_capacity(points_to_add.len());

                    (kdtree, points_to_add)
                },
                |(mut kdtree, points_to_add)| {
                    points_to_add
                        .iter()
                        .for_each(|point| kdtree.add(&point.0, point.1));
                    black_box(())
                },
                BatchSize::SmallInput,
            );
        },
    );
}

fn bench_add_to_populated_float<A: Axis, T: Content, const K: usize, IDX: Index<T = IDX>>(
    group: &mut BenchmarkGroup<WallTime>,
    initial_size: usize,
    subtype: &str,
) where
    usize: Cast<IDX>,
    Standard: Distribution<([A; K], T)>,
{
    group.bench_with_input(
        BenchmarkId::new(subtype, initial_size),
        &initial_size,
        |b, &size| {
            b.iter_batched(
                || {
                    let points_to_add: Vec<([A; K], T)> = (0..QTY_TO_ADD_TO_POPULATED)
                        .map(|_| rand::random::<([A; K], T)>())
                        .collect();

                    let mut initial_points = vec![];
                    for _ in 0..size {
                        initial_points.push(rand::random::<([A; K], T)>());
                    }
                    let mut kdtree = KdTree::<A, T, K, BUCKET_SIZE, IDX>::with_capacity(
                        size + points_to_add.len(),
                    );

                    for point in &initial_points {
                        kdtree.add(&point.0, point.1);
                    }

                    (kdtree, points_to_add)
                },
                |(mut kdtree, points_to_add)| {
                    points_to_add
                        .iter()
                        .for_each(|point| kdtree.add(&point.0, point.1));
                    black_box(())
                },
                BatchSize::SmallInput,
            );
        },
    );
}

fn bench_add_to_empty_fixed_u16<A: Unsigned, T: Content, const K: usize, IDX: Index<T = IDX>>(
    group: &mut BenchmarkGroup<WallTime>,
    qty_to_add: usize,
    subtype: &str,
) where
    usize: Cast<IDX>,
    Standard: Distribution<T>,
    FixedU16<A>: AxisFixed,
{
    group.bench_with_input(
        BenchmarkId::new(subtype, qty_to_add),
        &qty_to_add,
        |b, &size| {
            b.iter_batched(
                || {
                    let mut points_to_add = vec![];
                    for _ in 0..size {
                        points_to_add.push(rand_data_fixed_u16_entry::<A, T, K>());
                    }
                    let kdtree = FixedKdTree::<FixedU16<A>, T, K, BUCKET_SIZE, IDX>::with_capacity(
                        size + points_to_add.len(),
                    );

                    (kdtree, points_to_add)
                },
                |(mut kdtree, points_to_add)| {
                    points_to_add.iter().for_each(|point| {
                        kdtree.add(black_box(&point.0), point.1);
                        black_box(())
                    })
                },
                BatchSize::SmallInput,
            );
        },
    );
}

fn bench_add_to_populated_fixed_u16<A: Unsigned, T: Content, const K: usize, IDX: Index<T = IDX>>(
    group: &mut BenchmarkGroup<WallTime>,
    initial_size: usize,
    subtype: &str,
) where
    usize: Cast<IDX>,
    Standard: Distribution<T>,
    FixedU16<A>: AxisFixed,
{
    group.bench_with_input(
        BenchmarkId::new(subtype, initial_size),
        &initial_size,
        |b, &size| {
            b.iter_batched(
                || {
                    let mut points = vec![];
                    for _ in 0..QTY_TO_ADD_TO_POPULATED {
                        points.push(rand_data_fixed_u16_entry::<A, T, K>());
                    }

                    let mut initial_points = vec![];
                    for _ in 0..size {
                        initial_points.push(rand_data_fixed_u16_entry::<A, T, K>());
                    }
                    let mut kdtree =
                        FixedKdTree::<FixedU16<A>, T, K, BUCKET_SIZE, IDX>::with_capacity(
                            size + points.len(),
                        );
                    for point in &initial_points {
                        kdtree.add(&point.0, point.1);
                    }

                    (kdtree, points)
                },
                |(mut kdtree, points_to_add)| {
                    points_to_add.iter().for_each(|point| {
                        kdtree.add(black_box(&point.0), point.1);
                        black_box(())
                    })
                },
                BatchSize::SmallInput,
            );
        },
    );
}

criterion_group!(benches, add_to_empty, add_to_populated);
criterion_main!(benches);
