use az::Cast;
use criterion::{AxisScale, BatchSize, BenchmarkGroup, BenchmarkId, black_box, Criterion, criterion_group, criterion_main, PlotConfiguration, Throughput};
use criterion::measurement::WallTime;
use fixed::FixedU16;
use fixed::types::extra::{U16, Unsigned};
use rand::distributions::Standard;
use rand_distr::Distribution;

use sok::batch_benches;
use sok::distance::squared_euclidean;
use sok::fixed::distance::squared_euclidean as squared_euclidean_fixedpoint;
use sok::float::kdtree::{Axis, KdTree};
use sok::fixed::kdtree::{Axis as AxisFixed, KdTree as FixedKdTree};
use sok::test_utils::{rand_data_fixed_u16_entry, rand_data_fixed_u16_point};
use sok::types::{Content, Index};

const BUCKET_SIZE: usize = 32;
const QUERY_POINTS_PER_LOOP: usize = 1000;

type FXP = U16; // FixedU16<U16>;

macro_rules! bench_float {
    ($group:ident, $a:ty, $t:ty, $k:tt, $idx: ty, $size:tt, $subtype: expr) => {
        bench_query_nearest_one_float::<$a, $t, $k, $idx>(&mut $group, $size, QUERY_POINTS_PER_LOOP, $subtype);
    }
}

macro_rules! bench_fixed {
    ($group:ident, $a:ty, $t:ty, $k:tt, $idx:ty, $size:tt, $subtype: expr) => {
        bench_query_nearest_one_fixed::<$a, $t, $k, $idx>(&mut $group, $size, QUERY_POINTS_PER_LOOP, $subtype);
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

fn bench_query_nearest_one_float<A: Axis, T: Content, const K: usize, IDX: Index<T = IDX>>(group: &mut BenchmarkGroup<WallTime>, initial_size: usize, points_qty: usize, subtype: &str) where usize: Cast<IDX>, Standard: Distribution<([A; K], T)>, Standard: Distribution<[A; K]> {
    group.bench_with_input(BenchmarkId::new(subtype, initial_size), &initial_size, |b, &size| {
        b.iter_batched(|| {
            let points: Vec<[A; K]> =
                (0..points_qty).into_iter().map(|_| rand::random::<[A; K]>()).collect();

            let mut initial_points = vec![];
            for _ in 0..size {
                initial_points.push(rand::random::<([A; K], T)>());
            }
            let mut kdtree =
                KdTree::<A, T, K, BUCKET_SIZE, IDX>::with_capacity(size);
            for i in 0..initial_points.len() {
                kdtree.add(&initial_points[i].0, initial_points[i].1);
            }

            (kdtree, points)
        }, |(kdtree, points_to_query)| {
            black_box(points_to_query
                .iter()
                .for_each(|point| black_box({
                    kdtree.nearest_one(&point, &squared_euclidean);
                })))
        }, BatchSize::SmallInput);
    });
}

fn bench_query_nearest_one_fixed<A: Unsigned, T: Content, const K: usize, IDX: Index<T = IDX>>(group: &mut BenchmarkGroup<WallTime>, initial_size: usize, points_qty: usize, subtype: &str) where usize: Cast<IDX>, Standard: Distribution<T>, FixedU16<A>: AxisFixed {
    group.bench_with_input(BenchmarkId::new(subtype, initial_size), &initial_size, |b, &size| {
        b.iter_batched(|| {
            let points: Vec<[_; K]> =
                (0..points_qty).into_iter().map(|_| rand_data_fixed_u16_point::<A, K>()).collect();

            let mut initial_points = vec![];
            for _ in 0..size {
                initial_points.push(rand_data_fixed_u16_entry::<A, T, K>());
            }
            let mut kdtree =
                FixedKdTree::<FixedU16<A>, T, K, BUCKET_SIZE, IDX>::with_capacity(size + points.len());
            for i in 0..initial_points.len() {
                kdtree.add(&initial_points[i].0, initial_points[i].1);
            }

            (kdtree, points)
        }, |(kdtree, points_to_query)| {
            black_box(points_to_query
                .iter()
                .for_each(|point| black_box({
                    kdtree.nearest_one(&point, &squared_euclidean_fixedpoint);
                })))
        }, BatchSize::SmallInput);
    });
}

criterion_group!(benches, nearest_one);
criterion_main!(benches);
