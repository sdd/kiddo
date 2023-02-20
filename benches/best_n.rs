use az::{Az, Cast};
use criterion::measurement::WallTime;
use criterion::{
    black_box, criterion_group, criterion_main, AxisScale, BatchSize, BenchmarkGroup, BenchmarkId,
    Criterion, PlotConfiguration, Throughput,
};
use fixed::types::extra::{LeEqU16, Unsigned, U16};
use fixed::FixedU16;
use kiddo::batch_benches;
use kiddo::distance::squared_euclidean;
use kiddo::fixed::distance::squared_euclidean as squared_euclidean_fixedpoint;
use kiddo::fixed::kdtree::{Axis as AxisFixed, KdTree as KdTreeFixed};
use kiddo::float::kdtree::{Axis, KdTree};
use kiddo::test_utils::{
    build_populated_tree_and_query_points_fixed, build_populated_tree_and_query_points_float,
    process_queries_fixed, process_queries_float,
};
use kiddo::types::{Content, Index};
use rand::distributions::Standard;
use rand_distr::Distribution;

const BUCKET_SIZE: usize = 32;
const QUERY_POINTS_PER_LOOP: usize = 100;

type FXP = U16; // FixedU16<U16>;

macro_rules! bench_float_10 {
    ($group:ident, $a:ty, $t:ty, $k:tt, $idx: ty, $size:tt, $subtype: expr) => {
        bench_query_float_10::<$a, $t, $k, $idx>(&mut $group, $size, $subtype);
    };
}

macro_rules! bench_fixed_10 {
    ($group:ident, $a:ty, $t:ty, $k:tt, $idx:ty, $size:tt, $subtype: expr) => {
        bench_query_fixed_10::<$a, $t, $k, $idx>(&mut $group, $size, $subtype);
    };
}

pub fn best_10(c: &mut Criterion) {
    let mut group = c.benchmark_group("Query: Best 10");
    group.throughput(Throughput::Elements(QUERY_POINTS_PER_LOOP as u64));

    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    group.plot_config(plot_config);

    batch_benches!(
        group,
        bench_float_10,
        [(f64, 2), (f64, 3), (f64, 4), (f32, 3)],
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
        bench_fixed_10,
        [(FXP, 3)],
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

fn perform_query_float_10<
    A: Axis,
    T: Content + 'static,
    const K: usize,
    const B: usize,
    IDX: Index<T = IDX> + 'static,
>(
    kdtree: &KdTree<A, T, K, BUCKET_SIZE, IDX>,
    point: &[A; K],
) where
    usize: Cast<IDX>,
    f64: Cast<A>,
{
    kdtree
        .best_n_within(&point, 0.05f64.az::<A>(), 10, &squared_euclidean)
        .for_each(|res_item| {
            black_box({
                let _x = res_item;
            });
        })
}

fn perform_query_fixed_10<
    A: Unsigned,
    T: Content + 'static,
    const K: usize,
    const B: usize,
    IDX: Index<T = IDX> + 'static,
>(
    kdtree: &KdTreeFixed<FixedU16<A>, T, K, BUCKET_SIZE, IDX>,
    point: &[FixedU16<A>; K],
) where
    usize: Cast<IDX>,
    FixedU16<A>: AxisFixed,
    A: LeEqU16,
{
    kdtree
        .best_n_within(
            &point,
            FixedU16::<A>::from_num(0.05f64),
            10,
            &squared_euclidean_fixedpoint,
        )
        .for_each(|res_item| {
            black_box({
                let _x = res_item;
            });
        })
}

fn bench_query_float_10<
    'a,
    A: Axis + 'static,
    T: Content + 'static,
    const K: usize,
    IDX: Index<T = IDX> + 'static,
>(
    group: &'a mut BenchmarkGroup<WallTime>,
    initial_size: usize,
    subtype: &str,
) where
    usize: Cast<IDX>,
    f64: Cast<A>,
    Standard: Distribution<T>,
    Standard: Distribution<[A; K]>,
{
    group.bench_with_input(
        BenchmarkId::new(subtype, initial_size),
        &initial_size,
        |b, &size| {
            b.iter_batched(
                || {
                    build_populated_tree_and_query_points_float::<A, T, K, BUCKET_SIZE, IDX>(
                        size,
                        QUERY_POINTS_PER_LOOP,
                    )
                },
                process_queries_float(perform_query_float_10::<A, T, K, BUCKET_SIZE, IDX>),
                BatchSize::SmallInput,
            );
        },
    );
}

fn bench_query_fixed_10<
    'a,
    A: Unsigned,
    T: Content + 'static,
    const K: usize,
    IDX: Index<T = IDX> + 'static,
>(
    group: &'a mut BenchmarkGroup<WallTime>,
    initial_size: usize,
    subtype: &str,
) where
    usize: Cast<IDX>,
    Standard: Distribution<T>,
    FixedU16<A>: AxisFixed,
    A: LeEqU16,
{
    group.bench_with_input(
        BenchmarkId::new(subtype, initial_size),
        &initial_size,
        |b, &size| {
            b.iter_batched(
                || {
                    build_populated_tree_and_query_points_fixed::<A, T, K, BUCKET_SIZE, IDX>(
                        size,
                        QUERY_POINTS_PER_LOOP,
                    )
                },
                process_queries_fixed(perform_query_fixed_10::<A, T, K, BUCKET_SIZE, IDX>),
                BatchSize::SmallInput,
            );
        },
    );
}

criterion_group!(benches, best_10);
criterion_main!(benches);
