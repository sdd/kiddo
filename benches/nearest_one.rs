use az::Cast;
use codspeed_criterion_compat::{
    criterion_group, criterion_main, measurement::WallTime, AxisScale, BatchSize, BenchmarkGroup,
    BenchmarkId, Criterion, PlotConfiguration, Throughput,
};
use fixed::types::extra::{Unsigned, U8};
use fixed::{FixedU16, FixedU32};
use kiddo::batch_benches;
use kiddo::distance::fixed::SquaredEuclidean as SquaredEuclideanFixed;
use kiddo::distance::float::SquaredEuclidean;
use kiddo::mutable::fixed::kdtree::KdTree as KdTreeFixed;
use kiddo::mutable::float::kdtree::KdTree;
use kiddo::test_utils::{
    build_populated_tree_and_query_points_fixed, build_populated_tree_and_query_points_float,
    process_queries_fixed, process_queries_float,
};
use kiddo::traits::{Axis, AxisFixed, Content, Index};
use rand::distr::StandardUniform;
use rand_distr::Distribution;

const BUCKET_SIZE: usize = 32;
const QUERY_POINTS_PER_LOOP: usize = 1000;

type Fxd = U8; // FixedU16<U8>;
type FxdR = FixedU32<U8>;

macro_rules! bench_float {
    ($group:ident, $a:ty, $t:ty, $k:tt, $idx: ty, $size:tt, $subtype: expr) => {
        bench_query_nearest_one_float::<$a, $t, $k, $idx>(
            &mut $group,
            $size,
            QUERY_POINTS_PER_LOOP,
            $subtype,
        );
    };
}

macro_rules! bench_fixed {
    ($group:ident, $a:ty, $t:ty, $k:tt, $idx:ty, $size:tt, $subtype: expr) => {
        bench_query_nearest_one_fixed::<$a, $t, $k, $idx>(
            &mut $group,
            $size,
            QUERY_POINTS_PER_LOOP,
            $subtype,
        );
    };
}

pub fn nearest_one(c: &mut Criterion) {
    let mut group = c.benchmark_group("Query Nearest 1");
    group.throughput(Throughput::Elements(QUERY_POINTS_PER_LOOP as u64));

    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    group.plot_config(plot_config);

    batch_benches!(
        group,
        bench_float,
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
        bench_fixed,
        [(Fxd, 3)],
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

fn perform_query_float<
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
{
    kdtree.nearest_one::<SquaredEuclidean>(point);
}

fn perform_query_fixed<
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
{
    kdtree.nearest_one::<SquaredEuclideanFixed, FxdR>(point);
}

fn bench_query_nearest_one_float<
    A: Axis + 'static,
    T: Content + 'static,
    const K: usize,
    IDX: Index<T = IDX> + 'static,
>(
    group: &mut BenchmarkGroup<WallTime>,
    initial_size: usize,
    query_point_qty: usize,
    subtype: &str,
) where
    usize: Cast<IDX>,
    StandardUniform: Distribution<T>,
    StandardUniform: Distribution<[A; K]>,
{
    group.bench_with_input(
        BenchmarkId::new(subtype, initial_size),
        &initial_size,
        |b, &size| {
            b.iter_batched(
                || {
                    build_populated_tree_and_query_points_float::<A, T, K, BUCKET_SIZE, IDX>(
                        size,
                        query_point_qty,
                    )
                },
                process_queries_float(perform_query_float::<A, T, K, BUCKET_SIZE, IDX>),
                BatchSize::SmallInput,
            );
        },
    );
}

fn bench_query_nearest_one_fixed<
    A: Unsigned,
    T: Content + 'static,
    const K: usize,
    IDX: Index<T = IDX> + 'static,
>(
    group: &mut BenchmarkGroup<WallTime>,
    initial_size: usize,
    query_point_qty: usize,
    subtype: &str,
) where
    usize: Cast<IDX>,
    StandardUniform: Distribution<T>,
    FixedU16<A>: AxisFixed,
{
    group.bench_with_input(
        BenchmarkId::new(subtype, initial_size),
        &initial_size,
        |b, &size| {
            b.iter_batched(
                || {
                    build_populated_tree_and_query_points_fixed::<A, T, K, BUCKET_SIZE, IDX>(
                        size,
                        query_point_qty,
                    )
                },
                process_queries_fixed(perform_query_fixed::<A, T, K, BUCKET_SIZE, IDX>),
                BatchSize::SmallInput,
            );
        },
    );
}

criterion_group!(benches, nearest_one);
criterion_main!(benches);
