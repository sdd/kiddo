use az::Cast;
use criterion::measurement::WallTime;
use criterion::{
    criterion_group, criterion_main, AxisScale, BatchSize, BenchmarkGroup, BenchmarkId, Criterion,
    PlotConfiguration, Throughput,
};
use kiddo::batch_benches;
use kiddo::float::distance::SquaredEuclidean;
use kiddo::float_leaf_simd::leaf_node::BestFromDists;
use kiddo::immutable::float::kdtree::{Axis, ImmutableKdTree};
use kiddo::test_utils::{
    build_populated_tree_and_query_points_immutable_float, process_queries_immutable_float,
};
use kiddo::types::Content;
use rand::distributions::Standard;
use rand_distr::Distribution;

const BUCKET_SIZE: usize = 32;
const QUERY_POINTS_PER_LOOP: usize = 1000;

macro_rules! bench_float {
    ($group:ident, $a:ty, $t:ty, $k:tt, $idx:tt, $size:tt, $subtype: expr) => {
        bench_query_nearest_one_immutable_float::<$a, $t, $k>(
            &mut $group,
            $size,
            QUERY_POINTS_PER_LOOP,
            $subtype,
        );
    };
}

pub fn nearest_one_immutable_float(c: &mut Criterion) {
    let mut group = c.benchmark_group("Query Nearest 1");
    group.throughput(Throughput::Elements(QUERY_POINTS_PER_LOOP as u64));

    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    group.plot_config(plot_config);

    batch_benches!(
        group,
        bench_float,
        [(f64, 2), (f64, 3), (f64, 4)],
        [
            (1_000, u16, usize),
            (10_000, u16, usize),
            (100_000, u32, usize),
            (1_000_000, u32, usize)
        ]
    );

    group.finish();
}

fn perform_query_immutable_float<A: Axis, T: Content + 'static, const K: usize, const B: usize>(
    kdtree: &ImmutableKdTree<A, T, K, BUCKET_SIZE>,
    point: &[A; K],
) where
    A: BestFromDists<T, 32>,
    usize: Cast<T>,
{
    kdtree.nearest_one::<SquaredEuclidean>(&point);
}

fn bench_query_nearest_one_immutable_float<
    'a,
    A: Axis + 'static,
    T: Content + 'static,
    const K: usize,
>(
    group: &'a mut BenchmarkGroup<WallTime>,
    initial_size: usize,
    query_point_qty: usize,
    subtype: &str,
) where
    A: BestFromDists<T, 32>,
    usize: Cast<T>,
    Standard: Distribution<T>,
    Standard: Distribution<[A; K]>,
{
    group.bench_with_input(
        BenchmarkId::new(subtype, initial_size),
        &initial_size,
        |b, &size| {
            b.iter_batched(
                || {
                    build_populated_tree_and_query_points_immutable_float::<A, T, K, BUCKET_SIZE>(
                        size,
                        query_point_qty,
                    )
                },
                process_queries_immutable_float(
                    perform_query_immutable_float::<A, T, K, BUCKET_SIZE>,
                ),
                BatchSize::SmallInput,
            );
        },
    );
}

criterion_group!(benches, nearest_one_immutable_float);
criterion_main!(benches);
