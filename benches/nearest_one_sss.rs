use az::Cast;
use criterion::measurement::WallTime;
use criterion::{
    criterion_group, criterion_main, AxisScale, BatchSize, BenchmarkGroup, BenchmarkId, Criterion,
    PlotConfiguration, Throughput,
};
use kiddo::batch_benches;
use kiddo::distance::squared_euclidean;
use kiddo::float_sss::kdtree::{Axis, KdTree};
use kiddo::test_utils::{
    build_populated_tree_and_query_points_float_sss, process_queries_float_sss,
};
use kiddo::types::{Content, Index};
use rand::distributions::Standard;
use rand_distr::Distribution;

const BUCKET_SIZE: usize = 32;
const QUERY_POINTS_PER_LOOP: usize = 1000;

macro_rules! bench_float {
    ($group:ident, $a:ty, $t:ty, $k:tt, $idx: ty, $size:tt, $subtype: expr) => {
        bench_query_nearest_one_float_sss::<$a, $t, $k, $idx>(
            &mut $group,
            $size,
            QUERY_POINTS_PER_LOOP,
            $subtype,
        );
    };
}

pub fn nearest_one_sss(c: &mut Criterion) {
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

    group.finish();
}

fn perform_query_float_sss<
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
    kdtree.nearest_one(&point, &squared_euclidean);
}

fn bench_query_nearest_one_float_sss<
    'a,
    A: Axis + 'static,
    T: Content + 'static,
    const K: usize,
    IDX: Index<T = IDX> + 'static,
>(
    group: &'a mut BenchmarkGroup<WallTime>,
    initial_size: usize,
    query_point_qty: usize,
    subtype: &str,
) where
    usize: Cast<IDX>,
    Standard: Distribution<T>,
    Standard: Distribution<[A; K]>,
{
    group.bench_with_input(
        BenchmarkId::new(subtype, initial_size),
        &initial_size,
        |b, &size| {
            b.iter_batched(
                || {
                    build_populated_tree_and_query_points_float_sss::<A, T, K, BUCKET_SIZE, IDX>(
                        size,
                        query_point_qty,
                    )
                },
                process_queries_float_sss(perform_query_float_sss::<A, T, K, BUCKET_SIZE, IDX>),
                BatchSize::SmallInput,
            );
        },
    );
}

criterion_group!(benches, nearest_one_sss);
criterion_main!(benches);
