use az::Cast;
use criterion::{
    criterion_group, criterion_main, measurement::WallTime, AxisScale, BatchSize, BenchmarkGroup,
    BenchmarkId, Criterion, PlotConfiguration, Throughput,
};
use kiddo::distance::float::SquaredEuclidean;
use kiddo::immutable::float::kdtree::ImmutableKdTree;
use kiddo::leaf_slice::float::{LeafSliceFloat, LeafSliceFloatChunk};
// use kiddo::stem_strategies::donnelly_4::DonnellyFullArith;
use kiddo::stem_strategies::Donnelly;
use kiddo::test_utils::{
    build_populated_tree_and_query_points_immutable_float, process_queries_immutable_float,
};
use kiddo::traits::{Axis, Content};
use kiddo::{batch_benches, Eytzinger};
use rand::distr::StandardUniform;
use rand_distr::Distribution;

const BUCKET_SIZE: usize = 32;
const QUERY_POINTS_PER_LOOP: usize = 1000;
const L: u32 = 4;

macro_rules! bench_float_all {
    ($group:ident, $a:ty, $t:ty, $k:tt, $idx:tt, $size:tt, $subtype:expr) => {{
        // bench_query_nearest_one::<$a, $t, Eytzinger<$k>, $k>(
        //     &mut $group,
        //     $size,
        //     QUERY_POINTS_PER_LOOP,
        //     &format!("Eytzinger/{}", $subtype),
        // );

        bench_query_nearest_one::<$a, $t, Donnelly<L, 64, 4, $k>, $k>(
            &mut $group,
            $size,
            QUERY_POINTS_PER_LOOP,
            &format!("DonnellyOrig/{}", $subtype),
        );

        // bench_query_nearest_one::<$a, $t, DonnellyFullArith<L, 64, 4>, $k>(
        //     &mut $group,
        //     $size,
        //     QUERY_POINTS_PER_LOOP,
        //     &format!("DonnellyFullArith/{}", $subtype),
        // );
    }};
}

pub fn nearest_one_all_stems(c: &mut Criterion) {
    let mut group = c.benchmark_group("Query Nearest 1");
    group.throughput(Throughput::Elements(QUERY_POINTS_PER_LOOP as u64));

    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    group.plot_config(plot_config);

    // Run the benches for all three stem strategies
    batch_benches!(
        group,
        bench_float_all,
        [(f32, 2), (f32, 3), (f32, 4)],
        [
            (1_000, u16, usize),
            (10_000, u16, usize),
            (100_000, u32, usize),
            (1_000_000, u32, usize),
            (10_000_000, u32, usize)
        ]
    );

    group.finish();
}

fn bench_query_nearest_one<A, T, Stem, const K: usize>(
    group: &mut BenchmarkGroup<WallTime>,
    initial_size: usize,
    query_point_qty: usize,
    label: &str,
) where
    A: Axis + LeafSliceFloat<T> + LeafSliceFloatChunk<T, K> + 'static,
    T: Content + 'static,
    usize: Cast<T>,
    Stem: kiddo::StemStrategy + 'static,
    StandardUniform: Distribution<T>,
    StandardUniform: Distribution<[A; K]>,
{
    group.bench_with_input(
        BenchmarkId::new(label, initial_size),
        &initial_size,
        |b, &size| {
            b.iter_batched(
                || {
                    build_populated_tree_and_query_points_immutable_float::<
                        A,
                        T,
                        Stem,
                        K,
                        BUCKET_SIZE,
                    >(size, query_point_qty)
                },
                process_queries_immutable_float(
                    |tree: &ImmutableKdTree<A, T, Stem, K, BUCKET_SIZE>, point: &[A; K]| {
                        tree.approx_nearest_one::<SquaredEuclidean>(point);
                    },
                ),
                BatchSize::SmallInput,
            );
        },
    );
}

criterion_group!(benches, nearest_one_all_stems);
criterion_main!(benches);
