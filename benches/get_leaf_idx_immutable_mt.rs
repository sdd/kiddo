use az::Cast;
use criterion::{
    criterion_group, criterion_main, measurement::WallTime, AxisScale, BatchSize, BenchmarkGroup,
    BenchmarkId, Criterion, PlotConfiguration, Throughput,
};
use kiddo::immutable::float::kdtree::ImmutableKdTree;
use kiddo::leaf_slice::float::{LeafSliceFloat, LeafSliceFloatChunk};
use kiddo::stem_strategies::Donnelly;
use kiddo::test_utils::{
    build_populated_tree_and_query_points_immutable_float, process_queries_immutable_float,
};
use kiddo::traits::{Axis, Content};
use kiddo::{batch_benches, Eytzinger};
use rand::distr::StandardUniform;
use rand_distr::Distribution;
use rayon::prelude::*;

const BUCKET_SIZE: usize = 32;
const QUERY_POINTS_PER_LOOP: usize = 1_000_000;

macro_rules! bench_float_all {
    ($group:ident, $a:ty, $t:ty, $k:tt, $idx:tt, $size:tt, $subtype:expr) => {{
        bench_query_leaf_idx::<$a, $t, Eytzinger<$k>, $k>(
            &mut $group,
            $size,
            QUERY_POINTS_PER_LOOP,
            &format!("Eytzinger/{}", $subtype),
        );

        bench_query_leaf_idx::<$a, $t, Donnelly<4, 64, 4, $k>, $k>(
            &mut $group,
            $size,
            QUERY_POINTS_PER_LOOP,
            &format!("Donnelly/{}", $subtype),
        );
    }};
}

pub fn get_leaf_idx_all_stems(c: &mut Criterion) {
    let mut group = c.benchmark_group("Query Leaf Idx");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(10)); // optional
    group.throughput(Throughput::Elements(QUERY_POINTS_PER_LOOP as u64));
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    batch_benches!(
        group,
        bench_float_all,
        [(f32, 3)],
        [
            // (4_096, u32, usize), // Full L1
            // (8_192, u32, usize),
            // (16_536, u32, usize),
            // (32_768, u32, usize),
            // (65_536, u32, usize),
            // (131_072, u32, usize),
            // (262_144, u32, usize),
            // (524_288, u32, usize),
            (1_048_576, u32, usize),
            (2_097_152, u32, usize),
            (4_194_304, u32, usize),
            (8_388_608, u32, usize),
            (16_777_216, u32, usize) // (33_554_432, u32, usize),
                                     // (67_108_864, u32, usize)
        ]
    );

    group.finish();
}

fn bench_query_leaf_idx<A, T, Stem, const K: usize>(
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
                        tree.get_leaf_node_idx(point, None);
                    },
                ),
                BatchSize::SmallInput,
            );
        },
    );
}

criterion_group!(benches, get_leaf_idx_all_stems);
criterion_main!(benches);
