use codspeed_criterion_compat::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use kiddo::kd_tree::KdTree;
use kiddo::leaf_strategy::VecOfArenas;
use kiddo::stem_strategy::Eytzinger;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

const K: usize = 3;
const B: usize = 32;
const DEFAULT_MIN_LOG2_POINTS: u32 = 16;
const DEFAULT_MAX_LOG2_POINTS: u32 = 22;
const POINT_SEED: u64 = 0x5eed_0000_0000_0301;

type Tree = KdTree<f64, u32, Eytzinger<K>, VecOfArenas<f64, u32, K, B>, K, B>;

fn read_u32_env(var: &str, default: u32) -> u32 {
    std::env::var(var)
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(default)
}

fn build_points(point_count: usize) -> Vec<[f64; K]> {
    let mut rng = ChaCha8Rng::seed_from_u64(POINT_SEED ^ point_count as u64);
    (0..point_count).map(|_| rng.random::<[f64; K]>()).collect()
}

fn v6_construction(c: &mut Criterion) {
    let min_log2_points = read_u32_env("KIDDO_BENCH_MIN_LOG2_POINTS", DEFAULT_MIN_LOG2_POINTS);
    let max_log2_points = read_u32_env("KIDDO_BENCH_MAX_LOG2_POINTS", DEFAULT_MAX_LOG2_POINTS);

    let mut group = c.benchmark_group("v6 new_from_slice construction");

    for log2_points in min_log2_points..=max_log2_points {
        let point_count = 1usize << log2_points;
        let points = build_points(point_count);
        group.throughput(Throughput::Elements(point_count as u64));

        group.bench_function(
            BenchmarkId::new("Eytzinger/VecOfArenas", point_count),
            |b| {
                b.iter(|| black_box(Tree::new_from_slice(black_box(&points)).unwrap()));
            },
        );
    }

    group.finish();
}

criterion_group!(benches, v6_construction);
criterion_main!(benches);
