use codspeed_criterion_compat::{black_box, criterion_group, criterion_main, measurement::WallTime, AxisScale, BatchSize, BenchmarkGroup, BenchmarkId, Criterion, PlotConfiguration, Throughput};
use kiddo::donnelly_stem_layout::donnelly_get_idx_v2;
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

const TRAVERSALS_PER_BENCHMARK: usize = 1_000;
const RNG_SEED: u64 = 42;

pub fn donnelly_traversal(c: &mut Criterion) {
    let mut group = c.benchmark_group("Donnelly Traversal");
    group.throughput(Throughput::Elements(TRAVERSALS_PER_BENCHMARK as u64));

    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Linear);
    group.plot_config(plot_config);

    // Test different tree depths to simulate realistic usage
    for depth in [5, 10, 15, 20, 25] {
        group.bench_with_input(
            BenchmarkId::new("D3", depth),
            &depth,
            |b, &depth| {
                b.iter_batched(
                    || generate_flattened_calls(depth, TRAVERSALS_PER_BENCHMARK),
                    |calls| {
                        for (curr_idx, is_right_child, level) in calls {
                            black_box(donnelly_get_idx_v2(curr_idx, is_right_child, level));
                        }
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

/// Generate pre-computed function call parameters to minimize overhead in the hot loop
fn generate_flattened_calls(depth: usize, traversal_count: usize) -> Vec<(u32, bool, u32)> {
    let mut rng = StdRng::seed_from_u64(RNG_SEED);
    let mut calls = Vec::with_capacity(depth * traversal_count);

    for _ in 0..traversal_count {
        let mut current_idx = 0u32;
        let mut level = 0u32;

        for _ in 0..depth {
            let is_right_child = rng.gen_bool(0.5);
            calls.push((current_idx, is_right_child, level));
            // Pre-compute next state for realistic progression
            current_idx = donnelly_get_idx_v2(current_idx, is_right_child, level);
            level += 1;
        }
    }

    calls
}

criterion_group!(benches, donnelly_traversal);
criterion_main!(benches);