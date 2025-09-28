use codspeed_criterion_compat::{
    black_box, criterion_group, criterion_main, measurement::WallTime, AxisScale, BatchSize,
    BenchmarkGroup, BenchmarkId, Criterion, PlotConfiguration, Throughput,
};
use kiddo::donnelly_stem_layout::{donnelly_get_idx_v2_branchless, LOG2_ITEMS_PER_CACHE_LINE};
use kiddo::{stem_strategies::donnelly_4::DonnellyFullArith, StemStrategy};

use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

const TRAVERSALS_PER_BENCHMARK: usize = 1_000;
const RNG_SEED: u64 = 42;

pub fn donnelly_traversal_branchless(c: &mut Criterion) {
    let mut group = c.benchmark_group("Donnelly Traversal");

    for depth in [5, 10, 15, 20, 25] {
        let total_ops = depth * TRAVERSALS_PER_BENCHMARK;
        group.throughput(Throughput::Elements(total_ops as u64));

        group.bench_with_input(BenchmarkId::new("FullArith", depth), &depth, |b, &depth| {
            b.iter_batched_ref(
                || generate_calls(depth, TRAVERSALS_PER_BENCHMARK),
                |calls| {
                    for (selections, mut strat) in calls {
                        for &sel in selections.iter() {
                            black_box(strat.get_child_idx(sel, 0));
                        }
                    }
                },
                BatchSize::SmallInput,
            )
        });

        group.bench_with_input(
            BenchmarkId::new("Branchless", depth),
            &depth,
            |b, &depth| {
                b.iter_batched(
                    || generate_calls(depth, TRAVERSALS_PER_BENCHMARK),
                    |calls| {
                        for (selections, _) in calls {
                            let mut idx = 0;
                            let mut minor = 0;
                            for sel in selections {
                                idx = black_box(donnelly_get_idx_v2_branchless(
                                    idx as u32, sel, minor,
                                )) as usize;
                                minor = (minor + 1) % LOG2_ITEMS_PER_CACHE_LINE;
                            }
                        }
                    },
                    BatchSize::SmallInput,
                )
            },
        );
    }

    group.finish();
}

fn generate_calls(
    depth: usize,
    traversal_count: usize,
) -> Vec<(Vec<bool>, DonnellyFullArith<3, 64, 4>)> {
    let mut rng = StdRng::seed_from_u64(RNG_SEED);
    (0..traversal_count)
        .map(|_| {
            let selections = (0..depth).map(|_| rng.gen_bool(0.5)).collect();
            (selections, DonnellyFullArith::new_query())
        })
        .collect()
}

criterion_group!(benches, donnelly_traversal_branchless);
criterion_main!(benches);
