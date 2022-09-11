use criterion::{black_box, criterion_group, criterion_main, Criterion};
use sok::KdTree;
use rayon::prelude::*;

use sok::distance::squared_euclidean;

const K: usize = 3;
const BUCKET_SIZE: usize = 32;
const QUERY: usize = 1_000_000;

fn criterion_benchmark(c: &mut Criterion) {

    // Bench building tree
    for ndata in [3, 4, 5, 6, 7].map(|p| 10_usize.pow(p)) {

        let data: Vec<[f32; K]> = (0..ndata)
            .map(|_| [(); K].map(|_| rand::random()))
            .collect();
        let query: Vec<[f32; K]> = (0..QUERY)
            .map(|_| [(); K].map(|_| rand::random()))
            .collect();

        let mut group = c.benchmark_group(
            format!(
                "{:?} queries (ndata = {})", QUERY, ndata
            )
        );

        let mut kdtree =
            KdTree::<_, _, K, BUCKET_SIZE>::with_capacity(ndata);
        for idx in 0..ndata {
            kdtree.add(&data[idx], idx);
        }

        group.bench_function(
            "non-periodic",
            |b| {
                b.iter(|| {
                    let v: Vec<_> = black_box(&query)
                        .par_iter()
                        .map_with(black_box(&kdtree), |t, q| {
                            let (dist, idx) = t.nearest_one(black_box(&q), &squared_euclidean);
                            drop(dist);
                            drop(idx);
                        })
                        .collect();
                    drop(v)
                })
            },
        );
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
