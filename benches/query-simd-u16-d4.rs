use criterion::{black_box, criterion_group, criterion_main, Criterion};
use fixed::types::extra::U14;
use fixed::FixedU16;
use sok::fixed::distance::squared_euclidean;
use sok::fixed::kdtree::KdTree;

use rayon::prelude::*;

type FXPT = FixedU16<U14>;

const K: usize = 4;
const BUCKET_SIZE: usize = 32;
const QUERY: usize = 1_000_000;

fn criterion_benchmark(c: &mut Criterion) {
    // Bench building tree
    for ndata in [3, 4, 5, 6, 7].map(|p| 10_usize.pow(p)) {
        let data: Vec<[FXPT; K]> = (0..ndata)
            .map(|_| [(); K].map(|_| {
                let val: u16 = rand::random();
                unsafe { std::mem::transmute(val) }
            }))
            .collect();
        let query: Vec<[FXPT; K]> = (0..QUERY)
            .map(|_| [(); K].map(|_| {
                let val: u16 = rand::random();
                unsafe { std::mem::transmute(val) }
            }))
            .collect();

        let mut group = c.benchmark_group(format!(
            "{:?} queries SIMD u16 4D (ndata = {})",
            QUERY, ndata
        ));

        let mut kdtree: KdTree<FXPT, u32, K, BUCKET_SIZE, u32> = KdTree::with_capacity(BUCKET_SIZE);
        for idx in 0..ndata {
            kdtree.add(&data[idx], idx as u32);
        }

        group.bench_function("non-periodic", |b| {
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
        });
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
