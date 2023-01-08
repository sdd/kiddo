use criterion::{black_box, criterion_group, criterion_main, Criterion};

use sok::tuned::u16::d4::distance::squared_euclidean;
use sok::tuned::u16::d4::kdtree::{A, KdTree};

use rayon::prelude::*;

const K: usize = 4;
const BUCKET_SIZE: usize = 32;
const QUERY: usize = 10_000;


const RADIUS: A = A::unwrapped_from_str("0.002");

fn criterion_benchmark(c: &mut Criterion) {
    // Bench building tree
    for ndata in [3, 4, 5, 6, 7].map(|p| 10_usize.pow(p)) {
        let data: Vec<[A; K]> = (0..ndata)
            .map(|_| [(); K].map(|_| {
                let val: u16 = rand::random();
                unsafe { std::mem::transmute(val) }
            }))
            .collect();
        let query: Vec<[A; K]> = (0..QUERY)
            .map(|_| [(); K].map(|_| {
                let val: u16 = rand::random();
                unsafe { std::mem::transmute(val) }
            }))
            .collect();

        let mut group = c.benchmark_group(format!(
            "{:?} within SIMD u16 4D (ndata = {})",
            QUERY, ndata
        ));

        let mut kdtree = KdTree::with_capacity(BUCKET_SIZE);
        for idx in 0..ndata {
            kdtree.add(&data[idx], idx as u32);
        }

        group.bench_function("non-periodic", |b| {
            b.iter(|| {
                let v: Vec<_> = black_box(&query)
                    .par_iter()
                    .map_with(black_box(&kdtree), |t, q| {
                        let results = t.within(q, RADIUS, &squared_euclidean);
                        drop(results);
                    })
                    .collect();
                drop(v)
            })
        });
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
