use criterion::{black_box, criterion_group, criterion_main, Criterion};

use sok::float::kdtree::KdTree;

type A = f64;
type I = u32;
const K: usize = 3;
const BUCKET_SIZE: usize = 32;

fn build(c: &mut Criterion) {
    // Bench building tree
    for ndata in [3, 4, 5, 6].map(|p| 10_usize.pow(p)) {
        let data: Vec<[A; K]> = (0..ndata)
            .map(|_| [(); K].map(|_| rand::random()))
            .collect();

        c.bench_function(format!("Build (ndata = {ndata})").as_str(), |b| {
            b.iter(|| {
                let mut kdtree = black_box(KdTree::<_, I, K, BUCKET_SIZE, u32>::with_capacity(ndata));
                for idx in 0..ndata {
                    black_box(kdtree.add(&data[idx], idx as I))
                }

                drop(kdtree)
            })
        });
    }
}

criterion_group!(benches, build);
criterion_main!(benches);
