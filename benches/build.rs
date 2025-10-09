use criterion::{black_box, criterion_group, criterion_main, Criterion};

use kiddo::KdTree;

const K: usize = 3;
const BUCKET_SIZE: usize = 32;

fn build(c: &mut Criterion) {

    // Bench building tree
    for ndata in [3, 4, 5, 6].map(|p| 10_usize.pow(p)) {

        let data: Vec<[f64; K]> = (0..ndata)
            .map(|_| [(); K].map(|_| rand::random()))
            .collect();

        c.bench_function(
            format!("Build (ndata = {ndata})").as_str(),
            |b| {
                b.iter(|| {
                    let mut kdtree =
                        black_box(KdTree::with_capacity(BUCKET_SIZE).unwrap());
                    for idx in 0..ndata {
                        black_box(kdtree.add(&data[idx], idx).unwrap())
                    }

                    drop(kdtree)
                })
            }
        );
    }
}

criterion_group!(benches, build);
criterion_main!(benches);
