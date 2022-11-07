use criterion::{black_box, criterion_group, criterion_main, Criterion};

use sok::float::kdtree::KdTree;

type FLT = f32;
const K: usize = 4;
const BUCKET_SIZE: usize = 32;

fn build_float_f32_d4(c: &mut Criterion) {
    // Bench building tree
    for ndata in [3, 4, 5, 6].map(|p| 10_usize.pow(p)) {
        let data: Vec<[f32; K]> = (0..ndata)
            .map(|_| [(); K].map(|_| rand::random()))
            .collect();

        c.bench_function(
            format!("Build float f32 4D (ndata = {ndata})").as_str(),
            |b| {
                b.iter(|| {
                    let mut kdtree: KdTree<FLT, u32, K, BUCKET_SIZE, u16> = KdTree::with_capacity(BUCKET_SIZE);
                    for idx in 0..ndata {
                        black_box(kdtree.add(&data[idx], idx as u32))
                    }

                    drop(kdtree)
                })
            },
        );
    }
}

criterion_group!(benches, build_float_f32_d4);
criterion_main!(benches);
