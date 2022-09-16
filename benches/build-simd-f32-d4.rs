use criterion::{black_box, criterion_group, criterion_main, Criterion};

use sok::simd::f32::d4::kdtree::KdTree;

const K: usize = 4;
const BUCKET_SIZE: usize = 32;

fn build_simd_f32_d4(c: &mut Criterion) {
    // Bench building tree
    for ndata in [3, 4, 5, 6].map(|p| 10_usize.pow(p)) {
        let data: Vec<[f32; K]> = (0..ndata)
            .map(|_| [(); K].map(|_| rand::random()))
            .collect();

        c.bench_function(
            format!("Build SIMD f32 4D (ndata = {ndata})").as_str(),
            |b| {
                b.iter(|| {
                    let mut kdtree = black_box(KdTree::with_capacity(BUCKET_SIZE));
                    for idx in 0..ndata {
                        black_box(kdtree.add(&data[idx], idx))
                    }

                    drop(kdtree)
                })
            },
        );
    }
}

criterion_group!(benches, build_simd_f32_d4);
criterion_main!(benches);
