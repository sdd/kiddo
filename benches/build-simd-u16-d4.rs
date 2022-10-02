use criterion::{black_box, criterion_group, criterion_main, Criterion};

use fixed::types::extra::U14;
use fixed::FixedU16;

use sok::tuned::u16::d4::kdtree::KdTree;

const K: usize = 4;
const BUCKET_SIZE: usize = 32;

fn build_simd_u16_fixed_d4(c: &mut Criterion) {
    // Bench building tree
    for ndata in [3, 4, 5, 6].map(|p| 10_usize.pow(p)) {
        let data: Vec<[FixedU16<U14>; K]> = (0..ndata)
            .map(|_| [(); K].map(|_| {
                let val: u16 = rand::random();
                unsafe { std::mem::transmute(val) }
            }))
            .collect();

        c.bench_function(
            format!("Build SIMD u16 4D (ndata = {ndata})").as_str(),
            |b| {
                b.iter(|| {
                    let mut kdtree = black_box(KdTree::with_capacity(BUCKET_SIZE));
                    for idx in 0..ndata {
                        black_box(kdtree.add(&data[idx], idx as u32))
                    }

                    drop(kdtree)
                })
            },
        );
    }
}

criterion_group!(benches, build_simd_u16_fixed_d4);
criterion_main!(benches);
