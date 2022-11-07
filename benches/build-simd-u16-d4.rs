use criterion::{black_box, criterion_group, criterion_main, Criterion};

use fixed::types::extra::{U16, U14};
use fixed::FixedU16;

use sok::tuned::u16::d4::kdtree::KdTree;
use sok::fixed::kdtree::KdTree as KdTreeFixed;

const K: usize = 4;
const BUCKET_SIZE: usize = 32;

fn build_simd_u16_fixed_d4(c: &mut Criterion) {
    // Bench building tree
    for ndata in [3, 4, 5, 6].map(|p| 10_usize.pow(p)) {
        let data: Vec<[FixedU16<U16>; K]> = (0..ndata)
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

fn build_fixed(c: &mut Criterion) {
    const K: usize = 3;
    type FXA = FixedU16<U14>;

    for ndata in [3, 4, 5, 6].map(|p| 10_usize.pow(p)) {
        let data: Vec<[FXA; K]> = (0..ndata)
            .map(|_| [(); K].map(|_| {
                let val: u16 = rand::random();
                unsafe { std::mem::transmute(val) }
            }))
            .collect();

        c.bench_function(
            format!("Build generic fixed (ndata = {ndata})").as_str(),
            |b| {
                b.iter(|| {
                    let mut kdtree = black_box(
                        KdTreeFixed::<FXA, u32, K, 32, u32>::with_capacity(BUCKET_SIZE)
                    );
                    for idx in 0..ndata {
                        black_box(kdtree.add(&data[idx], idx as u32))
                    }

                    drop(kdtree)
                })
            },
        );
    }
}

criterion_group!(benches, build_simd_u16_fixed_d4, build_fixed);
criterion_main!(benches);
