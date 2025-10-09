use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use aligned::{Aligned, A16};
use kiddo::distance::squared_euclidean;
use kiddo::KdTree;
use num_traits::FromPrimitive;
use rand_distr::Distribution;
use rand_distr::UnitSphere as SPHERE;

use std::arch::x86_64::*;

union SimdToArray {
    array: [f32; 4],
    simd: __m128,
}

//fn rand_data() -> ([f64; 3], f64) {
//rand::random()
//}

fn rand_unit_sphere_point_f64() -> [f64; 3] {
    SPHERE.sample(&mut rand::thread_rng())
}

fn rand_unit_sphere_point_f32() -> [f32; 3] {
    let sph64: [f64; 3] = SPHERE.sample(&mut rand::thread_rng());
    let res: Aligned<A16, _> = Aligned([
        f32::from_f64(sph64[0]).unwrap(),
        f32::from_f64(sph64[1]).unwrap(),
        f32::from_f64(sph64[2]).unwrap(),
    ]);
    *res
}

fn rand_unit_sphere_point_f32_qwalign() -> [f32; 4] {
    let sph64: [f64; 3] = SPHERE.sample(&mut rand::thread_rng());
    let res: Aligned<A16, _> = Aligned([
        f32::from_f64(sph64[0]).unwrap(),
        f32::from_f64(sph64[1]).unwrap(),
        f32::from_f64(sph64[2]).unwrap(),
        0f32,
    ]);
    *res
}

fn rand_data() -> ([f64; 3], f64) {
    rand::random()
}

fn rand_sphere_data() -> ([f64; 3], usize) {
    (rand_unit_sphere_point_f64(), rand::random())
}

fn rand_sphere_data_f32() -> ([f32; 3], usize) {
    (rand_unit_sphere_point_f32(), rand::random())
}

fn rand_sphere_data_f32_qw() -> ([f32; 4], usize) {
    (rand_unit_sphere_point_f32_qwalign(), rand::random())
}

pub fn nearest_1_euclidean2(c: &mut Criterion) {
    let mut group = c.benchmark_group("nearest(1)");

    for size in [100, 1_000, 10_000, 100_000, 1_000_000].iter() {
        //group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let point = rand_sphere_data();

            let mut points = vec![];
            let mut kdtree = KdTree::with_per_node_capacity(16).unwrap();
            for _ in 0..size {
                points.push(rand_sphere_data());
            }
            for i in 0..points.len() {
                kdtree.add(&points[i].0, points[i].1).unwrap();
            }

            b.iter(|| black_box(kdtree.nearest_one(&point.0, &squared_euclidean)).unwrap());
        });
    }
}

criterion_group!(
    benches,
    nearest_1_euclidean2,
);
criterion_main!(benches);
