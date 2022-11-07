use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
// use rayon::prelude::*;

use rand_distr::Distribution;
use rand_distr::UnitSphere as SPHERE;
use sok::float::distance::squared_euclidean;
use sok::float::kdtree::KdTree;

const K: usize = 3;
const BUCKET_SIZE: usize = 32;
const QUERY: usize = 1_000;

fn rand_unit_sphere_point_f32() -> [f32; 3] {
    SPHERE.sample(&mut rand::thread_rng())
}

fn rand_sphere_data() -> ([f32; 3], usize) {
    (rand_unit_sphere_point_f32(), rand::random())
}

pub fn nearest_n_10_euclidean2(c: &mut Criterion) {
    let mut group = c.benchmark_group("nearest_n(10)");

    for size in [1_000, 10_000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(1000));

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let data: Vec<_> = (0..size).map(|_x| rand_sphere_data()).collect();

            let mut kdtree = KdTree::<_, _, K, BUCKET_SIZE, u32>::with_capacity(size as usize);

            for point in data {
                kdtree.add(&point.0, point.1);
            }

            let query_points: Vec<[f32; K]> = (0..QUERY)
                .map(|_| [(); K].map(|_| rand::random()))
                .collect();

            b.iter(|| black_box({
                for point in query_points.iter() {
                    let _res: Vec<_> = kdtree.nearest_n(&point, 10, &squared_euclidean).collect();
                }
            }));
        });
    }
}

criterion_group!(benches, nearest_n_10_euclidean2);
criterion_main!(benches);
