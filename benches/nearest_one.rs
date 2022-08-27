use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use rand_distr::Distribution;
use rand_distr::UnitSphere as SPHERE;
use sok::distance::squared_euclidean;
use sok::KdTree;

const BUCKET_SIZE: usize = 32;

fn rand_unit_sphere_point_f64() -> [f64; 3] {
    SPHERE.sample(&mut rand::thread_rng())
}

fn rand_sphere_data() -> ([f64; 3], usize) {
    (rand_unit_sphere_point_f64(), rand::random())
}

pub fn nearest_one_3d(c: &mut Criterion) {
    let mut group = c.benchmark_group("nearest_one");

    for size in [100, 1_000, 10_000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let mut points = vec![];
            let mut kdtree = KdTree::<_, _, 3, BUCKET_SIZE>::with_capacity(size);
            for _ in 0..size {
                points.push(rand_sphere_data());
            }
            for i in 0..points.len() {
                kdtree.add(&points[i].0, points[i].1);
            }

            b.iter(|| black_box(kdtree.nearest_one(&rand_sphere_data().0, &squared_euclidean)));
        });
    }
}

criterion_group!(benches, nearest_one_3d,);
criterion_main!(benches);
