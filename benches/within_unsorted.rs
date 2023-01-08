use az::Cast;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput, BenchmarkGroup};
use criterion::measurement::WallTime;

use rand_distr::Distribution;
use rand_distr::UnitSphere as SPHERE;
use sok::distance::squared_euclidean;
use sok::float::kdtree::{Index, KdTree};

const BUCKET_SIZE: usize = 32;

fn rand_unit_sphere_point_f64() -> [f64; 3] {
    SPHERE.sample(&mut rand::thread_rng())
}

fn rand_sphere_data() -> ([f64; 3], usize) {
    (rand_unit_sphere_point_f64(), rand::random())
}

fn bench_within_unsorted_3d<IDX: Index<T = IDX>>(group: &mut BenchmarkGroup<WallTime>, size: &usize, radius: f64) where usize: Cast<IDX>  {
    group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
        let point = rand_sphere_data();

        let mut points = vec![];
        let mut kdtree = KdTree::<_, _, 3, BUCKET_SIZE, IDX>::with_capacity(size);
        for _ in 0..size {
            points.push(rand_sphere_data());
        }
        for i in 0..points.len() {
            kdtree.add(&points[i].0, points[i].1);
        }

        b.iter(|| black_box(kdtree.within_unsorted(&point.0, radius, &squared_euclidean)));
    });
}

pub fn within_unsorted_small_euclidean2(c: &mut Criterion) {
    let mut group = c.benchmark_group("within_unsorted(0.01)");
    group.throughput(Throughput::Elements(1));

    for size in [100, 1_000, 10_000, 100_000, 1_000_000].iter() {
        bench_within_unsorted_3d::<u16>(&mut group, size, 0.01);
    }
    for size in [1_000_000].iter() {
        bench_within_unsorted_3d::<u32>(&mut group, size, 0.01);
    }
}

pub fn within_unsorted_medium_euclidean2(c: &mut Criterion) {
    let mut group = c.benchmark_group("within_unsorted(0.05)");
    group.throughput(Throughput::Elements(1));

    for size in [100, 1_000, 10_000, 100_000, 1_000_000].iter() {
        bench_within_unsorted_3d::<u16>(&mut group, size, 0.05);
    }
    for size in [1_000_000].iter() {
        bench_within_unsorted_3d::<u32>(&mut group, size, 0.05);
    }
}

pub fn within_unsorted_large_euclidean2(c: &mut Criterion) {
    let mut group = c.benchmark_group("within_unsorted(0.25)");
    group.throughput(Throughput::Elements(1));

    for size in [100, 1_000, 10_000, 100_000, 1_000_000].iter() {
        bench_within_unsorted_3d::<u16>(&mut group, size, 0.25);
    }
    for size in [1_000_000].iter() {
        bench_within_unsorted_3d::<u32>(&mut group, size, 0.25);
    }
}

criterion_group!(
    benches,
    within_unsorted_small_euclidean2,
    within_unsorted_medium_euclidean2,
    within_unsorted_large_euclidean2,
);
criterion_main!(benches);
