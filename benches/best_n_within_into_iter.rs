use az::Cast;
use criterion::{BenchmarkGroup, BenchmarkId, black_box, Criterion, criterion_group, criterion_main, Throughput};
use criterion::measurement::WallTime;

use rand_distr::Distribution;
use rand_distr::UnitSphere as SPHERE;
use sok::distance::squared_euclidean;
use sok::float::kdtree::KdTree;
use sok::types::Index;

const BUCKET_SIZE: usize = 32;

fn rand_unit_sphere_point_f64() -> [f64; 3] {
    SPHERE.sample(&mut rand::thread_rng())
}

fn rand_sphere_data() -> ([f64; 3], usize) {
    (rand_unit_sphere_point_f64(), rand::random())
}

fn bench_best_n_3d<IDX: Index<T = IDX>> (group: &mut BenchmarkGroup<WallTime>, size: &usize, radius: f64, max_qty: usize) where usize: Cast<IDX> {
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

        b.iter(|| {
            black_box(
                kdtree
                    .best_n_within_into_iter(&point.0, radius, max_qty, &squared_euclidean)
                    .for_each(|x| {
                        let _y = x;
                    }),
            )
        });
    });
}

pub fn best_1_within_small_euclidean2(c: &mut Criterion) {
    let mut group = c.benchmark_group("best 1: within(0.01)");
    group.throughput(Throughput::Elements(1));

    for size in [100, 1_000, 10_000, 100_000].iter() {
        bench_best_n_3d::<u16>(&mut group, size, 0.01, 1);
    }
    for size in [1_000_000].iter() {
        bench_best_n_3d::<u32>(&mut group, size, 0.01, 1);
    }
}

pub fn best_100_within_small_euclidean2(c: &mut Criterion) {
    let mut group = c.benchmark_group("best 100: within(0.01)");
    group.throughput(Throughput::Elements(1));

    for size in [100, 1_000, 10_000, 100_000].iter() {
        bench_best_n_3d::<u16>(&mut group, size, 0.01, 100);
    }
    for size in [1_000_000].iter() {
        bench_best_n_3d::<u32>(&mut group, size, 0.01, 100);
    }
}

pub fn best_100_within_medium_euclidean2(c: &mut Criterion) {
    let mut group = c.benchmark_group("best 100: within(0.05)");
    group.throughput(Throughput::Elements(1));

    for size in [100, 1_000, 10_000, 100_000].iter() {
        bench_best_n_3d::<u16>(&mut group, size, 0.05, 100);
    }
    for size in [1_000_000].iter() {
        bench_best_n_3d::<u32>(&mut group, size, 0.05, 100);
    }
}

pub fn best_100_within_large_euclidean2(c: &mut Criterion) {
    let mut group = c.benchmark_group("best 100: within(0.25)");
    group.throughput(Throughput::Elements(1));

    for size in [100, 1_000, 10_000, 100_000].iter() {
        bench_best_n_3d::<u16>(&mut group, size, 0.25, 100);
    }
    for size in [1_000_000].iter() {
        bench_best_n_3d::<u32>(&mut group, size, 0.25, 100);
    }
}

criterion_group!(
    benches,
    best_1_within_small_euclidean2,
    best_100_within_small_euclidean2,
    best_100_within_medium_euclidean2,
    best_100_within_large_euclidean2
);
criterion_main!(benches);
