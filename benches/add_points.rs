use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use sok::KdTree;

const BUCKET_SIZE: usize = 32;

fn rand_data_2d() -> ([f64; 2], usize) {
    rand::random()
}

fn rand_data_3d() -> ([f64; 3], usize) {
    rand::random()
}

fn rand_data_4d() -> ([f64; 4], usize) {
    rand::random()
}

fn rand_data_3d_f32() -> ([f32; 3], usize) {
    rand::random()
}

pub fn add_100_2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("add 100 items to 2d kdtree of increasing size");

    for size in [100, 1_000, 10_000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(100));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let points_to_add: Vec<([f64; 2], usize)> =
                (0..100).into_iter().map(|_| rand_data_2d()).collect();

            let mut points = vec![];
            let mut kdtree =
                KdTree::<_, _, 2, BUCKET_SIZE>::with_capacity(size + points_to_add.len());
            for _ in 0..size {
                points.push(rand_data_2d());
            }
            for i in 0..points.len() {
                kdtree.add(&points[i].0, points[i].1);
            }

            b.iter(|| {
                points_to_add
                    .iter()
                    .for_each(|point| black_box(kdtree.add(black_box(&point.0), point.1)))
            });
        });
    }
}

pub fn add_100_3d(c: &mut Criterion) {
    let mut group = c.benchmark_group("add 100 items to 3d kdtree of increasing size");

    for size in [100, 1_000, 10_000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(100));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let points_to_add: Vec<([f64; 3], usize)> =
                (0..100).into_iter().map(|_| rand_data_3d()).collect();

            let mut points = vec![];
            let mut kdtree =
                KdTree::<_, _, 3, BUCKET_SIZE>::with_capacity(size + points_to_add.len());
            for _ in 0..size {
                points.push(rand_data_3d());
            }
            for i in 0..points.len() {
                kdtree.add(&points[i].0, points[i].1);
            }

            b.iter(|| {
                points_to_add
                    .iter()
                    .for_each(|point| black_box(kdtree.add(black_box(&point.0), point.1)))
            });
        });
    }
}

pub fn add_100_4d(c: &mut Criterion) {
    let mut group = c.benchmark_group("add 100 items to d4 kdtree of increasing size");

    for size in [100, 1_000, 10_000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(100));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let points_to_add: Vec<([f64; 4], usize)> =
                (0..100).into_iter().map(|_| rand_data_4d()).collect();

            let mut points = vec![];
            let mut kdtree =
                KdTree::<_, _, 4, BUCKET_SIZE>::with_capacity(size + points_to_add.len());
            for _ in 0..size {
                points.push(rand_data_4d());
            }
            for i in 0..points.len() {
                kdtree.add(&points[i].0, points[i].1);
            }

            b.iter(|| {
                points_to_add
                    .iter()
                    .for_each(|point| black_box(kdtree.add(black_box(&point.0), point.1)))
            });
        });
    }
}

pub fn add_100_3d_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("add 100 items to 3d kdtree (f32) of increasing size");

    for size in [100, 1_000, 10_000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(100));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let points_to_add: Vec<([f32; 3], usize)> =
                (0..100).into_iter().map(|_| rand_data_3d_f32()).collect();

            let mut points = vec![];
            let mut kdtree =
                KdTree::<_, _, 3, BUCKET_SIZE>::with_capacity(size + points_to_add.len());
            for _ in 0..size {
                points.push(rand_data_3d_f32());
            }
            for i in 0..points.len() {
                kdtree.add(&points[i].0, points[i].1);
            }

            b.iter(|| {
                points_to_add
                    .iter()
                    .for_each(|point| black_box(kdtree.add(black_box(&point.0), point.1)))
            });
        });
    }
}

criterion_group!(benches, add_100_2d, add_100_3d, add_100_4d, add_100_3d_f32);
criterion_main!(benches);
