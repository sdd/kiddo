use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use fixed::FixedU16;
use fixed::types::extra::U16;

use sok::float::kdtree::KdTree;
use sok::fixed::kdtree::KdTree as FixedKdTree;

const BUCKET_SIZE: usize = 32;

fn rand_data_2d() -> ([f64; 2], u32) {
    rand::random()
}

fn rand_data_3d() -> ([f64; 3], u32) {
    rand::random()
}

fn rand_data_4d() -> ([f64; 4], u32) {
    rand::random()
}

fn rand_data_3d_f32() -> ([f32; 3], u32) {
    rand::random()
}

type FXP = FixedU16<U16>;

fn randfxp() -> FXP {
    let val: u16 = rand::random();
    unsafe { std::mem::transmute(val) }
}

fn rand_data_3d_fixed_u16() -> ([FXP; 3], u32) {
    ([randfxp(), randfxp(), randfxp()], rand::random())
}

pub fn add_100_2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("add 100 items, 2d f64 kdtree");

    for size in [100, 1_000, 10_000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(100));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let points_to_add: Vec<([f64; 2], u32)> =
                (0..100).into_iter().map(|_| rand_data_2d()).collect();

            let mut points = vec![];
            let mut kdtree =
                KdTree::<f64, u32, 2, BUCKET_SIZE, u32>::with_capacity(size + points_to_add.len());
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
    let mut group = c.benchmark_group("add 100 items, 3d f64 kdtree");

    for size in [100, 1_000, 10_000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(100));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let points_to_add: Vec<([f64; 3], u32)> =
                (0..100).into_iter().map(|_| rand_data_3d()).collect();

            let mut points = vec![];
            let mut kdtree =
                KdTree::<f64, u32, 3, BUCKET_SIZE, u32>::with_capacity(size + points_to_add.len());
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
    let mut group = c.benchmark_group("add 100 items, 4d f64 kdtree");

    for size in [100, 1_000, 10_000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(100));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let points_to_add: Vec<([f64; 4], u32)> =
                (0..100).into_iter().map(|_| rand_data_4d()).collect();

            let mut points = vec![];
            let mut kdtree =
                KdTree::<f64, u32, 4, BUCKET_SIZE, u32>::with_capacity(size + points_to_add.len());
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
    let mut group = c.benchmark_group("add 100 items, 3d f32 kdtree");

    for size in [100, 1_000, 10_000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(100));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let points_to_add: Vec<([f32; 3], u32)> =
                (0..100).into_iter().map(|_| rand_data_3d_f32()).collect();

            let mut points = vec![];
            let mut kdtree =
                KdTree::<f32, u32, 3, BUCKET_SIZE, u32>::with_capacity(size + points_to_add.len());
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

pub fn add_100_3d_fixed_u16(c: &mut Criterion) {
    let mut group = c.benchmark_group("add 100 items, 3d fixed point u16 kdtree");

    for size in [100, 1_000, 10_000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(100));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {

            let points_to_add: Vec<([_; 3], u32)> =
                (0..100).into_iter().map(|_| rand_data_3d_fixed_u16()).collect();

            let mut points = vec![];
            let mut kdtree =
                FixedKdTree::<FXP, u32, 3, BUCKET_SIZE, u32>::with_capacity(size + points_to_add.len());
            for _ in 0..size {
                points.push(rand_data_3d_fixed_u16());
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

criterion_group!(benches, add_100_2d, add_100_3d, add_100_4d, add_100_3d_f32, add_100_3d_fixed_u16);
criterion_main!(benches);
