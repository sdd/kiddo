use az::Cast;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput, BenchmarkGroup, BatchSize, PlotConfiguration, AxisScale};
use criterion::measurement::WallTime;
use fixed::FixedU16;
use fixed::types::extra::U16;
use rand::distributions::{Distribution, Standard};

use sok::float::kdtree::{Axis, Content, Index, KdTree};
use sok::fixed::kdtree::{Axis as AxisFixed, Content as ContentFixed, Index as IndexFixed, KdTree as FixedKdTree};

const BUCKET_SIZE: usize = 32;

type FXP = FixedU16<U16>;

fn rand_data_fxp() -> FXP {
    let val: u16 = rand::random();
    unsafe { std::mem::transmute(val) }
}

fn rand_data_fixed_u16_3d() -> ([FXP; 3], u32) {
    ([rand_data_fxp(), rand_data_fxp(), rand_data_fxp()], rand::random())
}

pub fn add_to_empty(c: &mut Criterion) {
    let mut group = c.benchmark_group("Add to Empty Tree");

    let plot_config = PlotConfiguration::default()
        .summary_scale(AxisScale::Logarithmic);

    group.plot_config(plot_config);

    for &size in [100, 1_000, 10_000, 100_000].iter() {
        bench_add_to_empty_float::<f64, u32, 2, u16>(&mut group, size, "2D f64");
    }

    for &size in [1_000_000].iter() {
        bench_add_to_empty_float::<f64, u32, 2, u32>(&mut group, size, "2D f64");
    }

    for &size in [100, 1_000, 10_000, 100_000].iter() {
        bench_add_to_empty_float::<f64, u32, 3, u16>(&mut group, size, "3D f64");
    }

    for &size in [1_000_000].iter() {
        bench_add_to_empty_float::<f64, u32, 3, u32>(&mut group,  size, "3D f64");
    }

    for &size in [100, 1_000, 10_000, 100_000].iter() {
        bench_add_to_empty_float::<f64, u32, 4, u16>(&mut group,  size, "4D f64");
    }

    for &size in [1_000_000].iter() {
        bench_add_to_empty_float::<f64, u32, 4, u32>(&mut group,  size, "4D f64");
    }

    for &size in [100, 1_000, 10_000, 100_000].iter() {
        bench_add_to_empty_float::<f32, u32, 3, u16>(&mut group, size, "3D f32");
    }

    for &size in [1_000_000].iter() {
        bench_add_to_empty_float::<f32, u32, 3, u32>(&mut group, size, "3D f32");
    }

    for &size in [100, 1_000, 10_000, 100_000].iter() {
        bench_add_to_empty_3d_fixed::<u16>(&mut group, size, "3D u16");

        // TODO: could not get this to fully work with fixed. Could not figure out
        //       syntax to get the random fixed-point data working generically.
        // bench_add_fixed::<FXP, u32, 3, u16>(&mut group, size, 100);
    }

    for &size in [1_000_000].iter() {
        bench_add_to_empty_3d_fixed::<u32>(&mut group, size, "3D u16");
    }

    group.finish();
}

pub fn add_to_populated(c: &mut Criterion) {
    let mut group = c.benchmark_group("add to Populated Tree");
    group.throughput(Throughput::Elements(100));

    let plot_config = PlotConfiguration::default()
        .summary_scale(AxisScale::Logarithmic);

    group.plot_config(plot_config);

    for &size in [100, 1_000, 10_000, 100_000].iter() {
        bench_add_to_full_float::<f64, u32, 2, u16>(&mut group, size, 100, "2D f64");
    }

    for &size in [1_000_000].iter() {
        bench_add_to_full_float::<f64, u32, 2, u32>(&mut group, size, 100, "2D f64");
    }

    for &size in [100, 1_000, 10_000, 100_000].iter() {
        bench_add_to_full_float::<f64, u32, 3, u16>(&mut group, size, 100, "3D f64");
    }

    for &size in [1_000_000].iter() {
        bench_add_to_full_float::<f64, u32, 3, u32>(&mut group, size, 100, "3D f64");
    }

    for &size in [100, 1_000, 10_000, 100_000].iter() {
        bench_add_to_full_float::<f64, u32, 4, u16>(&mut group, size, 100, "4D f64");
    }

    for &size in [1_000_000].iter() {
        bench_add_to_full_float::<f64, u32, 4, u32>(&mut group, size, 100, "4D f64");
    }

    for &size in [100, 1_000, 10_000, 100_000].iter() {
        bench_add_to_full_float::<f32, u32, 3, u16>(&mut group, size, 100, "3D f32");
    }

    for &size in [1_000_000].iter() {
        bench_add_to_full_float::<f32, u32, 3, u32>(&mut group, size, 100, "3D f32");
    }

    for &size in [100, 1_000, 10_000, 100_000].iter() {
        bench_add_to_full_3d_fixed::<u16>(&mut group, size, 100, "3D u16");

        // TODO: could not get this to fully work with fixed. Could not figure out
        //       syntax to get the random fixed-point data working generically.
        // bench_add_fixed::<FXP, u32, 3, u16>(&mut group, size, 100);
    }

    for &size in [1_000_000].iter() {
        bench_add_to_full_3d_fixed::<u32>(&mut group, size, 100, "3D u16");
    }

    group.finish();
}

fn bench_add_to_empty_float<A: Axis, T: Content, const K: usize, IDX: Index<T = IDX>>(group: &mut BenchmarkGroup<WallTime>, qty_to_add: usize, subtype: &str) where usize: Cast<IDX>, Standard: Distribution<([A; K], T)> {
    group.bench_with_input(BenchmarkId::new(subtype, qty_to_add), &qty_to_add, |b, &size| {
        b.iter_batched(|| {
                let points_to_add: Vec<([A; K], T)> =
                    (0..size).into_iter().map(|_| rand::random::<([A; K], T)>()).collect();

                let kdtree =
                    KdTree::<A, T, K, BUCKET_SIZE, IDX>::with_capacity(points_to_add.len());

                (kdtree, points_to_add)
            }, |(mut kdtree, points_to_add)| {
            black_box(points_to_add
                .iter()
                .for_each(|point| kdtree.add(&point.0, point.1)))
        }, BatchSize::SmallInput);
    });
}

fn bench_add_to_full_float<A: Axis, T: Content, const K: usize, IDX: Index<T = IDX>>(group: &mut BenchmarkGroup<WallTime>, initial_size: usize, qty_to_add: usize, subtype: &str) where usize: Cast<IDX>, Standard: Distribution<([A; K], T)> {
    group.bench_with_input(BenchmarkId::new(subtype, initial_size), &initial_size, |b, &size| {
        b.iter_batched(|| {
            let points_to_add: Vec<([A; K], T)> =
                (0..qty_to_add).into_iter().map(|_| rand::random::<([A; K], T)>()).collect();

            let mut initial_points = vec![];
            for _ in 0..size {
                initial_points.push(rand::random::<([A; K], T)>());
            }
            let mut kdtree =
                KdTree::<A, T, K, BUCKET_SIZE, IDX>::with_capacity(size + points_to_add.len());

            for i in 0..initial_points.len() {
                kdtree.add(&initial_points[i].0, initial_points[i].1);
            }

            (kdtree, points_to_add)
        }, |(mut kdtree, points_to_add)| {
            black_box(points_to_add
                .iter()
                .for_each(|point| kdtree.add(&point.0, point.1)))
        }, BatchSize::SmallInput);
    });
}

#[allow(dead_code)]
fn bench_add_fixed<A: AxisFixed, T: ContentFixed, const K: usize, IDX: IndexFixed<T = IDX>>(group: &mut BenchmarkGroup<WallTime>, _initial_size: usize, qty_to_add: usize, subtype: &str, param_value: usize) where usize: Cast<IDX>, Standard: Distribution<([A; K], T)> {
    group.bench_with_input(BenchmarkId::new(subtype, param_value), &qty_to_add, |b, &size| {
        b.iter_batched(|| {
            let points_to_add: Vec<([A; K], T)> =
                (0..qty_to_add).into_iter().map(|_| rand::random::<([A; K], T)>()).collect();

            let mut points = vec![];
            let mut kdtree =
                FixedKdTree::<A, T, K, BUCKET_SIZE, IDX>::with_capacity(size + points_to_add.len());
            for _ in 0..size {
                points.push(rand::random::<([A; K], T)>());
            }
            for i in 0..points.len() {
                kdtree.add(&points[i].0, points[i].1);
            }

            (kdtree, points_to_add)
        }, |(mut kdtree, points_to_add)| {
            black_box(points_to_add
                .iter()
                .for_each(|point| kdtree.add(&point.0, point.1)))
        }, BatchSize::SmallInput);
    });
}

fn bench_add_to_empty_3d_fixed<IDX: IndexFixed<T = IDX>>(group: &mut BenchmarkGroup<WallTime>, qty_to_add: usize, subtype: &str) where usize: Cast<IDX>  {
    group.bench_with_input(BenchmarkId::new(subtype, qty_to_add), &qty_to_add, |b, &size| {
        b.iter_batched(|| {
            let points_to_add: Vec<([_; 3], u32)> =
                (0..qty_to_add).into_iter().map(|_| rand_data_fixed_u16_3d()).collect();

            let kdtree =
                FixedKdTree::<FXP, u32, 3, BUCKET_SIZE, IDX>::with_capacity(size + points_to_add.len());

            (kdtree, points_to_add)
        }, |(mut kdtree, points_to_add)| {
            points_to_add
                .iter()
                .for_each(|point| black_box(kdtree.add(black_box(&point.0), point.1)))
        }, BatchSize::SmallInput);
    });
}

fn bench_add_to_full_3d_fixed<IDX: IndexFixed<T = IDX>>(group: &mut BenchmarkGroup<WallTime>, initial_size: usize, qty_to_add: usize, subtype: &str) where usize: Cast<IDX>  {
    group.bench_with_input(BenchmarkId::new(subtype, initial_size), &initial_size, |b, &size| {
        b.iter_batched(|| {
            let points_to_add: Vec<([_; 3], u32)> =
                (0..qty_to_add).into_iter().map(|_| rand_data_fixed_u16_3d()).collect();

            let mut points = vec![];
            let mut kdtree =
                FixedKdTree::<FXP, u32, 3, BUCKET_SIZE, IDX>::with_capacity(size + points_to_add.len());
            for _ in 0..size {
                points.push(rand_data_fixed_u16_3d());
            }
            for i in 0..points.len() {
                kdtree.add(&points[i].0, points[i].1);
            }

            (kdtree, points_to_add)
        }, |(mut kdtree, points_to_add)| {
            points_to_add
                .iter()
                .for_each(|point| black_box(kdtree.add(black_box(&point.0), point.1)))
        }, BatchSize::SmallInput);
    });
}

criterion_group!(benches, add_to_empty, add_to_populated);
criterion_main!(benches);
