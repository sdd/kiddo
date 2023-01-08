use az::Cast;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput, PlotConfiguration, AxisScale, BenchmarkGroup, BatchSize};
use criterion::measurement::WallTime;
use rand::distributions::Standard;
// use rayon::prelude::*;

use rand_distr::Distribution;
use rand_distr::UnitSphere as SPHERE;
use sok::float::distance::squared_euclidean;
use sok::float::kdtree::{Axis, Content, Index, KdTree};

const K: usize = 3;
const BUCKET_SIZE: usize = 32;
const QUERY: usize = 1_000;

/*fn rand_unit_sphere_point_f32() -> [f32; 3] {
    SPHERE.sample(&mut rand::thread_rng())
}

fn rand_sphere_data() -> ([f32; 3], usize) {
    (rand_unit_sphere_point_f32(), rand::random())
}*/

pub fn nearest_n(c: &mut Criterion) {
    let mut group = c.benchmark_group("Query Nearest n");

    let plot_config = PlotConfiguration::default()
        .summary_scale(AxisScale::Logarithmic);

    group.plot_config(plot_config);

    for &size in [100, 1_000, 10_000, 100_000].iter() {
        bench_query_nearest_n_float::<f64, u32, 2, u16>(&mut group, size, 100,"2D f64");
    }
}

fn bench_query_nearest_n_float<A: Axis, T: Content, const K: usize, IDX: Index<T = IDX>>(group: &mut BenchmarkGroup<WallTime>, initial_size: usize, nearest_qty: usize, subtype: &str) where usize: Cast<IDX>, Standard: Distribution<([A; K], T)> {
    group.bench_with_input(BenchmarkId::new(subtype, initial_size), &initial_size, |b, &size| {
        let mut res: Vec<(A, T)> = Vec::with_capacity(nearest_qty);
        b.iter_batched(|| {
            let points_to_query: Vec<[A; K]> =
                (0..qty_to_add).into_iter().map(|_| rand::random::<[A; K]>()).collect();

            let mut initial_points = vec![];
            for _ in 0..size {
                initial_points.push(rand::random::<([A; K], T)>());
            }
            let mut kdtree =
                KdTree::<A, T, K, BUCKET_SIZE, IDX>::with_capacity(size);

            for i in 0..initial_points.len() {
                kdtree.add(&initial_points[i].0, initial_points[i].1);
            }

            res.clear();

            (kdtree, points_to_query)
        }, |(mut kdtree, points_to_query)| {
            black_box(points_to_query
                .iter()
                .for_each(|point| {
                    res.extend(kdtree.nearest_n(&point, nearest_qty, &squared_euclidean));
                }))
        }, BatchSize::SmallInput);
    });
}

/*pub fn nearest_n_10_euclidean2(c: &mut Criterion) {
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
}*/

criterion_group!(benches, nearest_n);
criterion_main!(benches);
