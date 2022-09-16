use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
// use rayon::prelude::*;

use rand_distr::Distribution;
use rand_distr::UnitSphere as SPHERE;
use sok::distance::squared_euclidean;
use sok::KdTree;

const K: usize = 3;
const BUCKET_SIZE: usize = 32;
const QUERY: usize = 1_000_000;

fn rand_unit_sphere_point_f64() -> [f64; 3] {
    SPHERE.sample(&mut rand::thread_rng())
}

fn rand_sphere_data() -> ([f64; 3], usize) {
    (rand_unit_sphere_point_f64(), rand::random())
}

pub fn nearest_one_3d(c: &mut Criterion) {
    let mut group = c.benchmark_group("nearest_one");

    for &size in [100_000].iter() {
        let data: Vec<_> = (0..size).map(|_x| rand_sphere_data()).collect();

        let mut kdtree = KdTree::<_, _, K, BUCKET_SIZE>::with_capacity(size as usize);

        for point in data {
            kdtree.add(&point.0, point.1);
        }

        let query_points: Vec<[f64; K]> = (0..QUERY)
            .map(|_| [(); K].map(|_| rand::random()))
            .collect();

        // group.throughput(Throughput::Elements(QUERY.try_into().unwrap()));
        // group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
        //     b.iter(|| {
        //         let _v: Vec<_> = black_box(&query_points)
        //             .iter()
        //             // .par_iter()
        //             .map_with(black_box(&kdtree), |t, q| {
        //                 let res = t.nearest_one(black_box(&q), &squared_euclidean);
        //                 drop(res)
        //             })
        //             .collect();
        //
        //        // black_box(kdtree.nearest_one(&rand_sphere_data().0, &squared_euclidean))
        //     });
        // });

        group.throughput(Throughput::Elements(QUERY.try_into().unwrap()));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                black_box(&query_points).iter().for_each(|point| {
                    let res = black_box(kdtree.nearest_one(point, &squared_euclidean));
                })
            });
        });
    }
}

criterion_group!(benches, nearest_one_3d,);
criterion_main!(benches);
