use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rayon::prelude::*;
use sok::float::kdtree::KdTree;

use sok::distance::squared_euclidean;

type A = f64;
type I = u32;
const K: usize = 3;
const BUCKET_SIZE: usize = 32;
const QUERY: usize = 1_000_000;

fn criterion_benchmark(c: &mut Criterion) {
    for ndata in [3, 4, 5, 6, 7].map(|p| 10_usize.pow(p)) {
        let data: Vec<[A; K]> = (0..ndata)
            .map(|_| [(); K].map(|_| rand::random()))
            .collect();
        let query: Vec<[A; K]> = (0..QUERY)
            .map(|_| [(); K].map(|_| rand::random()))
            .collect();


        let mut group = c.benchmark_group(
            format!(
                "{} queries (ndata = {})",
                QUERY.to_formatted_string(&Locale::en),
                ndata.to_formatted_string(&Locale::en)
            )
        );
        group.plot_config(
            criterion::PlotConfiguration::default()
                .summary_scale(criterion::AxisScale::Logarithmic)
        );

        for log2k in [5] {
            let mut kdtree = KdTree::<A, I, K, BUCKET_SIZE, u32>::with_capacity(ndata);
            for idx in 0..ndata {
                kdtree.add(&data[idx], idx as I);
            }

            group.bench_function(
                format!("Query nearest n"),
                |b| {
                    b.iter(|| {
                        let v: Vec<_> = black_box(&query)
                            .par_iter()
                            .map_with(black_box(&tree), |t, q| {
                                let results = black_box(t.nearest(black_box(q), black_box(2_usize.pow(log2k))));
                                drop(results);
                            })
                            .collect();
                        drop(v)
                    })
                },
            );
        }
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
