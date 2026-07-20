#![cfg_attr(coverage_nightly, feature(coverage_attribute))]
#![cfg_attr(coverage_nightly, coverage(off))]

use kiddo::kd_tree::KdTree;
use kiddo::leaf_strategy::FlatVec;
use kiddo::stem_strategy::Eytzinger;
use kiddo::test_utils::NearestNBenchmarkCollector;
use kiddo::SquaredEuclidean;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::hint::black_box;
use std::num::NonZeroUsize;
use std::time::Instant;

const K: usize = 3;
const B: usize = 32;
const DEFAULT_QUERY_COUNT: usize = 1_024;
const DEFAULT_SAMPLE_COUNT: usize = 5;
const DEFAULT_TREE_SIZES: &str = "262144,1048576,4194304";
const DEFAULT_MAX_QTYS: &str = "16,20,24,32,48,64,80,96,112,128,144,160,176,192,208,224,256";
const POINT_SEED: u64 = 0x5eed_0000_0000_0001;
const QUERY_SEED: u64 = 0x5eed_0000_0000_0002;

type F64Tree = KdTree<f64, u32, Eytzinger, FlatVec<f64, u32, K, B>, K, B>;
type F32Tree = KdTree<f32, u32, Eytzinger, FlatVec<f32, u32, K, B>, K, B>;

#[derive(Clone, Copy)]
struct Measurement {
    point_count: usize,
    axis: &'static str,
    sorted: bool,
    max_qty: usize,
    heap_ns: f64,
    fused_ns: f64,
}

fn read_usize_env(var: &str, default: usize) -> usize {
    std::env::var(var)
        .ok()
        .map(|value| {
            value
                .parse::<usize>()
                .unwrap_or_else(|_| panic!("{var} must be a positive integer"))
        })
        .unwrap_or(default)
}

fn read_usize_list_env(var: &str, default: &str) -> Vec<usize> {
    let value = std::env::var(var).unwrap_or_else(|_| default.to_owned());
    let mut values: Vec<_> = value
        .split(',')
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .map(|item| {
            item.parse::<usize>()
                .unwrap_or_else(|_| panic!("{var} contains an invalid integer: {item}"))
        })
        .collect();
    assert!(!values.is_empty(), "{var} must not be empty");
    assert!(
        values.iter().all(|value| *value > 0),
        "{var} values must be positive"
    );
    values.sort_unstable();
    values.dedup();
    values
}

fn build_points_f64(point_count: usize) -> Vec<[f64; K]> {
    let mut rng = ChaCha8Rng::seed_from_u64(POINT_SEED);
    (0..point_count).map(|_| rng.random::<[f64; K]>()).collect()
}

fn build_queries_f64(query_count: usize) -> Vec<[f64; K]> {
    let mut rng = ChaCha8Rng::seed_from_u64(QUERY_SEED);
    (0..query_count).map(|_| rng.random::<[f64; K]>()).collect()
}

fn build_points_f32(point_count: usize) -> Vec<[f32; K]> {
    build_points_f64(point_count)
        .into_iter()
        .map(|point| point.map(|coord| coord as f32))
        .collect()
}

fn build_queries_f32(query_count: usize) -> Vec<[f32; K]> {
    build_queries_f64(query_count)
        .into_iter()
        .map(|point| point.map(|coord| coord as f32))
        .collect()
}

fn median(mut samples: Vec<f64>) -> f64 {
    assert!(!samples.is_empty());
    samples.sort_unstable_by(f64::total_cmp);
    let midpoint = samples.len() / 2;
    if samples.len().is_multiple_of(2) {
        (samples[midpoint - 1] + samples[midpoint]) / 2.0
    } else {
        samples[midpoint]
    }
}

macro_rules! axis_benchmark {
    ($measure:ident, $run:ident, $checksum:ident, $tree:ty, $axis:ty, $metric:ty) => {
        fn $run(
            tree: &$tree,
            queries: &[[$axis; K]],
            max_qty: NonZeroUsize,
            sorted: bool,
            collector: NearestNBenchmarkCollector,
        ) -> f64 {
            let start = Instant::now();
            let mut result_len = 0usize;
            for query in queries {
                let results = tree.nearest_n_with_forced_collector::<$metric>(
                    black_box(query),
                    max_qty,
                    sorted,
                    collector,
                );
                result_len = result_len.wrapping_add(results.len());
                black_box(results);
            }
            black_box(result_len);
            start.elapsed().as_nanos() as f64 / queries.len() as f64
        }

        fn $checksum(
            tree: &$tree,
            queries: &[[$axis; K]],
            max_qty: NonZeroUsize,
            sorted: bool,
            collector: NearestNBenchmarkCollector,
        ) -> (usize, u64, u64) {
            let mut result_len = 0usize;
            let mut item_sum = 0u64;
            let mut distance_sum = 0u64;
            for query in queries {
                let results = tree
                    .nearest_n_with_forced_collector::<$metric>(query, max_qty, sorted, collector);
                result_len = result_len.wrapping_add(results.len());
                for result in results {
                    item_sum = item_sum.wrapping_add(result.item as u64);
                    distance_sum = distance_sum.wrapping_add(result.distance.to_bits() as u64);
                }
            }
            (result_len, item_sum, distance_sum)
        }

        fn $measure(
            tree: &$tree,
            queries: &[[$axis; K]],
            point_count: usize,
            axis: &'static str,
            max_qty: usize,
            sorted: bool,
            sample_count: usize,
        ) -> Measurement {
            let max_qty = NonZeroUsize::new(max_qty).unwrap();
            let collectors = [
                NearestNBenchmarkCollector::BinaryHeap,
                NearestNBenchmarkCollector::ThresholdVecFused,
            ];

            let expected = $checksum(tree, queries, max_qty, sorted, collectors[0]);
            assert_eq!(
                $checksum(tree, queries, max_qty, sorted, collectors[1]),
                expected,
                "collector mismatch: points={point_count} axis={axis} sorted={sorted} k={max_qty}"
            );

            let mut samples = [Vec::new(), Vec::new()];
            for sample_idx in 0..sample_count {
                for offset in 0..collectors.len() {
                    let collector_idx = (sample_idx + offset) % collectors.len();
                    samples[collector_idx].push($run(
                        tree,
                        queries,
                        max_qty,
                        sorted,
                        collectors[collector_idx],
                    ));
                }
            }

            Measurement {
                point_count,
                axis,
                sorted,
                max_qty: max_qty.get(),
                heap_ns: median(std::mem::take(&mut samples[0])),
                fused_ns: median(std::mem::take(&mut samples[1])),
            }
        }
    };
}

axis_benchmark!(
    measure_f64,
    run_f64,
    checksum_f64,
    F64Tree,
    f64,
    SquaredEuclidean<f64>
);
axis_benchmark!(
    measure_f32,
    run_f32,
    checksum_f32,
    F32Tree,
    f32,
    SquaredEuclidean<f32>
);

fn print_measurement(measurement: Measurement) {
    let fused_speedup = (measurement.heap_ns - measurement.fused_ns) / measurement.heap_ns * 100.0;
    println!(
        "case,{},{},{},{},{:.2},{:.2},{:.2},{}",
        measurement.point_count,
        measurement.axis,
        if measurement.sorted {
            "sorted"
        } else {
            "unsorted"
        },
        measurement.max_qty,
        measurement.heap_ns,
        measurement.fused_ns,
        fused_speedup,
        if fused_speedup > 0.0 { "fused" } else { "heap" }
    );
}

fn print_recommendations(measurements: &[Measurement], max_qties: &[usize]) {
    println!(
        "summary_header,mode,k,fused_wins,total,worst_fused_speedup_pct,median_fused_speedup_pct"
    );
    for sorted in [true, false] {
        let mut largest_unanimous_win = None;
        let mut first_unanimous_heap_win = None;

        for &max_qty in max_qties {
            let matching: Vec<_> = measurements
                .iter()
                .filter(|measurement| {
                    measurement.sorted == sorted && measurement.max_qty == max_qty
                })
                .collect();
            let speedups: Vec<_> = matching
                .iter()
                .map(|measurement| {
                    (measurement.heap_ns - measurement.fused_ns) / measurement.heap_ns * 100.0
                })
                .collect();
            let wins = speedups.iter().filter(|speedup| **speedup > 0.0).count();
            let worst = speedups.iter().copied().fold(f64::INFINITY, f64::min);
            let median_speedup = median(speedups);
            if wins == matching.len() {
                largest_unanimous_win = Some(max_qty);
            } else if wins == 0 && first_unanimous_heap_win.is_none() {
                first_unanimous_heap_win = Some(max_qty);
            }
            println!(
                "summary,{},{},{},{},{:.2},{:.2}",
                if sorted { "sorted" } else { "unsorted" },
                max_qty,
                wins,
                matching.len(),
                worst,
                median_speedup
            );
        }

        println!(
            "recommendation,{},largest_unanimous_fused_win={},first_unanimous_heap_win={}",
            if sorted { "sorted" } else { "unsorted" },
            largest_unanimous_win
                .map(|value| value.to_string())
                .unwrap_or_else(|| "none".to_owned()),
            first_unanimous_heap_win
                .map(|value| value.to_string())
                .unwrap_or_else(|| "none".to_owned())
        );
    }
}

fn main() {
    let tree_sizes = read_usize_list_env("KIDDO_PROFILE_TREE_SIZES", DEFAULT_TREE_SIZES);
    let max_qties = read_usize_list_env("KIDDO_PROFILE_MAX_QTYS", DEFAULT_MAX_QTYS);
    let query_count = read_usize_env("KIDDO_PROFILE_QUERIES", DEFAULT_QUERY_COUNT);
    let sample_count = read_usize_env("KIDDO_PROFILE_SAMPLES", DEFAULT_SAMPLE_COUNT);
    assert!(query_count > 0, "KIDDO_PROFILE_QUERIES must be positive");
    assert!(sample_count > 0, "KIDDO_PROFILE_SAMPLES must be positive");

    println!(
        "# result collection threshold profile: arch={} dims={} tree_sizes={:?} queries={} samples={} ks={:?}",
        std::env::consts::ARCH,
        K,
        tree_sizes,
        query_count,
        sample_count,
        max_qties
    );
    println!("case_header,points,axis,mode,k,heap_ns,fused_ns,fused_speedup_pct,winner");

    let f64_queries = build_queries_f64(query_count);
    let f32_queries = build_queries_f32(query_count);
    let mut measurements = Vec::new();

    for point_count in tree_sizes {
        let f64_tree = F64Tree::new_from_slice(&build_points_f64(point_count)).unwrap();
        for sorted in [true, false] {
            for &max_qty in &max_qties {
                let measurement = measure_f64(
                    &f64_tree,
                    &f64_queries,
                    point_count,
                    "f64",
                    max_qty,
                    sorted,
                    sample_count,
                );
                print_measurement(measurement);
                measurements.push(measurement);
            }
        }

        let f32_tree = F32Tree::new_from_slice(&build_points_f32(point_count)).unwrap();
        for sorted in [true, false] {
            for &max_qty in &max_qties {
                let measurement = measure_f32(
                    &f32_tree,
                    &f32_queries,
                    point_count,
                    "f32",
                    max_qty,
                    sorted,
                    sample_count,
                );
                print_measurement(measurement);
                measurements.push(measurement);
            }
        }
    }

    print_recommendations(&measurements, &max_qties);
}
