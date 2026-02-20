use kiddo::kd_tree::leaf_strategies::FlatVec;
use kiddo::kd_tree::KdTree;
use kiddo::stem_strategies::{Block3, Block4, Donnelly, DonnellyMarkerSimd};
use kiddo::traits::DistanceMetric;
use kiddo::traits_unified_2::SquaredEuclidean;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const CONTENT_SEED: u64 = 1_260_253_197;
const QUERY_SEED: u64 = 13_787_848_794_416_797_126;
const POINT_COUNT: usize = 1_022;

fn build_points_f32_k2() -> Vec<[f32; 2]> {
    let mut rng = StdRng::seed_from_u64(CONTENT_SEED);
    (0..POINT_COUNT)
        .map(|_| [rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0)])
        .collect()
}

fn build_query_f32_k2() -> [f32; 2] {
    let mut rng = StdRng::seed_from_u64(QUERY_SEED);
    [rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0)]
}

fn linear_search(points: &[[f32; 2]], query: &[f32; 2]) -> (f32, usize) {
    let mut best_dist = f32::MAX;
    let mut best_idx = 0usize;
    for (idx, point) in points.iter().enumerate() {
        let dist = kiddo::SquaredEuclidean::dist(query, point);
        if dist < best_dist {
            best_dist = dist;
            best_idx = idx;
        }
    }
    (best_dist, best_idx)
}

fn linear_search_f64(points: &[[f64; 2]], query: &[f64; 2]) -> (f64, usize) {
    let mut best_dist = f64::MAX;
    let mut best_idx = 0usize;
    for (idx, point) in points.iter().enumerate() {
        let dist = kiddo::SquaredEuclidean::dist(query, point);
        if dist < best_dist {
            best_dist = dist;
            best_idx = idx;
        }
    }
    (best_dist, best_idx)
}

#[test]
#[cfg(feature = "simd")]
fn regression_donnelly_simd_block4_f32_nearest_one_matches_scalar_and_linear() {
    let points = build_points_f32_k2();
    let query = build_query_f32_k2();
    let expected = linear_search(&points, &query);

    let tree_scalar: KdTree<f32, usize, Donnelly<4, 64, 4, 2>, FlatVec<f32, usize, 2, 16>, 2, 16> =
        KdTree::new_from_slice(&points);
    let tree_simd: KdTree<
        f32,
        usize,
        DonnellyMarkerSimd<Block4, 64, 4, 2>,
        FlatVec<f32, usize, 2, 16>,
        2,
        16,
    > = KdTree::new_from_slice(&points);

    let scalar_result = tree_scalar.nearest_one::<SquaredEuclidean<f32>>(&query);
    let simd_result = tree_simd.nearest_one::<SquaredEuclidean<f32>>(&query);

    assert_eq!(scalar_result.0, expected.0);
    assert_eq!(scalar_result.1, expected.1);
    assert_eq!(simd_result.0, expected.0);
    assert_eq!(simd_result.1, expected.1);
}

#[test]
#[cfg(feature = "simd")]
fn regression_donnelly_simd_block4_f32_approx_self_lookup_hits_zero_distance() {
    let points = build_points_f32_k2();

    let tree_scalar: KdTree<f32, usize, Donnelly<4, 64, 4, 2>, FlatVec<f32, usize, 2, 16>, 2, 16> =
        KdTree::new_from_slice(&points);
    let tree_simd: KdTree<
        f32,
        usize,
        DonnellyMarkerSimd<Block4, 64, 4, 2>,
        FlatVec<f32, usize, 2, 16>,
        2,
        16,
    > = KdTree::new_from_slice(&points);

    for point in points.iter() {
        let scalar_result = tree_scalar.approx_nearest_one::<SquaredEuclidean<f32>>(point);
        let simd_result = tree_simd.approx_nearest_one::<SquaredEuclidean<f32>>(point);

        assert_eq!(scalar_result.0, 0.0);
        assert_eq!(simd_result.0, 0.0);
    }
}

#[test]
#[cfg(feature = "simd")]
fn control_donnelly_simd_block3_f64_nearest_one_matches_scalar_and_linear() {
    let mut rng = StdRng::seed_from_u64(CONTENT_SEED);
    let points: Vec<[f64; 2]> = (0..POINT_COUNT)
        .map(|_| [rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0)])
        .collect();

    let mut rng = StdRng::seed_from_u64(QUERY_SEED);
    let query = [rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0)];

    let expected = linear_search_f64(&points, &query);

    let tree_scalar: KdTree<f64, usize, Donnelly<3, 64, 8, 2>, FlatVec<f64, usize, 2, 16>, 2, 16> =
        KdTree::new_from_slice(&points);
    let tree_simd: KdTree<
        f64,
        usize,
        DonnellyMarkerSimd<Block3, 64, 8, 2>,
        FlatVec<f64, usize, 2, 16>,
        2,
        16,
    > = KdTree::new_from_slice(&points);

    let scalar_result = tree_scalar.nearest_one::<SquaredEuclidean<f64>>(&query);
    let simd_result = tree_simd.nearest_one::<SquaredEuclidean<f64>>(&query);

    assert_eq!(scalar_result.0, expected.0);
    assert_eq!(scalar_result.1, expected.1);
    assert_eq!(simd_result.0, expected.0);
    assert_eq!(simd_result.1, expected.1);
}
