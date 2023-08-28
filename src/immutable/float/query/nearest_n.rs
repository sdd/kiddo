use crate::distance_metric::DistanceMetric;
use crate::float::kdtree::Axis;
use crate::immutable::float::kdtree::ImmutableKdTree;
use crate::nearest_neighbour::NearestNeighbour;
use crate::types::Content;
use std::collections::BinaryHeap;
use std::ops::Rem;

use crate::generate_immutable_nearest_n;

macro_rules! generate_immutable_float_nearest_n {
    ($doctest_build_tree:tt) => {
        generate_immutable_nearest_n!((
            "Finds the nearest `qty` elements to `query`, according the specified
distance metric function.
# Examples

```rust
    use kiddo::immutable::float::kdtree::ImmutableKdTree;
    use kiddo::float::distance::SquaredEuclidean;

    ",
            $doctest_build_tree,
            "

    let nearest: Vec<_> = tree.nearest_n::<SquaredEuclidean>(&[1.0, 2.0, 5.1], 1);

    assert_eq!(nearest.len(), 1);
    assert!((nearest[0].distance - 0.01f64).abs() < f64::EPSILON);
    assert_eq!(nearest[0].item, 0);
```"
        ));
    };
}

impl<A: Axis, T: Content, const K: usize, const B: usize> ImmutableKdTree<A, T, K, B> {
    generate_immutable_float_nearest_n!(
        "let content: Vec<[f64; 3]> = vec!(
            [1.0, 2.0, 5.0],
            [2.0, 3.0, 6.0]
        );

        let tree: ImmutableKdTree<f64, u32, 3, 32> = ImmutableKdTree::new_from_slice(&content);"
    );
}

#[cfg(feature = "rkyv")]
use crate::immutable::float::kdtree::ArchivedImmutableKdTree;
#[cfg(feature = "rkyv")]
impl<
        A: Axis + rkyv::Archive<Archived = A>,
        T: Content + rkyv::Archive<Archived = T>,
        const K: usize,
        const B: usize,
    > ArchivedImmutableKdTree<A, T, K, B>
{
    generate_immutable_float_nearest_n!(
        "use std::fs::File;
    use memmap::MmapOptions;

    let mmap = unsafe { MmapOptions::new().map(&File::open(\"./examples/immutable-doctest-tree.rkyv\").unwrap()).unwrap() };
    let tree = unsafe { rkyv::archived_root::<ImmutableKdTree<f64, u32, 3, 32>>(&mmap) };"
    );
}

#[cfg(test)]
mod tests {
    use crate::distance_metric::DistanceMetric;
    use crate::float::distance::SquaredEuclidean;
    use crate::float::kdtree::Axis;
    use crate::immutable::float::kdtree::ImmutableKdTree;
    use rand::Rng;

    type AX = f32;

    #[test]
    fn can_query_nearest_n_item() {
        let content_to_add: [[AX; 4]; 16] = [
            [0.9f32, 0.0f32, 0.9f32, 0.0f32],
            [0.4f32, 0.5f32, 0.4f32, 0.51f32],
            [0.12f32, 0.3f32, 0.12f32, 0.3f32],
            [0.7f32, 0.2f32, 0.7f32, 0.22f32],
            [0.13f32, 0.4f32, 0.13f32, 0.4f32],
            [0.6f32, 0.3f32, 0.6f32, 0.33f32],
            [0.2f32, 0.7f32, 0.2f32, 0.7f32],
            [0.14f32, 0.5f32, 0.14f32, 0.5f32],
            [0.3f32, 0.6f32, 0.3f32, 0.6f32],
            [0.10f32, 0.1f32, 0.10f32, 0.1f32],
            [0.16f32, 0.7f32, 0.16f32, 0.7f32],
            [0.1f32, 0.8f32, 0.1f32, 0.8f32],
            [0.15f32, 0.6f32, 0.15f32, 0.6f32],
            [0.5f32, 0.4f32, 0.5f32, 0.44f32],
            [0.8f32, 0.1f32, 0.8f32, 0.15f32],
            [0.11f32, 0.2f32, 0.11f32, 0.2f32],
        ];

        let tree: ImmutableKdTree<AX, u32, 4, 4> = ImmutableKdTree::new_from_slice(&content_to_add);

        assert_eq!(tree.size(), 16);

        let query_point = [0.78f32, 0.55f32, 0.78f32, 0.55f32];

        let expected = vec![(0.17569996, 5), (0.19139998, 13), (0.24420004, 3)];

        let result: Vec<_> = tree
            .nearest_n::<SquaredEuclidean>(&query_point, 3)
            .into_iter()
            .map(|n| (n.distance, n.item))
            .collect();
        assert_eq!(result, expected);

        let qty = 10;
        let mut rng = rand::thread_rng();
        for _i in 0..1000 {
            let query_point = [
                rng.gen_range(0f32..1f32),
                rng.gen_range(0f32..1f32),
                rng.gen_range(0f32..1f32),
                rng.gen_range(0f32..1f32),
            ];
            let expected = linear_search(&content_to_add, qty, &query_point);

            let result: Vec<_> = tree
                .nearest_n::<SquaredEuclidean>(&query_point, qty)
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();

            let result_dists: Vec<_> = result.iter().map(|(d, _)| d).collect();
            let expected_dists: Vec<_> = expected.iter().map(|(d, _)| d).collect();

            assert_eq!(result_dists, expected_dists);
        }
    }

    #[test]
    fn can_query_nearest_10_items_large_scale() {
        const TREE_SIZE: usize = 100_000;
        const NUM_QUERIES: usize = 100;
        const N: usize = 10;

        let content_to_add: Vec<[f32; 4]> =
            (0..TREE_SIZE).map(|_| rand::random::<[f32; 4]>()).collect();

        let tree: ImmutableKdTree<AX, u32, 4, 32> =
            ImmutableKdTree::new_from_slice(&content_to_add);

        assert_eq!(tree.size(), TREE_SIZE);

        let query_points: Vec<[f32; 4]> = (0..NUM_QUERIES)
            .map(|_| rand::random::<[f32; 4]>())
            .collect();

        for query_point in query_points {
            let expected = linear_search(&content_to_add, N, &query_point);

            let result: Vec<_> = tree
                .nearest_n::<SquaredEuclidean>(&query_point, N)
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();

            let result_dists: Vec<_> = result.iter().map(|(d, _)| d).collect();
            let expected_dists: Vec<_> = expected.iter().map(|(d, _)| d).collect();

            assert_eq!(result_dists, expected_dists);
        }
    }

    fn linear_search<A: Axis, const K: usize>(
        content: &[[A; K]],
        qty: usize,
        query_point: &[A; K],
    ) -> Vec<(A, usize)> {
        let mut results = vec![];

        for (idx, p) in content.iter().enumerate() {
            let dist = SquaredEuclidean::dist(query_point, &p);
            if results.len() < qty {
                results.push((dist, idx));
                results.sort_by(|(a_dist, _), (b_dist, _)| a_dist.partial_cmp(&b_dist).unwrap());
            } else if dist < results[qty - 1].0 {
                results[qty - 1] = (dist, idx);
                results.sort_by(|(a_dist, _), (b_dist, _)| a_dist.partial_cmp(&b_dist).unwrap());
            }
        }

        results
    }
}
