use az::Cast;
use std::ops::Rem;

use crate::distance_metric::DistanceMetric;
use crate::float::kdtree::Axis;
use crate::float_leaf_slice::leaf_slice::LeafSliceFloat;
use crate::generate_immutable_nearest_one;
use crate::immutable::float::kdtree::ImmutableKdTree;
use crate::nearest_neighbour::NearestNeighbour;
use crate::types::Content;

macro_rules! generate_immutable_float_nearest_one {
    ($doctest_build_tree:tt) => {
        generate_immutable_nearest_one!((
            "Queries the tree to find the nearest item to the `query` point.

Faster than querying for nearest_n(point, 1, ...) due
to not needing to allocate memory or maintain sorted results.

# Examples

```rust
    use kiddo::ImmutableKdTree;
    use kiddo::SquaredEuclidean;

    ",
            $doctest_build_tree,
            "

    let nearest = tree.nearest_one::<SquaredEuclidean>(&[1.0, 2.0, 5.1]);

    assert!((nearest.distance - 0.01f64).abs() < f64::EPSILON);
    assert_eq!(nearest.item, 0);
```"
        ));
    };
}

impl<A, T, const K: usize, const B: usize> ImmutableKdTree<A, T, K, B>
where
    A: Axis + LeafSliceFloat<T, K>,
    T: Content,
    usize: Cast<T>,
{
    generate_immutable_float_nearest_one!(
        "let content: Vec<[f64; 3]> = vec!(
            [1.0, 2.0, 5.0],
            [2.0, 3.0, 6.0]
        );

        let tree: ImmutableKdTree<f64, 3> = ImmutableKdTree::new_from_slice(&content);"
    );
}

#[cfg(feature = "rkyv")]
use crate::immutable::float::kdtree::ArchivedImmutableKdTree;
#[cfg(feature = "rkyv")]
impl<A, T, const K: usize, const B: usize> ArchivedImmutableKdTree<A, T, K, B>
where
    A: Axis + LeafSliceFloat<T, K> + rkyv::Archive<Archived = A>,
    T: Content + rkyv::Archive<Archived = T>,
    usize: Cast<T>,
{
    generate_immutable_float_nearest_one!(
        "use std::fs::File;
    use memmap::MmapOptions;

    let mmap = unsafe { MmapOptions::new().map(&File::open(\"./examples/immutable-doctest-tree.rkyv\").unwrap()).unwrap() };
    let tree = unsafe { rkyv::archived_root::<ImmutableKdTree<f64, 3>>(&mmap) };"
    );
}

#[cfg(test)]
mod tests {
    use crate::distance_metric::DistanceMetric;
    use crate::float::distance::SquaredEuclidean;
    use crate::float::kdtree::Axis;
    use crate::immutable::float::kdtree::ImmutableKdTree;
    use crate::nearest_neighbour::NearestNeighbour;
    use rand::{Rng, SeedableRng};

    #[test]
    fn can_query_nearest_one_item_f64() {
        let content_to_add: [[f64; 4]; 16] = [
            [0.9f64, 0.0f64, 0.9f64, 0.0f64],
            [0.4f64, 0.5f64, 0.4f64, 0.51f64],
            [0.12f64, 0.3f64, 0.12f64, 0.3f64],
            [0.7f64, 0.2f64, 0.7f64, 0.22f64],
            [0.13f64, 0.4f64, 0.13f64, 0.4f64],
            [0.6f64, 0.3f64, 0.6f64, 0.33f64],
            [0.2f64, 0.7f64, 0.2f64, 0.7f64],
            [0.14f64, 0.5f64, 0.14f64, 0.5f64],
            [0.3f64, 0.6f64, 0.3f64, 0.6f64],
            [0.10f64, 0.1f64, 0.10f64, 0.1f64],
            [0.16f64, 0.7f64, 0.16f64, 0.7f64],
            [0.1f64, 0.8f64, 0.1f64, 0.8f64],
            [0.15f64, 0.6f64, 0.15f64, 0.6f64],
            [0.5f64, 0.4f64, 0.5f64, 0.44f64],
            [0.8f64, 0.1f64, 0.8f64, 0.15f64],
            [0.11f64, 0.2f64, 0.11f64, 0.2f64],
        ];

        let tree: ImmutableKdTree<f64, u32, 4, 4> =
            ImmutableKdTree::new_from_slice(&content_to_add);

        assert_eq!(tree.size(), 16);

        println!("Tree: {:?}", &tree);

        let query_point = [0.78f64, 0.55f64, 0.78f64, 0.55f64];

        let expected = NearestNeighbour {
            distance: 0.17570000000000008,
            item: 5,
        };

        let result = tree.nearest_one::<SquaredEuclidean>(&query_point);
        assert_eq!(result.distance, expected.distance);

        let mut rng = rand::thread_rng();
        for _i in 0..1000 {
            let query_point = [
                rng.gen_range(0f64..1f64),
                rng.gen_range(0f64..1f64),
                rng.gen_range(0f64..1f64),
                rng.gen_range(0f64..1f64),
            ];
            let expected = linear_search(&content_to_add, &query_point);

            // println!("query #{:?}: {:?}", _i, &query_point);
            let result = tree.nearest_one::<SquaredEuclidean>(&query_point);
            // println!("result: {:?}, expected: {:?}", &result, &expected);

            assert_eq!(result.distance, expected.distance);
        }
    }

    #[test]
    fn can_query_nearest_one_item_f32() {
        let content_to_add: [[f32; 4]; 16] = [
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

        let tree: ImmutableKdTree<f32, u32, 4, 4> =
            ImmutableKdTree::new_from_slice(&content_to_add);

        assert_eq!(tree.size(), 16);

        let query_point = [0.78f32, 0.55f32, 0.78f32, 0.55f32];

        let expected = NearestNeighbour {
            distance: 0.17569996,
            item: 5,
        };

        let result = tree.nearest_one::<SquaredEuclidean>(&query_point);
        assert_eq!(result.distance, expected.distance);

        let mut rng = rand::thread_rng();
        for _i in 0..1000 {
            let query_point = [
                rng.gen_range(0f32..1f32),
                rng.gen_range(0f32..1f32),
                rng.gen_range(0f32..1f32),
                rng.gen_range(0f32..1f32),
            ];
            let expected = linear_search(&content_to_add, &query_point);

            // println!("query #{:?}: {:?}", _i, &query_point);
            let result = tree.nearest_one::<SquaredEuclidean>(&query_point);

            assert_eq!(result.distance, expected.distance);
        }
    }

    #[test]
    fn can_query_nearest_one_item_large_scale_f64() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(3);

        const TREE_SIZE: usize = 100_000;
        const NUM_QUERIES: usize = 1000;

        let content_to_add: Vec<[f64; 4]> = (0..TREE_SIZE).map(|_| rng.gen::<[f64; 4]>()).collect();

        let tree: ImmutableKdTree<f64, u32, 4, 256> =
            ImmutableKdTree::new_from_slice(&content_to_add);

        assert_eq!(tree.size(), TREE_SIZE);

        let query_points: Vec<[f64; 4]> = (0..NUM_QUERIES).map(|_| rng.gen::<[f64; 4]>()).collect();

        for (_i, query_point) in query_points.iter().enumerate() {
            let expected = linear_search(&content_to_add, query_point);

            // println!("query #{:?}", _i);
            let result = tree.nearest_one::<SquaredEuclidean>(query_point);
            // println!("result: {:?} ({:?})", &result, content_to_add[result.item as usize]);
            // println!("expected: {:?} ({:?})", &expected, content_to_add[expected.item as usize]);

            assert_eq!(result.item as usize, expected.item);
            assert_eq!(result.distance, expected.distance);
        }
    }

    #[test]
    fn can_query_nearest_one_item_large_scale_f32() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(3);

        const TREE_SIZE: usize = 100_000;
        const NUM_QUERIES: usize = 1000;

        let content_to_add: Vec<[f32; 4]> = (0..TREE_SIZE).map(|_| rng.gen::<[f32; 4]>()).collect();

        let tree: ImmutableKdTree<f32, u32, 4, 256> =
            ImmutableKdTree::new_from_slice(&content_to_add);

        assert_eq!(tree.size(), TREE_SIZE);

        let query_points: Vec<[f32; 4]> = (0..NUM_QUERIES)
            .map(|_| rand::random::<[f32; 4]>())
            .collect();

        for (_i, query_point) in query_points.iter().enumerate() {
            let expected = linear_search(&content_to_add, query_point);

            let result = tree.nearest_one::<SquaredEuclidean>(query_point);

            assert_eq!(result.distance, expected.distance);
            assert_eq!(result.item as usize, expected.item);
        }
    }

    fn linear_search<A: Axis, const K: usize>(
        content: &[[A; K]],
        query_point: &[A; K],
    ) -> NearestNeighbour<A, usize> {
        let mut best_dist: A = A::infinity();
        let mut best_item: usize = usize::MAX;

        for (idx, p) in content.iter().enumerate() {
            let dist = SquaredEuclidean::dist(query_point, p);
            if dist < best_dist {
                best_item = idx;
                best_dist = dist;
            }
        }

        NearestNeighbour {
            distance: best_dist,
            item: best_item,
        }
    }
}
