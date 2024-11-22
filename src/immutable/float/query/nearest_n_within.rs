use az::Cast;
use sorted_vec::SortedVec;
use std::collections::BinaryHeap;
use std::num::NonZero;
use std::ops::Rem;

use crate::distance_metric::DistanceMetric;
use crate::float::kdtree::Axis;
use crate::float::result_collection::ResultCollection;
use crate::float_leaf_slice::leaf_slice::LeafSliceFloat;
use crate::immutable::float::kdtree::ImmutableKdTree;
use crate::nearest_neighbour::NearestNeighbour;
use crate::types::Content;

use crate::generate_immutable_nearest_n_within;

const MAX_VEC_RESULT_SIZE: usize = 20;

macro_rules! generate_immutable_float_nearest_n_within {
    ($doctest_build_tree:tt) => {
        generate_immutable_nearest_n_within!((
            "Finds up to n elements within `dist` of `query`, using the specified
distance metric function.

# Examples

```rust
    use std::num::NonZero;
    use kiddo::ImmutableKdTree;
    use kiddo::SquaredEuclidean;
    ",
            $doctest_build_tree,
            "

    let within = tree.nearest_n_within::<SquaredEuclidean>(&[1.0, 2.0, 5.0], 10f64, NonZero::new(2).unwrap(), true);

    assert_eq!(within.len(), 2);
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
    generate_immutable_float_nearest_n_within!(
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
impl<
        A: Axis + rkyv::Archive<Archived = A>,
        T: Content + rkyv::Archive<Archived = T>,
        const K: usize,
        const B: usize,
    > ArchivedImmutableKdTree<A, T, K, B>
where
    A: Axis + LeafSliceFloat<T, K>,
    T: Content,
    usize: Cast<T>,
{
    generate_immutable_float_nearest_n_within!(
        "use std::fs::File;
use memmap::MmapOptions;

let mmap = unsafe { MmapOptions::new().map(&File::open(\"./examples/immutable-dynamic-doctest-tree.rkyv\").unwrap()).unwrap() };
let tree = unsafe { rkyv::archived_root::<ImmutableKdTree<f64, 3>>(&mmap) };"
    );
}

#[cfg(test)]
mod tests {
    use crate::distance_metric::DistanceMetric;
    use crate::float::distance::SquaredEuclidean;
    use crate::float::kdtree::Axis;
    use crate::immutable::float::kdtree::ImmutableKdTree;
    use rand::Rng;
    use std::cmp::Ordering;
    use std::num::NonZero;

    type AX = f32;

    #[test]
    fn can_query_items_within_radius() {
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

        let radius = 0.2;
        let max_qty = NonZero::new(3).unwrap();

        let expected = linear_search(&content_to_add, &query_point, radius);

        let mut result: Vec<_> = tree
            .nearest_n_within::<SquaredEuclidean>(&query_point, radius, max_qty, true)
            .into_iter()
            .map(|n| (n.distance, n.item))
            .collect();
        stabilize_sort(&mut result);
        assert_eq!(result, expected);

        let mut rng = rand::thread_rng();
        for _i in 0..1000 {
            let query_point = [
                rng.gen_range(0f32..1f32),
                rng.gen_range(0f32..1f32),
                rng.gen_range(0f32..1f32),
                rng.gen_range(0f32..1f32),
            ];
            let radius: f32 = 2.0;
            let max_qty = NonZero::new(3).unwrap();

            let expected = linear_search(&content_to_add, &query_point, radius)
                .into_iter()
                .take(max_qty.into())
                .collect::<Vec<_>>();

            let mut result: Vec<_> = tree
                .nearest_n_within::<SquaredEuclidean>(&query_point, radius, max_qty, true)
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();
            stabilize_sort(&mut result);

            assert_eq!(result, expected);
        }
    }

    #[test]
    fn can_query_items_within_radius_large_scale() {
        const TREE_SIZE: usize = 100_000;
        const NUM_QUERIES: usize = 100;
        const RADIUS: f32 = 0.2;
        const MAX_QTY: NonZero<usize> = NonZero::new(3).unwrap();

        let content_to_add: Vec<[f32; 4]> =
            (0..TREE_SIZE).map(|_| rand::random::<[f32; 4]>()).collect();

        let tree: ImmutableKdTree<AX, u32, 4, 32> =
            ImmutableKdTree::new_from_slice(&content_to_add);
        assert_eq!(tree.size(), TREE_SIZE);

        let query_points: Vec<[f32; 4]> = (0..NUM_QUERIES)
            .map(|_| rand::random::<[f32; 4]>())
            .collect();

        for query_point in query_points {
            let expected = linear_search(&content_to_add, &query_point, RADIUS)
                .into_iter()
                .take(MAX_QTY.into())
                .collect::<Vec<_>>();

            let mut result: Vec<_> = tree
                .nearest_n_within::<SquaredEuclidean>(&query_point, RADIUS, MAX_QTY, true)
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();

            stabilize_sort(&mut result);

            assert_eq!(result, expected);
        }
    }

    fn linear_search<A: Axis, const K: usize>(
        content: &[[A; K]],
        query_point: &[A; K],
        radius: A,
    ) -> Vec<(A, u32)> {
        let mut matching_items = vec![];

        for (idx, p) in content.iter().enumerate() {
            let dist = SquaredEuclidean::dist(query_point, p);
            if dist < radius {
                matching_items.push((dist, idx as u32));
            }
        }

        stabilize_sort(&mut matching_items);

        matching_items
    }

    fn stabilize_sort<A: Axis>(matching_items: &mut [(A, u32)]) {
        matching_items.sort_unstable_by(|a, b| {
            let dist_cmp = a.0.partial_cmp(&b.0).unwrap();
            if dist_cmp == Ordering::Equal {
                a.1.cmp(&b.1)
            } else {
                dist_cmp
            }
        });
    }
}
