use az::{Az, Cast};
use sorted_vec::SortedVec;
use std::collections::BinaryHeap;
use std::ops::Rem;

use crate::float::kdtree::{Axis, KdTree};
use crate::float::result_collection::ResultCollection;
use crate::nearest_neighbour::NearestNeighbour;
use crate::traits::DistanceMetric;
use crate::traits::{is_stem_index, Content, Index};

use crate::generate_nearest_n_within_unsorted;

const MAX_VEC_RESULT_SIZE: usize = 20;

macro_rules! generate_float_nearest_n_within {
    ($doctest_build_tree:tt) => {
        generate_nearest_n_within_unsorted!((
            "Finds up to n elements within `dist` of `query`, using the specified
distance metric function.

Results are returned in as a ResultCollection, which can return a sorted or unsorted Vec.

# Examples

```rust
use std::num::NonZero;
use kiddo::KdTree;
use kiddo::SquaredEuclidean;
",
            $doctest_build_tree,
            "
let max_qty = NonZero::new(1).unwrap();
let within = tree.nearest_n_within::<SquaredEuclidean>(&[1.0, 2.0, 5.0], 10f64, max_qty, true);

assert_eq!(within.len(), 1);
```"
        ));
    };
}

impl<A: Axis, T: Content, const K: usize, const B: usize, IDX: Index<T = IDX>>
    KdTree<A, T, K, B, IDX>
where
    usize: Cast<IDX>,
{
    generate_float_nearest_n_within!(
        "
let mut tree: KdTree<f64, 3> = KdTree::new();
tree.add(&[1.0, 2.0, 5.0], 100);
tree.add(&[2.0, 3.0, 6.0], 101);"
    );
}

#[cfg(feature = "rkyv")]
use crate::float::kdtree::ArchivedKdTree;
#[cfg(feature = "rkyv")]
impl<
        A: Axis + rkyv::Archive<Archived = A>,
        T: Content + rkyv::Archive<Archived = T>,
        const K: usize,
        const B: usize,
        IDX: Index<T = IDX> + rkyv::Archive<Archived = IDX>,
    > ArchivedKdTree<A, T, K, B, IDX>
where
    usize: Cast<IDX>,
{
    generate_float_nearest_n_within!(
        "use std::fs::File;
use memmap::MmapOptions;

let mmap = unsafe { MmapOptions::new().map(&File::open(\"./examples/float-doctest-tree.rkyv\").unwrap()).unwrap() };
let tree = unsafe { rkyv::archived_root::<KdTree<f64, 3>>(&mmap) };"
    );
}

#[cfg(test)]
mod tests {
    use crate::float::distance::SquaredEuclidean;
    use crate::float::kdtree::{Axis, KdTree};
    use crate::traits::DistanceMetric;
    use rand::Rng;
    use std::cmp::Ordering;
    use std::num::NonZero;

    type AX = f32;

    #[test]
    fn can_query_nearest_n_items_within_radius() {
        let mut tree: KdTree<AX, u32, 4, 4, u32> = KdTree::new();

        let content_to_add: [([AX; 4], u32); 16] = [
            ([0.9f32, 0.0f32, 0.9f32, 0.0f32], 9),
            ([0.4f32, 0.5f32, 0.4f32, 0.5f32], 4),
            ([0.12f32, 0.3f32, 0.12f32, 0.3f32], 12),
            ([0.7f32, 0.2f32, 0.7f32, 0.2f32], 7),
            ([0.13f32, 0.4f32, 0.13f32, 0.4f32], 13),
            ([0.6f32, 0.3f32, 0.6f32, 0.3f32], 6),
            ([0.2f32, 0.7f32, 0.2f32, 0.7f32], 2),
            ([0.14f32, 0.5f32, 0.14f32, 0.5f32], 14),
            ([0.3f32, 0.6f32, 0.3f32, 0.6f32], 3),
            ([0.10f32, 0.1f32, 0.10f32, 0.1f32], 10),
            ([0.16f32, 0.7f32, 0.16f32, 0.7f32], 16),
            ([0.1f32, 0.8f32, 0.1f32, 0.8f32], 1),
            ([0.15f32, 0.6f32, 0.15f32, 0.6f32], 15),
            ([0.5f32, 0.4f32, 0.5f32, 0.4f32], 5),
            ([0.8f32, 0.1f32, 0.8f32, 0.1f32], 8),
            ([0.11f32, 0.2f32, 0.11f32, 0.2f32], 11),
        ];

        for (point, item) in content_to_add {
            tree.add(&point, item);
        }

        assert_eq!(tree.size(), 16);

        let query_point = [0.78f32, 0.55f32, 0.78f32, 0.55f32];

        let radius = 0.2;
        let max_qty = NonZero::new(3).unwrap();

        let expected = linear_search(&content_to_add, &query_point, radius)
            .into_iter()
            .take(max_qty.get())
            .collect::<Vec<_>>();

        let result: Vec<_> = tree
            .nearest_n_within::<SquaredEuclidean>(&query_point, radius, max_qty, true)
            .into_iter()
            .map(|n| (n.distance, n.item))
            .collect();
        assert_eq!(result, expected);

        let mut rng = rand::thread_rng();
        for _i in 0..1000 {
            let query_point = [
                rng.gen_range(0f32..1f32),
                rng.gen_range(0f32..1f32),
                rng.gen_range(0f32..1f32),
                rng.gen_range(0f32..1f32),
            ];
            let radius = 0.2;
            let max_qty = NonZero::new(3).unwrap();

            let expected = linear_search(&content_to_add, &query_point, radius)
                .into_iter()
                .take(max_qty.get())
                .collect::<Vec<_>>();

            let result: Vec<_> = tree
                .nearest_n_within::<SquaredEuclidean>(&query_point, radius, max_qty, true)
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();

            assert_eq!(result, expected);
        }
    }

    #[test]
    fn can_query_nearest_n_items_within_radius_unsorted() {
        let mut tree: KdTree<AX, u32, 4, 4, u32> = KdTree::new();

        let content_to_add: [([AX; 4], u32); 16] = [
            ([0.9f32, 0.0f32, 0.9f32, 0.0f32], 9),
            ([0.4f32, 0.5f32, 0.4f32, 0.5f32], 4),
            ([0.12f32, 0.3f32, 0.12f32, 0.3f32], 12),
            ([0.7f32, 0.2f32, 0.7f32, 0.2f32], 7),
            ([0.13f32, 0.4f32, 0.13f32, 0.4f32], 13),
            ([0.6f32, 0.3f32, 0.6f32, 0.3f32], 6),
            ([0.2f32, 0.7f32, 0.2f32, 0.7f32], 2),
            ([0.14f32, 0.5f32, 0.14f32, 0.5f32], 14),
            ([0.3f32, 0.6f32, 0.3f32, 0.6f32], 3),
            ([0.10f32, 0.1f32, 0.10f32, 0.1f32], 10),
            ([0.16f32, 0.7f32, 0.16f32, 0.7f32], 16),
            ([0.1f32, 0.8f32, 0.1f32, 0.8f32], 1),
            ([0.15f32, 0.6f32, 0.15f32, 0.6f32], 15),
            ([0.5f32, 0.4f32, 0.5f32, 0.4f32], 5),
            ([0.8f32, 0.1f32, 0.8f32, 0.1f32], 8),
            ([0.11f32, 0.2f32, 0.11f32, 0.2f32], 11),
        ];

        for (point, item) in content_to_add {
            tree.add(&point, item);
        }

        assert_eq!(tree.size(), 16);

        let query_point = [0.78f32, 0.55f32, 0.78f32, 0.55f32];

        let radius = 0.2;
        let max_qty = NonZero::new(3).unwrap();

        let expected = linear_search(&content_to_add, &query_point, radius)
            .into_iter()
            .take(max_qty.get())
            .collect::<Vec<_>>();

        let mut result: Vec<_> = tree
            .nearest_n_within::<SquaredEuclidean>(&query_point, radius, max_qty, false)
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
            let radius = 0.2;
            let max_qty = NonZero::new(3).unwrap();

            let expected = linear_search(&content_to_add, &query_point, radius)
                .into_iter()
                .take(max_qty.get())
                .collect::<Vec<_>>();

            let mut result: Vec<_> = tree
                .nearest_n_within::<SquaredEuclidean>(&query_point, radius, max_qty, false)
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();
            stabilize_sort(&mut result);

            assert_eq!(result, expected);
        }
    }

    #[test]
    fn can_query_nearest_n_items_unsorted_max_qty() {
        let mut tree: KdTree<AX, u32, 4, 4, u32> = KdTree::new();

        let content_to_add: [([AX; 4], u32); 16] = [
            ([0.9f32, 0.0f32, 0.9f32, 0.0f32], 9),
            ([0.4f32, 0.5f32, 0.4f32, 0.5f32], 4),
            ([0.12f32, 0.3f32, 0.12f32, 0.3f32], 12),
            ([0.7f32, 0.2f32, 0.7f32, 0.2f32], 7),
            ([0.13f32, 0.4f32, 0.13f32, 0.4f32], 13),
            ([0.6f32, 0.3f32, 0.6f32, 0.3f32], 6),
            ([0.2f32, 0.7f32, 0.2f32, 0.7f32], 2),
            ([0.14f32, 0.5f32, 0.14f32, 0.5f32], 14),
            ([0.3f32, 0.6f32, 0.3f32, 0.6f32], 3),
            ([0.10f32, 0.1f32, 0.10f32, 0.1f32], 10),
            ([0.16f32, 0.7f32, 0.16f32, 0.7f32], 16),
            ([0.1f32, 0.8f32, 0.1f32, 0.8f32], 1),
            ([0.15f32, 0.6f32, 0.15f32, 0.6f32], 15),
            ([0.5f32, 0.4f32, 0.5f32, 0.4f32], 5),
            ([0.8f32, 0.1f32, 0.8f32, 0.1f32], 8),
            ([0.11f32, 0.2f32, 0.11f32, 0.2f32], 11),
        ];

        for (point, item) in content_to_add {
            tree.add(&point, item);
        }

        let query_point = [0.78f32, 0.55f32, 0.78f32, 0.55f32];

        let radius = 100.0;
        let max_qty = NonZero::new(1).unwrap();

        let result_unsorted: Vec<_> = tree
            .nearest_n_within::<SquaredEuclidean>(&query_point, radius, max_qty, false)
            .into_iter()
            .map(|n| (n.distance, n.item))
            .collect();

        let result_sorted: Vec<_> = tree
            .nearest_n_within::<SquaredEuclidean>(&query_point, radius, max_qty, true)
            .into_iter()
            .map(|n| (n.distance, n.item))
            .collect();

        assert_eq!(result_unsorted.len(), max_qty.get());
        assert_eq!(result_sorted.len(), max_qty.get());
    }

    #[test]
    fn can_query_nearest_n_items_within_radius_unsorted_large_scale() {
        const TREE_SIZE: usize = 100_000;
        const NUM_QUERIES: usize = 100;
        const RADIUS: f32 = 0.2;

        let max_qty = NonZero::new(3).unwrap();

        let content_to_add: Vec<([f32; 4], u32)> = (0..TREE_SIZE)
            .map(|_| rand::random::<([f32; 4], u32)>())
            .collect();

        let mut tree: KdTree<AX, u32, 4, 32, u32> = KdTree::with_capacity(TREE_SIZE);
        content_to_add
            .iter()
            .for_each(|(point, content)| tree.add(point, *content));
        assert_eq!(tree.size(), TREE_SIZE as u32);

        let query_points: Vec<[f32; 4]> = (0..NUM_QUERIES)
            .map(|_| rand::random::<[f32; 4]>())
            .collect();

        for query_point in query_points {
            let expected = linear_search(&content_to_add, &query_point, RADIUS)
                .into_iter()
                .take(max_qty.get())
                .collect::<Vec<_>>();

            let mut result: Vec<_> = tree
                .nearest_n_within::<SquaredEuclidean>(&query_point, RADIUS, max_qty, true)
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();

            stabilize_sort(&mut result);
            assert_eq!(result, expected);
        }
    }

    fn linear_search<A: Axis, const K: usize>(
        content: &[([A; K], u32)],
        query_point: &[A; K],
        radius: A,
    ) -> Vec<(A, u32)> {
        let mut matching_items = vec![];

        for &(p, item) in content {
            let dist = SquaredEuclidean::dist(query_point, &p);
            if dist < radius {
                matching_items.push((dist, item));
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
