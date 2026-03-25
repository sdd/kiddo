use az::{Az, Cast};
use sorted_vec::SortedVec;
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::ops::Rem;

use crate::float::kdtree::{Axis, KdTree};
use crate::float::result_collection::ResultCollection;
use crate::nearest_neighbour::NearestNeighbour;
use crate::rkyv_utils::transform;
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

    /// Finds up to `max_items` elements within `dist` of `query` with periodic boundary conditions.
    ///
    /// `box_size` gives the periodic box length for each axis. Query points are expected
    /// to be wrapped into the same principal cell as the points stored in the tree.
    ///
    /// This first implementation checks all `3^K` wrapped query images and merges duplicate
    /// items that can arise from multiple images.
    #[inline]
    pub fn nearest_n_within_periodic<D>(
        &self,
        query: &[A; K],
        dist: A,
        max_items: std::num::NonZero<usize>,
        sorted: bool,
        box_size: &[A; K],
    ) -> Vec<NearestNeighbour<A, T>>
    where
        D: DistanceMetric<A, K>,
        T: std::hash::Hash + Eq,
    {
        box_size.iter().for_each(|axis_len| {
            assert!(
                *axis_len > A::zero(),
                "periodic box sizes must be strictly positive"
            );
        });

        let mut wrapped_query = *query;
        let mut best_by_item: HashMap<T, A> = HashMap::new();

        self.nearest_n_within_periodic_recurse::<D>(
            query,
            dist,
            max_items,
            box_size,
            0,
            &mut wrapped_query,
            &mut best_by_item,
        );

        let mut results: Vec<_> = best_by_item
            .into_iter()
            .map(|(item, distance)| NearestNeighbour { distance, item })
            .collect();

        if sorted {
            results.sort();
        }
        results.truncate(max_items.get());
        results
    }

    fn nearest_n_within_periodic_recurse<D>(
        &self,
        query: &[A; K],
        dist: A,
        max_items: std::num::NonZero<usize>,
        box_size: &[A; K],
        axis: usize,
        wrapped_query: &mut [A; K],
        best_by_item: &mut HashMap<T, A>,
    ) where
        D: DistanceMetric<A, K>,
        T: std::hash::Hash + Eq,
    {
        if axis == K {
            for candidate in self.nearest_n_within::<D>(wrapped_query, dist, max_items, false) {
                best_by_item
                    .entry(candidate.item)
                    .and_modify(|best_distance| {
                        if candidate.distance < *best_distance {
                            *best_distance = candidate.distance;
                        }
                    })
                    .or_insert(candidate.distance);
            }
            return;
        }

        let original = query[axis];
        let axis_len = box_size[axis];

        wrapped_query[axis] = original - axis_len;
        self.nearest_n_within_periodic_recurse::<D>(
            query,
            dist,
            max_items,
            box_size,
            axis + 1,
            wrapped_query,
            best_by_item,
        );

        wrapped_query[axis] = original;
        self.nearest_n_within_periodic_recurse::<D>(
            query,
            dist,
            max_items,
            box_size,
            axis + 1,
            wrapped_query,
            best_by_item,
        );

        wrapped_query[axis] = original + axis_len;
        self.nearest_n_within_periodic_recurse::<D>(
            query,
            dist,
            max_items,
            box_size,
            axis + 1,
            wrapped_query,
            best_by_item,
        );

        wrapped_query[axis] = original;
    }
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

let mmap = unsafe { MmapOptions::new().map(&File::open(\"./examples/float-doctest-tree.rkyv\").expect(\"./examples/float-doctest-tree.rkyv missing\")).unwrap() };
let tree = unsafe { rkyv::archived_root::<KdTree<f64, 3>>(&mmap) };"
    );
}

#[cfg(feature = "rkyv_08")]
use crate::float::kdtree::ArchivedR8KdTree;
#[cfg(feature = "rkyv_08")]
impl<
        A: Axis + rkyv_08::Archive,
        T: Content + rkyv_08::Archive,
        const K: usize,
        const B: usize,
        IDX: Index<T = IDX>,
    > ArchivedR8KdTree<A, T, K, B, IDX>
where
    usize: Cast<IDX>,
    IDX: rkyv_08::Archive,
{
    generate_float_nearest_n_within!(
        "use std::fs::File;
    use memmap::MmapOptions;
    use kiddo::float::kdtree::ArchivedR8KdTree;

    let mmap = unsafe { MmapOptions::new().map(&File::open(\"./examples/float-doctest-tree-rkyv_08.rkyv\").expect(\"./examples/float-doctest-tree-rkyv_08.rkyv missing\")).unwrap() };
    let tree = unsafe { rkyv_08::access_unchecked::<ArchivedR8KdTree<f64, u64, 3, 32, u32>>(&mmap) };"
    );
}

#[cfg(test)]
mod tests {
    use crate::float::distance::SquaredEuclidean;
    use crate::float::kdtree::{Axis, KdTree};
    use crate::nearest_neighbour::NearestNeighbour;
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

        let mut rng = rand::rng();
        for _i in 0..1000 {
            let query_point = [
                rng.random_range(0f32..1f32),
                rng.random_range(0f32..1f32),
                rng.random_range(0f32..1f32),
                rng.random_range(0f32..1f32),
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

        let mut rng = rand::rng();
        for _i in 0..1000 {
            let query_point = [
                rng.random_range(0f32..1f32),
                rng.random_range(0f32..1f32),
                rng.random_range(0f32..1f32),
                rng.random_range(0f32..1f32),
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
        assert_eq!(tree.size(), TREE_SIZE);

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

    #[test]
    fn can_query_nearest_n_items_within_periodic_boundaries() {
        let mut tree: KdTree<f64, u32, 2, 8, u32> = KdTree::new();
        let content_to_add = [
            ([0.95f64, 0.50f64], 1),
            ([0.92f64, 0.55f64], 2),
            ([0.40f64, 0.50f64], 3),
            ([0.10f64, 0.10f64], 4),
        ];

        for (point, item) in content_to_add {
            tree.add(&point, item);
        }

        let query_point = [0.05f64, 0.50f64];
        let box_size = [1.0f64, 1.0f64];
        let radius = 0.03f64;
        let max_qty = NonZero::new(2).unwrap();

        let result = tree.nearest_n_within_periodic::<SquaredEuclidean>(
            &query_point,
            radius,
            max_qty,
            true,
            &box_size,
        );

        assert_eq!(result.len(), 2);
        assert!((result[0].distance - 0.01f64).abs() < f64::EPSILON);
        assert_eq!(result[0].item, 1);
        assert!((result[1].distance - 0.0194f64).abs() < f64::EPSILON);
        assert_eq!(result[1].item, 2);
    }

    #[test]
    fn can_query_nearest_n_items_within_periodic_boundaries_unsorted() {
        let mut tree: KdTree<f64, u32, 2, 8, u32> = KdTree::new();
        let content_to_add = [
            ([0.95f64, 0.50f64], 1),
            ([0.92f64, 0.55f64], 2),
            ([0.40f64, 0.50f64], 3),
            ([0.10f64, 0.10f64], 4),
        ];

        for (point, item) in content_to_add {
            tree.add(&point, item);
        }

        let query_point = [0.05f64, 0.50f64];
        let box_size = [1.0f64, 1.0f64];
        let radius = 0.03f64;
        let max_qty = NonZero::new(2).unwrap();

        let mut result = tree.nearest_n_within_periodic::<SquaredEuclidean>(
            &query_point,
            radius,
            max_qty,
            false,
            &box_size,
        );
        stabilize_neighbours(&mut result);

        assert_eq!(result.len(), 2);
        assert!((result[0].distance - 0.01f64).abs() < f64::EPSILON);
        assert_eq!(result[0].item, 1);
        assert!((result[1].distance - 0.0194f64).abs() < f64::EPSILON);
        assert_eq!(result[1].item, 2);
    }

    #[test]
    fn can_query_nearest_n_items_within_periodic_boundaries_large_scale() {
        const TREE_SIZE: usize = 10_000;
        const NUM_QUERIES: usize = 200;
        const RADIUS: f32 = 0.05;

        let max_qty = NonZero::new(5).unwrap();
        let content_to_add: Vec<([f32; 3], u32)> = (0..TREE_SIZE)
            .map(|_| rand::random::<([f32; 3], u32)>())
            .collect();

        let mut tree: KdTree<f32, u32, 3, 32, u32> = KdTree::with_capacity(TREE_SIZE);
        content_to_add
            .iter()
            .for_each(|(point, content)| tree.add(point, *content));

        let box_size = [1.0f32, 1.0f32, 1.0f32];
        let query_points: Vec<[f32; 3]> = (0..NUM_QUERIES)
            .map(|_| rand::random::<[f32; 3]>())
            .collect();

        for query_point in query_points {
            let expected = linear_search_periodic(&content_to_add, &query_point, RADIUS, max_qty, &box_size);
            let result = tree.nearest_n_within_periodic::<SquaredEuclidean>(
                &query_point,
                RADIUS,
                max_qty,
                true,
                &box_size,
            );

            assert_eq!(result.len(), expected.len());
            for (actual, expected) in result.iter().zip(expected.iter()) {
                assert!((actual.distance - expected.distance).abs() < 1e-5);
                assert_eq!(actual.item, expected.item);
            }
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

    fn linear_search_periodic<A: Axis, const K: usize>(
        content: &[([A; K], u32)],
        query_point: &[A; K],
        radius: A,
        max_qty: NonZero<usize>,
        box_size: &[A; K],
    ) -> Vec<NearestNeighbour<A, u32>> {
        let mut matching_items = vec![];

        for &(point, item) in content {
            let dist = periodic_dist::<A, K>(query_point, &point, box_size);
            if dist < radius {
                matching_items.push(NearestNeighbour {
                    distance: dist,
                    item,
                });
            }
        }

        stabilize_neighbours(&mut matching_items);
        matching_items.truncate(max_qty.get());
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

    fn stabilize_neighbours<A: Axis>(matching_items: &mut [NearestNeighbour<A, u32>]) {
        matching_items.sort_unstable_by(|a, b| {
            let dist_cmp = a.distance.partial_cmp(&b.distance).unwrap();
            if dist_cmp == Ordering::Equal {
                a.item.cmp(&b.item)
            } else {
                dist_cmp
            }
        });
    }

    fn periodic_dist<A: Axis, const K: usize>(
        query: &[A; K],
        point: &[A; K],
        box_size: &[A; K],
    ) -> A {
        (0..K)
            .map(|axis| {
                let diff = (query[axis] - point[axis]).abs();
                let wrapped_diff = diff.min(box_size[axis] - diff);
                wrapped_diff * wrapped_diff
            })
            .fold(A::zero(), std::ops::Add::add)
    }
}
