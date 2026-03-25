use az::{Az, Cast};
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::ops::Rem;

use crate::float::kdtree::{Axis, KdTree};
use crate::nearest_neighbour::NearestNeighbour;
use crate::rkyv_utils::transform;
use crate::traits::DistanceMetric;
use crate::traits::{is_stem_index, Content, Index};

use crate::generate_nearest_n;

macro_rules! generate_float_nearest_n {
    ($doctest_build_tree:tt) => {
        generate_nearest_n!((
            "Finds the nearest `qty` elements to `query`, using the specified
distance metric function.
# Examples

```rust
    use kiddo::KdTree;
    use kiddo::SquaredEuclidean;

    ",
            $doctest_build_tree,
            "

    let nearest: Vec<_> = tree.nearest_n::<SquaredEuclidean>(&[1.0, 2.0, 5.1], 1);

    assert_eq!(nearest.len(), 1);
    assert!((nearest[0].distance - 0.01f64).abs() < f64::EPSILON);
    assert_eq!(nearest[0].item, 100);
```"
        ));
    };
}

impl<A: Axis, T: Content, const K: usize, const B: usize, IDX: Index<T = IDX>>
    KdTree<A, T, K, B, IDX>
where
    usize: Cast<IDX>,
{
    generate_float_nearest_n!(
        "let mut tree: KdTree<f64, 3> = KdTree::new();
    tree.add(&[1.0, 2.0, 5.0], 100);
    tree.add(&[2.0, 3.0, 6.0], 101);"
    );

    /// Finds the nearest `qty` elements to `query` with periodic boundary conditions.
    ///
    /// `box_size` gives the periodic box length for each axis. Query points are expected
    /// to be wrapped into the same principal cell as the points stored in the tree.
    ///
    /// This first implementation checks all `3^K` wrapped query images, merges duplicate
    /// items that arise from different images, and returns the best `qty` unique items.
    #[inline]
    pub fn nearest_n_periodic<D>(
        &self,
        query: &[A; K],
        qty: usize,
        box_size: &[A; K],
    ) -> Vec<NearestNeighbour<A, T>>
    where
        D: DistanceMetric<A, K>,
        T: std::hash::Hash + Eq,
    {
        if qty == 0 {
            return Vec::new();
        }

        box_size.iter().for_each(|axis_len| {
            assert!(
                *axis_len > A::zero(),
                "periodic box sizes must be strictly positive"
            );
        });

        let mut wrapped_query = *query;
        let mut best_by_item: HashMap<T, A> = HashMap::new();

        self.nearest_n_periodic_recurse::<D>(
            query,
            qty,
            box_size,
            0,
            &mut wrapped_query,
            &mut best_by_item,
        );

        let mut results: Vec<_> = best_by_item
            .into_iter()
            .map(|(item, distance)| NearestNeighbour { distance, item })
            .collect();

        results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        results.truncate(qty);
        results
    }

    fn nearest_n_periodic_recurse<D>(
        &self,
        query: &[A; K],
        qty: usize,
        box_size: &[A; K],
        axis: usize,
        wrapped_query: &mut [A; K],
        best_by_item: &mut HashMap<T, A>,
    ) where
        D: DistanceMetric<A, K>,
        T: std::hash::Hash + Eq,
    {
        if axis == K {
            for candidate in self.nearest_n::<D>(wrapped_query, qty) {
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
        self.nearest_n_periodic_recurse::<D>(
            query,
            qty,
            box_size,
            axis + 1,
            wrapped_query,
            best_by_item,
        );

        wrapped_query[axis] = original;
        self.nearest_n_periodic_recurse::<D>(
            query,
            qty,
            box_size,
            axis + 1,
            wrapped_query,
            best_by_item,
        );

        wrapped_query[axis] = original + axis_len;
        self.nearest_n_periodic_recurse::<D>(
            query,
            qty,
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
    generate_float_nearest_n!(
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
    generate_float_nearest_n!(
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

    type AX = f32;

    #[test]
    fn can_query_nearest_n_item() {
        let mut tree: KdTree<AX, u32, 4, 8, u32> = KdTree::new();

        let content_to_add: [([AX; 4], u32); 16] = [
            ([0.9f32, 0.0f32, 0.9f32, 0.0f32], 9),    // 1.34
            ([0.4f32, 0.5f32, 0.4f32, 0.51f32], 4),   // 0.86
            ([0.12f32, 0.3f32, 0.12f32, 0.3f32], 12), // 1.82
            ([0.7f32, 0.2f32, 0.7f32, 0.22f32], 7),   // 0.86
            ([0.13f32, 0.4f32, 0.13f32, 0.4f32], 13), // 1.56
            ([0.6f32, 0.3f32, 0.6f32, 0.33f32], 6),   // 0.86
            ([0.2f32, 0.7f32, 0.2f32, 0.7f32], 2),    // 1.46
            ([0.14f32, 0.5f32, 0.14f32, 0.5f32], 14), // 1.38
            ([0.3f32, 0.6f32, 0.3f32, 0.6f32], 3),    // 1.06
            ([0.10f32, 0.1f32, 0.10f32, 0.1f32], 10), // 2.26
            ([0.16f32, 0.7f32, 0.16f32, 0.7f32], 16), // 1.54
            ([0.1f32, 0.8f32, 0.1f32, 0.8f32], 1),    // 1.86
            ([0.15f32, 0.6f32, 0.15f32, 0.6f32], 15), // 1.36
            ([0.5f32, 0.4f32, 0.5f32, 0.44f32], 5),   // 0.86
            ([0.8f32, 0.1f32, 0.8f32, 0.15f32], 8),   // 0.86
            ([0.11f32, 0.2f32, 0.11f32, 0.2f32], 11), // 2.04
        ];

        for (point, item) in content_to_add {
            tree.add(&point, item);
        }

        assert_eq!(tree.size(), 16);

        let query_point = [0.78f32, 0.55f32, 0.78f32, 0.55f32];

        let expected = vec![(0.17569996, 6), (0.19139998, 5), (0.24420004, 7)];

        let result: Vec<_> = tree
            .nearest_n::<SquaredEuclidean>(&query_point, 3)
            .into_iter()
            .map(|n| (n.distance, n.item))
            .collect();
        assert_eq!(result, expected);

        let qty = 10;
        let mut rng = rand::rng();
        for _i in 0..1000 {
            let query_point = [
                rng.random_range(0f32..1f32),
                rng.random_range(0f32..1f32),
                rng.random_range(0f32..1f32),
                rng.random_range(0f32..1f32),
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

    #[test]
    fn can_query_nearest_n_item_with_periodic_boundaries() {
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

        let result = tree.nearest_n_periodic::<SquaredEuclidean>(&query_point, 2, &box_size);

        assert_eq!(result.len(), 2);
        assert!((result[0].distance - 0.01f64).abs() < f64::EPSILON);
        assert_eq!(result[0].item, 1);
        assert!((result[1].distance - 0.0194f64).abs() < f64::EPSILON);
        assert_eq!(result[1].item, 2);
    }

    #[test]
    fn can_query_nearest_n_item_with_periodic_boundaries_large_scale() {
        const TREE_SIZE: usize = 10_000;
        const NUM_QUERIES: usize = 200;
        const N: usize = 7;

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
            let expected =
                linear_search_periodic(&content_to_add, N, &query_point, &box_size);
            let result = tree.nearest_n_periodic::<SquaredEuclidean>(&query_point, N, &box_size);

            assert_eq!(result.len(), expected.len());
            for (actual, expected) in result.iter().zip(expected.iter()) {
                assert!((actual.distance - expected.distance).abs() < 1e-5);
                assert_eq!(actual.item, expected.item);
            }
        }
    }

    fn linear_search<A: Axis, const K: usize>(
        content: &[([A; K], u32)],
        qty: usize,
        query_point: &[A; K],
    ) -> Vec<(A, u32)> {
        let mut results = vec![];

        for &(p, item) in content {
            let dist = SquaredEuclidean::dist(query_point, &p);
            if results.len() < qty {
                results.push((dist, item));
                results.sort_by(|(a_dist, _), (b_dist, _)| a_dist.partial_cmp(b_dist).unwrap());
            } else if dist < results[qty - 1].0 {
                results[qty - 1] = (dist, item);
                results.sort_by(|(a_dist, _), (b_dist, _)| a_dist.partial_cmp(b_dist).unwrap());
            }
        }

        results
    }

    fn linear_search_periodic<A: Axis, const K: usize>(
        content: &[([A; K], u32)],
        qty: usize,
        query_point: &[A; K],
        box_size: &[A; K],
    ) -> Vec<NearestNeighbour<A, u32>> {
        let mut results = vec![];

        for &(point, item) in content {
            let distance = periodic_dist::<A, K>(query_point, &point, box_size);
            let candidate = NearestNeighbour { distance, item };

            if results.len() < qty {
                results.push(candidate);
                results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
            } else if distance < results[qty - 1].distance {
                results[qty - 1] = candidate;
                results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
            }
        }

        results
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
