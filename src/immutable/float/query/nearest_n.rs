use crate::float::kdtree::Axis;
use crate::float_leaf_slice::leaf_slice::{LeafSliceFloat, LeafSliceFloatChunk};
use crate::immutable::float::kdtree::ImmutableKdTree;
use crate::nearest_neighbour::NearestNeighbour;
use crate::traits::Content;
use crate::traits::DistanceMetric;
use az::Cast;
use std::collections::HashMap;
use std::num::NonZero;

use crate::generate_immutable_nearest_n;

macro_rules! generate_immutable_float_nearest_n {
    ($doctest_build_tree:tt) => {
        generate_immutable_nearest_n!((
            "Finds the nearest `qty` elements to `query`, according the specified
distance metric function.
# Examples

```rust
    use std::num::NonZero;
    use kiddo::ImmutableKdTree;
    use kiddo::SquaredEuclidean;

    ",
            $doctest_build_tree,
            "

    let nearest: Vec<_> = tree.nearest_n::<SquaredEuclidean>(&[1.0, 2.0, 5.1], NonZero::new(1).unwrap());

    assert_eq!(nearest.len(), 1);
    assert!((nearest[0].distance - 0.01f64).abs() < f64::EPSILON);
    assert_eq!(nearest[0].item, 0);
```"
        ));
    };
}

impl<A: Axis, T: Content, const K: usize, const B: usize> ImmutableKdTree<A, T, K, B>
where
    A: Axis + LeafSliceFloat<T> + LeafSliceFloatChunk<T, K>,
    T: Content,
    usize: Cast<T>,
{
    generate_immutable_float_nearest_n!(
        "let content: Vec<[f64; 3]> = vec!(
            [1.0, 2.0, 5.0],
            [2.0, 3.0, 6.0]
        );

        let tree: ImmutableKdTree<f64, 3> = ImmutableKdTree::new_from_slice(&content);"
    );

    /// Finds the nearest `qty` elements to `query` with periodic boundary conditions.
    ///
    /// `box_size` gives the periodic box length for each axis. Query points are expected
    /// to be wrapped into the same principal cell as the points stored in the tree.
    #[inline]
    pub fn nearest_n_periodic<D>(
        &self,
        query: &[A; K],
        max_qty: NonZero<usize>,
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

        self.nearest_n_periodic_recurse::<D>(
            query,
            max_qty,
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
        results.truncate(max_qty.get());
        results
    }

    fn nearest_n_periodic_recurse<D>(
        &self,
        query: &[A; K],
        max_qty: NonZero<usize>,
        box_size: &[A; K],
        axis: usize,
        wrapped_query: &mut [A; K],
        best_by_item: &mut HashMap<T, A>,
    ) where
        D: DistanceMetric<A, K>,
        T: std::hash::Hash + Eq,
    {
        if axis == K {
            for candidate in self.nearest_n::<D>(wrapped_query, max_qty) {
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
            max_qty,
            box_size,
            axis + 1,
            wrapped_query,
            best_by_item,
        );

        wrapped_query[axis] = original;
        self.nearest_n_periodic_recurse::<D>(
            query,
            max_qty,
            box_size,
            axis + 1,
            wrapped_query,
            best_by_item,
        );

        wrapped_query[axis] = original + axis_len;
        self.nearest_n_periodic_recurse::<D>(
            query,
            max_qty,
            box_size,
            axis + 1,
            wrapped_query,
            best_by_item,
        );

        wrapped_query[axis] = original;
    }
}

#[cfg(feature = "rkyv")]
use crate::immutable::float::kdtree::AlignedArchivedImmutableKdTree;
#[cfg(feature = "rkyv")]
impl<
        A: Axis + rkyv::Archive<Archived = A>,
        T: Content + rkyv::Archive<Archived = T>,
        const K: usize,
        const B: usize,
    > AlignedArchivedImmutableKdTree<'_, A, T, K, B>
{
    generate_immutable_float_nearest_n!(
        "use std::fs::File;
    use memmap::MmapOptions;

    use kiddo::immutable::float::kdtree::AlignedArchivedImmutableKdTree;

    let mmap = unsafe { MmapOptions::new().map(&File::open(\"./examples/immutable-doctest-tree.rkyv\").expect(\"./examples/immutable-doctest-tree.rkyv missing\")).unwrap() };
    let tree: AlignedArchivedImmutableKdTree<f64, u32, 3, 256> = AlignedArchivedImmutableKdTree::from_bytes(&mmap);"
    );
}

#[cfg(feature = "rkyv_08")]
impl<A, T, const K: usize, const B: usize>
    crate::immutable::float::kdtree::ArchivedR8ImmutableKdTree<A, T, K, B>
where
    A: Copy
        + Default
        + PartialOrd
        + Axis
        + LeafSliceFloat<T>
        + LeafSliceFloatChunk<T, K>
        + rkyv_08::Archive,
    T: Copy + Default + Content + rkyv_08::Archive,
    usize: Cast<T>,
{
    generate_immutable_float_nearest_n!(
        "use std::fs::File;
    use memmap::MmapOptions;
    use rkyv_08::{access_unchecked, Archived};
    use kiddo::immutable::float::kdtree::ArchivedR8ImmutableKdTree;

    let mmap = unsafe { MmapOptions::new().map(&File::open(\"./examples/immutable-doctest-tree_rkyv08.rkyv\").expect(\"./examples/immutable-doctest-tree_rkyv08.rkyv missing\")).unwrap() };
    let tree = unsafe { access_unchecked::<ArchivedR8ImmutableKdTree<f64, u32, 3, 256>>(&mmap) };"
    );
}

#[cfg(test)]
mod tests {
    use crate::float::distance::SquaredEuclidean;
    use crate::float::kdtree::Axis;
    use crate::immutable::float::kdtree::ImmutableKdTree;
    use crate::nearest_neighbour::NearestNeighbour;
    use crate::traits::DistanceMetric;
    use az::{Az, Cast};
    use rand::Rng;
    use std::num::NonZero;

    #[test]
    fn can_query_nearest_n_item_f32() {
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
        let max_qty = NonZero::new(10).unwrap();

        let expected = linear_search(&content_to_add, max_qty.into(), &query_point);
        // let expected = vec![(0.17569996, 5), (0.19139998, 13), (0.24420004, 3)];

        let result: Vec<_> = tree
            .nearest_n::<SquaredEuclidean>(&query_point, max_qty)
            .into_iter()
            .map(|n| (n.distance, n.item))
            .collect();
        assert_eq!(result, expected);

        let max_qty = NonZero::new(10).unwrap();
        let mut rng = rand::rng();
        for _i in 0..1000 {
            let query_point = [
                rng.random_range(0f32..1f32),
                rng.random_range(0f32..1f32),
                rng.random_range(0f32..1f32),
                rng.random_range(0f32..1f32),
            ];
            let expected = linear_search(&content_to_add, max_qty.into(), &query_point);

            let result: Vec<_> = tree
                .nearest_n::<SquaredEuclidean>(&query_point, max_qty)
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();

            assert_eq!(result, expected);
            println!("{_i}");
        }
    }

    #[test]
    fn can_query_nearest_10_items_large_scale_f32() {
        const TREE_SIZE: usize = 100_000;
        const NUM_QUERIES: usize = 100;

        let max_qty = NonZero::new(10).unwrap();

        let content_to_add: Vec<[f32; 4]> =
            (0..TREE_SIZE).map(|_| rand::random::<[f32; 4]>()).collect();

        let tree: ImmutableKdTree<f32, u32, 4, 32> =
            ImmutableKdTree::new_from_slice(&content_to_add);

        assert_eq!(tree.size(), TREE_SIZE);

        let query_points: Vec<[f32; 4]> = (0..NUM_QUERIES)
            .map(|_| rand::random::<[f32; 4]>())
            .collect();

        for query_point in query_points {
            let expected = linear_search(&content_to_add, max_qty.into(), &query_point);

            let result: Vec<_> = tree
                .nearest_n::<SquaredEuclidean>(&query_point, max_qty)
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();

            assert_eq!(result, expected);
        }
    }

    #[test]
    fn can_query_nearest_n_item_f64() {
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

        let query_point = [0.78f64, 0.55f64, 0.78f64, 0.55f64];

        let expected = vec![
            (0.17570000000000008, 5),
            (0.19140000000000001, 13),
            (0.2442000000000001, 3),
        ];

        let max_qty = NonZero::new(3).unwrap();

        let result: Vec<_> = tree
            .nearest_n::<SquaredEuclidean>(&query_point, max_qty)
            .into_iter()
            .map(|n| (n.distance, n.item))
            .collect();
        assert_eq!(result, expected);

        let max_qty = NonZero::new(10).unwrap();
        let mut rng = rand::rng();
        for _i in 0..1000 {
            let query_point = [
                rng.random_range(0f64..1f64),
                rng.random_range(0f64..1f64),
                rng.random_range(0f64..1f64),
                rng.random_range(0f64..1f64),
            ];
            let expected = linear_search(&content_to_add, max_qty.into(), &query_point);

            let result: Vec<_> = tree
                .nearest_n::<SquaredEuclidean>(&query_point, max_qty)
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();

            assert_eq!(result, expected);
        }
    }

    #[test]
    fn can_query_nearest_10_items_large_scale_f64() {
        const TREE_SIZE: usize = 100_000;
        const NUM_QUERIES: usize = 100;

        let max_qty = NonZero::new(10).unwrap();

        let content_to_add: Vec<[f64; 4]> =
            (0..TREE_SIZE).map(|_| rand::random::<[f64; 4]>()).collect();

        let tree: ImmutableKdTree<f64, u32, 4, 32> =
            ImmutableKdTree::new_from_slice(&content_to_add);

        assert_eq!(tree.size(), TREE_SIZE);

        let query_points: Vec<[f64; 4]> = (0..NUM_QUERIES)
            .map(|_| rand::random::<[f64; 4]>())
            .collect();

        for query_point in query_points {
            let expected = linear_search(&content_to_add, max_qty.into(), &query_point);

            let result: Vec<_> = tree
                .nearest_n::<SquaredEuclidean>(&query_point, max_qty)
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();

            assert_eq!(result, expected);
        }
    }

    #[test]
    fn can_query_nearest_n_item_with_periodic_boundaries_f64() {
        let content_to_add = [
            [0.95f64, 0.50f64],
            [0.92f64, 0.55f64],
            [0.40f64, 0.50f64],
            [0.10f64, 0.10f64],
        ];

        let tree: ImmutableKdTree<f64, u32, 2, 8> = ImmutableKdTree::new_from_slice(&content_to_add);
        let query_point = [0.05f64, 0.50f64];
        let box_size = [1.0f64, 1.0f64];
        let max_qty = NonZero::new(2).unwrap();

        let result = tree.nearest_n_periodic::<SquaredEuclidean>(&query_point, max_qty, &box_size);
        assert_eq!(result.len(), 2);
        assert!((result[0].distance - 0.01f64).abs() < f64::EPSILON);
        assert_eq!(result[0].item, 0);
        assert!((result[1].distance - 0.0194f64).abs() < f64::EPSILON);
        assert_eq!(result[1].item, 1);
    }

    #[test]
    fn can_query_nearest_n_item_with_periodic_boundaries_large_scale_f32() {
        const TREE_SIZE: usize = 10_000;
        const NUM_QUERIES: usize = 200;

        let max_qty = NonZero::new(5).unwrap();
        let content_to_add: Vec<[f32; 3]> = (0..TREE_SIZE).map(|_| rand::random::<[f32; 3]>()).collect();
        let tree: ImmutableKdTree<f32, u32, 3, 32> = ImmutableKdTree::new_from_slice(&content_to_add);
        let box_size = [1.0f32, 1.0f32, 1.0f32];
        let query_points: Vec<[f32; 3]> = (0..NUM_QUERIES).map(|_| rand::random::<[f32; 3]>()).collect();

        for query_point in query_points.iter() {
            let expected = linear_search_periodic(&content_to_add, max_qty.into(), query_point, &box_size);
            let result = tree.nearest_n_periodic::<SquaredEuclidean>(query_point, max_qty, &box_size);

            assert_eq!(result.len(), expected.len());
            for (actual, expected) in result.iter().zip(expected.iter()) {
                assert!((actual.distance - expected.distance).abs() < 1e-5);
                assert_eq!(actual.item, expected.item);
            }
        }
    }

    fn linear_search<A: Axis, R, const K: usize>(
        content: &[[A; K]],
        qty: usize,
        query_point: &[A; K],
    ) -> Vec<(A, R)>
    where
        usize: Cast<R>,
    {
        let mut results: Vec<(A, R)> = vec![];

        for (idx, p) in content.iter().enumerate() {
            let dist = SquaredEuclidean::dist(query_point, p);
            if results.len() < qty {
                results.push((dist, idx.az::<R>()));
                results.sort_by(|(a_dist, _), (b_dist, _)| a_dist.partial_cmp(b_dist).unwrap());
            } else if dist < results[qty - 1].0 {
                results[qty - 1] = (dist, idx.az::<R>());
                results.sort_by(|(a_dist, _), (b_dist, _)| a_dist.partial_cmp(b_dist).unwrap());
            }
        }

        results
    }

    fn linear_search_periodic<A: Axis, const K: usize>(
        content: &[[A; K]],
        qty: usize,
        query_point: &[A; K],
        box_size: &[A; K],
    ) -> Vec<NearestNeighbour<A, u32>> {
        let mut results = vec![];

        for (idx, point) in content.iter().enumerate() {
            let dist = periodic_dist(query_point, point, box_size);
            let candidate = NearestNeighbour {
                distance: dist,
                item: idx as u32,
            };

            if results.len() < qty {
                results.push(candidate);
                results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
            } else if dist < results[qty - 1].distance {
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
