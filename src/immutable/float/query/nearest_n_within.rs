use az::Cast;
use sorted_vec::SortedVec;
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::num::NonZero;
use std::ops::Rem;

use crate::float::kdtree::Axis;
use crate::float::result_collection::ResultCollection;
use crate::float_leaf_slice::leaf_slice::{LeafSliceFloat, LeafSliceFloatChunk};
use crate::immutable::float::kdtree::ImmutableKdTree;
use crate::nearest_neighbour::NearestNeighbour;
use crate::traits::Content;
use crate::traits::DistanceMetric;

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
    A: Axis + LeafSliceFloat<T> + LeafSliceFloatChunk<T, K>,
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

    #[inline]
    pub fn nearest_n_within_periodic<D>(
        &self,
        query: &[A; K],
        dist: A,
        max_items: NonZero<usize>,
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
        max_items: NonZero<usize>,
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
use crate::immutable::float::kdtree::AlignedArchivedImmutableKdTree;
#[cfg(feature = "rkyv")]
impl<
        A: Axis + rkyv::Archive<Archived = A>,
        T: Content + rkyv::Archive<Archived = T>,
        const K: usize,
        const B: usize,
    > AlignedArchivedImmutableKdTree<'_, A, T, K, B>
where
    A: Axis + LeafSliceFloat<T> + LeafSliceFloatChunk<T, K>,
    T: Content,
    usize: Cast<T>,
{
    generate_immutable_float_nearest_n_within!(
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
    generate_immutable_float_nearest_n_within!(
        "use std::fs::File;
    use memmap::MmapOptions;

    use kiddo::immutable::float::kdtree::ArchivedR8ImmutableKdTree;

    let mmap = unsafe { MmapOptions::new().map(&File::open(\"./examples/immutable-doctest-tree_rkyv08.rkyv\").expect(\"./examples/immutable-doctest-tree_rkyv08.rkyv missing\")).unwrap() };
    let tree = unsafe { rkyv_08::access_unchecked::<ArchivedR8ImmutableKdTree<f64, u32, 3, 256>>(&mmap) };"
    );
}

#[cfg(test)]
mod tests {
    use crate::float::distance::SquaredEuclidean;
    use crate::float::kdtree::Axis;
    use crate::immutable::float::kdtree::ImmutableKdTree;
    use crate::nearest_neighbour::NearestNeighbour;
    use crate::traits::DistanceMetric;
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

        let mut rng = rand::rng();
        for _i in 0..1000 {
            let query_point = [
                rng.random_range(0f32..1f32),
                rng.random_range(0f32..1f32),
                rng.random_range(0f32..1f32),
                rng.random_range(0f32..1f32),
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

        let max_qty: NonZero<usize> = NonZero::new(3).unwrap();

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
                .take(max_qty.into())
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
    fn can_query_items_within_periodic_boundaries() {
        let content_to_add = [
            [0.95f64, 0.50f64],
            [0.92f64, 0.55f64],
            [0.40f64, 0.50f64],
            [0.10f64, 0.10f64],
        ];

        let tree: ImmutableKdTree<f64, u32, 2, 8> = ImmutableKdTree::new_from_slice(&content_to_add);
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
        assert_eq!(result[0].item, 0);
        assert!((result[1].distance - 0.0194f64).abs() < f64::EPSILON);
        assert_eq!(result[1].item, 1);
    }

    #[test]
    fn can_query_items_within_periodic_boundaries_large_scale() {
        const TREE_SIZE: usize = 10_000;
        const NUM_QUERIES: usize = 200;
        const RADIUS: f32 = 0.05;

        let max_qty = NonZero::new(5).unwrap();
        let content_to_add: Vec<[f32; 3]> = (0..TREE_SIZE).map(|_| rand::random::<[f32; 3]>()).collect();
        let tree: ImmutableKdTree<AX, u32, 3, 32> = ImmutableKdTree::new_from_slice(&content_to_add);
        let box_size = [1.0f32, 1.0f32, 1.0f32];
        let query_points: Vec<[f32; 3]> = (0..NUM_QUERIES).map(|_| rand::random::<[f32; 3]>()).collect();

        for query_point in query_points.iter() {
            let expected = linear_search_periodic(&content_to_add, query_point, RADIUS, max_qty, &box_size);
            let result = tree.nearest_n_within_periodic::<SquaredEuclidean>(
                query_point,
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

    fn linear_search_periodic<A: Axis, const K: usize>(
        content: &[[A; K]],
        query_point: &[A; K],
        radius: A,
        max_qty: NonZero<usize>,
        box_size: &[A; K],
    ) -> Vec<NearestNeighbour<A, u32>> {
        let mut matching_items = vec![];

        for (idx, point) in content.iter().enumerate() {
            let dist = periodic_dist(query_point, point, box_size);
            if dist < radius {
                matching_items.push(NearestNeighbour {
                    distance: dist,
                    item: idx as u32,
                });
            }
        }

        matching_items.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        matching_items.truncate(max_qty.get());
        matching_items
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
