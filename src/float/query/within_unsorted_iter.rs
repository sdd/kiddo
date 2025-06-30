use az::{Az, Cast};
use generator::{done, Gn, Scope};
use std::ops::Rem;

use crate::float::kdtree::{Axis, KdTree};
use crate::nearest_neighbour::NearestNeighbour;
use crate::rkyv_utils::transform;
use crate::traits::DistanceMetric;
use crate::traits::{is_stem_index, Content, Index};
use crate::within_unsorted_iter::WithinUnsortedIter;

use crate::generate_within_unsorted_iter;

macro_rules! generate_float_within_unsorted_iter {
    ($doctest_build_tree:tt) => {
        generate_within_unsorted_iter!((
            "Finds all elements within `dist` of `query`, using the specified
distance metric function.

Returns an `Iterator`. Results are returned in arbitrary order.

# Examples

```rust
use kiddo::KdTree;
use kiddo::SquaredEuclidean;
",
            $doctest_build_tree,
            "

let within = tree.within_unsorted_iter::<SquaredEuclidean>(&[1.0, 2.0, 5.0], 10f64).collect::<Vec<_>>();

assert_eq!(within.len(), 2);
```"
        ));
    };
}

impl<
        'a,
        'query,
        A: Axis,
        T: Content,
        const K: usize,
        const B: usize,
        IDX: Index<T = IDX> + Send,
    > KdTree<A, T, K, B, IDX>
where
    usize: Cast<IDX>,
{
    generate_float_within_unsorted_iter!(
        "
let mut tree: KdTree<f64, 3> = KdTree::new();
tree.add(&[1.0, 2.0, 5.0], 100);
tree.add(&[2.0, 3.0, 6.0], 101);"
    );
}

#[cfg(feature = "rkyv_08")]
use crate::float::kdtree::ArchivedR8KdTree;
#[cfg(feature = "rkyv_08")]
impl<
        'a,
        'query,
        A: Axis + Send + rkyv::Archive,
        T: Content + Send + rkyv::Archive,
        const K: usize,
        const B: usize,
        IDX: Index<T = IDX> + Send,
    > ArchivedR8KdTree<A, T, K, B, IDX>
where
    usize: Cast<IDX>,
    IDX: rkyv::Archive,
    <A as rkyv::Archive>::Archived: Sync,
    <T as rkyv::Archive>::Archived: Sync,
    <IDX as rkyv::Archive>::Archived: Sync,
{
    generate_float_within_unsorted_iter!(
        "use std::fs::File;
    use memmap::MmapOptions;
    use kiddo::float::kdtree::ArchivedR8KdTree;

    let mmap = unsafe { MmapOptions::new().map(&File::open(\"./examples/float-doctest-tree-rkyv_08.rkyv\").expect(\"./examples/float-doctest-tree-rkyv_08.rkyv missing\")).unwrap() };
    let tree = unsafe { rkyv::access_unchecked::<ArchivedR8KdTree<f64, u64, 3, 32, u32>>(&mmap) };"
    );
}

#[cfg(test)]
mod tests {
    use crate::float::distance::Manhattan;
    use crate::float::kdtree::{Axis, KdTree};
    use crate::nearest_neighbour::NearestNeighbour;
    use crate::traits::DistanceMetric;
    use rand::Rng;
    use std::cmp::Ordering;

    type AX = f32;

    #[test]
    fn can_query_items_within_radius() {
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
        let expected = linear_search(&content_to_add, &query_point, radius);

        // Store some iterators in a way that the test will fail to compile
        // if the lifetime of the iterator is tied to the query as well as to
        // the lifetime of the tree
        let mut iterators = Vec::new();
        for _ in 0..2 {
            // take a copy of query_point to ensure that the lifetime of the
            // iterator is tied to the lifetime of the tree and not the lifetime
            // of the query
            let temp_query = query_point;

            let iter = tree.within_unsorted_iter::<Manhattan>(&temp_query, radius);

            iterators.push(iter);
        }

        for iter in iterators {
            let result: Vec<_> = iter.collect();
            assert_eq!(result, expected);
        }

        let mut rng = rand::rng();
        for _i in 0..1000 {
            let query_point = [
                rng.random_range(0f32..1f32),
                rng.random_range(0f32..1f32),
                rng.random_range(0f32..1f32),
                rng.random_range(0f32..1f32),
            ];
            let radius = 0.2;
            let expected = linear_search(&content_to_add, &query_point, radius);

            let mut result: Vec<_> = tree
                .within_unsorted_iter::<Manhattan>(&query_point, radius)
                .collect();
            stabilize_sort(&mut result);

            assert_eq!(result, expected);
        }
    }

    #[test]
    fn can_query_items_unsorted_within_radius_large_scale() {
        const TREE_SIZE: usize = 100_000;
        const NUM_QUERIES: usize = 100;
        const RADIUS: f32 = 0.2;

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
            let expected = linear_search(&content_to_add, &query_point, RADIUS);

            let mut result: Vec<_> = tree
                .within_unsorted_iter::<Manhattan>(&query_point, RADIUS)
                .collect();

            stabilize_sort(&mut result);
            assert_eq!(result, expected);
        }
    }

    fn linear_search<A: Axis, const K: usize>(
        content: &[([A; K], u32)],
        query_point: &[A; K],
        radius: A,
    ) -> Vec<NearestNeighbour<A, u32>> {
        let mut matching_items = vec![];

        for &(p, item) in content {
            let distance = Manhattan::dist(query_point, &p);
            if distance < radius {
                matching_items.push(NearestNeighbour { distance, item });
            }
        }

        stabilize_sort(&mut matching_items);
        matching_items
    }

    fn stabilize_sort<A: Axis>(matching_items: &mut [NearestNeighbour<A, u32>]) {
        matching_items.sort_unstable_by(|a, b| {
            let dist_cmp = a.distance.partial_cmp(&b.distance).unwrap();
            if dist_cmp == Ordering::Equal {
                a.item.cmp(&b.item)
            } else {
                dist_cmp
            }
        });
    }
}
