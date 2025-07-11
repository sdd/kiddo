use az::{Az, Cast};
use std::ops::Rem;

use crate::float::kdtree::{Axis, KdTree, LeafNode};
use crate::generate_nearest_one;
use crate::nearest_neighbour::NearestNeighbour;
use crate::rkyv_utils::transform;
use crate::traits::DistanceMetric;
use crate::traits::{is_stem_index, Content, Index};

macro_rules! generate_float_nearest_one {
    ($leafnode:ident, $doctest_build_tree:tt) => {
        generate_nearest_one!(
            $leafnode,
            (
                "Finds the nearest element to `query`, using the specified
distance metric function.

Faster than querying for nearest_n(point, 1, ...) due
to not needing to allocate memory or maintain sorted results.

The nearest_one_point version also returns the coordinates of the nearest point.

# Examples

```rust
    use kiddo::KdTree;
    use kiddo::SquaredEuclidean;

    ",
                $doctest_build_tree,
                "

    let nearest = tree.nearest_one::<SquaredEuclidean>(&[1.0, 2.0, 5.1]);

    assert!((nearest.distance - 0.01f64).abs() < f64::EPSILON);
    assert_eq!(nearest.item, 100);

    let (nearest, nearest_point) = tree.nearest_one_point::<SquaredEuclidean>(&[1.0, 2.0, 5.1]);

    assert!((nearest.distance - 0.01f64).abs() < f64::EPSILON);
    assert_eq!(nearest.item, 100);
    assert_eq!(nearest_point, [1.0, 2.0, 5.0]);
```"
            )
        );
    };
}

impl<A: Axis, T: Content, const K: usize, const B: usize, IDX: Index<T = IDX>>
    KdTree<A, T, K, B, IDX>
where
    usize: Cast<IDX>,
{
    generate_float_nearest_one!(
        LeafNode,
        "let mut tree: KdTree<f64, 3> = KdTree::new();
    tree.add(&[1.0, 2.0, 5.0], 100);
    tree.add(&[2.0, 3.0, 6.0], 101);"
    );
}

#[cfg(feature = "rkyv")]
use crate::float::kdtree::{ArchivedKdTree, ArchivedLeafNode};
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
    generate_float_nearest_one!(
        ArchivedLeafNode,
        "use std::fs::File;
    use memmap::MmapOptions;

    let mmap = unsafe { MmapOptions::new().map(&File::open(\"./examples/float-doctest-tree.rkyv\").expect(\"./examples/float-doctest-tree.rkyv missing\")).unwrap() };
    let tree = unsafe { rkyv::archived_root::<KdTree<f64, 3>>(&mmap) };"
    );
}

#[cfg(feature = "rkyv_08")]
use crate::float::kdtree::{ArchivedR8KdTree, ArchivedR8LeafNode};
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
    generate_float_nearest_one!(
        ArchivedR8LeafNode,
        "use std::fs::File;
    use memmap::MmapOptions;
    use kiddo::float::kdtree::ArchivedR8KdTree;

    let mmap = unsafe { MmapOptions::new().map(&File::open(\"./examples/float-doctest-tree-rkyv_08.rkyv\").expect(\"./examples/float-doctest-tree-rkyv_08.rkyv missing\")).unwrap() };
    let tree = unsafe { rkyv_08::access_unchecked::<ArchivedR8KdTree<f64, u64, 3, 32, u32>>(&mmap) };"
    );
}

#[cfg(test)]
mod tests {
    use crate::float::distance::Manhattan;
    use crate::float::kdtree::{Axis, KdTree};
    use crate::nearest_neighbour::NearestNeighbour;
    use crate::traits::DistanceMetric;
    use rand::Rng;

    type AX = f32;

    #[test]
    fn can_query_nearest_one_item() {
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

        let expected = NearestNeighbour {
            distance: 0.819_999_93,
            item: 5,
        };

        let result = tree.nearest_one::<Manhattan>(&query_point);
        assert_eq!(result.distance, expected.distance);

        let mut rng = rand::rng();
        for _i in 0..1000 {
            let query_point = [
                rng.random_range(0f32..1f32),
                rng.random_range(0f32..1f32),
                rng.random_range(0f32..1f32),
                rng.random_range(0f32..1f32),
            ];
            let expected = linear_search(&content_to_add, &query_point);

            let result = tree.nearest_one::<Manhattan>(&query_point);

            assert_eq!(result.distance, expected.distance);
        }
    }

    #[test]
    fn can_query_nearest_one_item_large_scale() {
        const TREE_SIZE: usize = 100_000;
        const NUM_QUERIES: usize = 100;

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
            let expected = linear_search(&content_to_add, &query_point);

            let result = tree.nearest_one::<Manhattan>(&query_point);

            assert_eq!(result.distance, expected.distance);
            assert_eq!(result.item, expected.item);
        }
    }

    fn linear_search<A: Axis, const K: usize>(
        content: &[([A; K], u32)],
        query_point: &[A; K],
    ) -> NearestNeighbour<A, u32> {
        let mut best_dist: A = A::infinity();
        let mut best_item: u32 = u32::MAX;

        for &(p, item) in content {
            let dist = Manhattan::dist(query_point, &p);
            if dist < best_dist {
                best_item = item;
                best_dist = dist;
            }
        }

        NearestNeighbour {
            distance: best_dist,
            item: best_item,
        }
    }
}
