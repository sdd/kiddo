use crate::float::kdtree::Axis;
use crate::float_leaf_slice::leaf_slice::{LeafSliceFloat, LeafSliceFloatChunk};
use crate::immutable::float::kdtree::ImmutableKdTree;
use crate::nearest_neighbour::NearestNeighbour;
use crate::traits::Content;
use crate::traits::DistanceMetric;
use az::Cast;
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
}
