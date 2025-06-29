use crate::float::kdtree::Axis;
#[allow(unused_imports)]
use crate::float_leaf_slice::leaf_slice::{LeafSliceFloat, LeafSliceFloatChunk};
use crate::generate_immutable_approx_nearest_one;
use crate::immutable::float::kdtree::ImmutableKdTree;
use crate::nearest_neighbour::NearestNeighbour;
use crate::traits::Content;
use crate::traits::DistanceMetric;
use az::Cast;

macro_rules! generate_immutable_approx_float_nearest_one {
    ($doctest_build_tree:tt) => {
        generate_immutable_approx_nearest_one!((
            "Queries the tree to find the approximate nearest element to `query`, using the specified
distance metric function.

Faster than querying for nearest_one(point) due
to not recursing up the tree to find potentially closer points in other branches.

# Examples

```rust
    use kiddo::ImmutableKdTree;
    use kiddo::SquaredEuclidean;

    ",
            $doctest_build_tree,
            "

    let nearest = tree.approx_nearest_one::<SquaredEuclidean>(&[1.0, 2.0, 5.1]);

    assert!((nearest.distance - 0.01f64).abs() < f64::EPSILON);
    assert_eq!(nearest.item, 0);
```"
        ));
    };
}

impl<A: Axis, T: Content, const K: usize, const B: usize> ImmutableKdTree<A, T, K, B> {
    generate_immutable_approx_float_nearest_one!(
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
    generate_immutable_approx_float_nearest_one!(
        "use std::fs::File;
    use memmap::MmapOptions;

    use kiddo::immutable::float::kdtree::AlignedArchivedImmutableKdTree;

    let mmap = unsafe { MmapOptions::new().map(&File::open(\"./examples/immutable-doctest-tree.rkyv\").expect(\"./examples/immutable-doctest-tree.rkyv missing\")).unwrap() };
    let tree: AlignedArchivedImmutableKdTree<f64, u32, 3, 256> = AlignedArchivedImmutableKdTree::from_bytes(&mmap);"
    );
}

#[cfg(feature = "rkyv_08")]
impl<A, T, const K: usize, const B: usize>
    crate::immutable::float::kdtree::ArchivedImmutableKdTree<A, T, K, B>
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
    generate_immutable_approx_float_nearest_one!(
        "use std::fs::File;
    use memmap::MmapOptions;
    use rkyv_08::{access_unchecked, Archived};
    use kiddo::immutable::float::kdtree::ArchivedImmutableKdTree;

    let mmap = unsafe { MmapOptions::new().map(&File::open(\"./examples/immutable-doctest-tree_rkyv08.rkyv\").expect(\"./examples/immutable-doctest-tree_rkyv08.rkyv missing\")).unwrap() };
    let tree = unsafe { access_unchecked::<ArchivedImmutableKdTree<f64, u32, 3, 256>>(&mmap) };"
    );
}

#[cfg(test)]
mod tests {
    use crate::float::distance::Manhattan;
    use crate::immutable::float::kdtree::ImmutableKdTree;
    use crate::nearest_neighbour::NearestNeighbour;

    type AX = f32;

    #[test]
    fn can_query_approx_nearest_one_item() {
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
        println!("Tree: {:?}", &tree);

        let query_point = [0.78f32, 0.55f32, 0.78f32, 0.55f32];

        let expected = NearestNeighbour {
            distance: 0.81999993,
            item: 13,
        };

        let result = tree.approx_nearest_one::<Manhattan>(&query_point);
        assert_eq!(result, expected);
    }
}
