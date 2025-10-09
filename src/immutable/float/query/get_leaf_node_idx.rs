use crate::generate_immutable_get_leaf_node_idx;
use crate::immutable::float::kdtree::ImmutableKdTree;
#[allow(unused_imports)]
use crate::leaf_slice::float::{LeafSliceFloat, LeafSliceFloatChunk};
use crate::traits::StemStrategy;
use crate::traits::{Axis, Content};
use az::Cast;

macro_rules! generate_immutable_float_get_leaf_node_idx {
    ($doctest_build_tree:tt) => {
        generate_immutable_get_leaf_node_idx!((
            "Queries the tree to find the index of the leaf node matching the query.

# Examples

```rust
    use kiddo::ImmutableKdTree;
    use kiddo::SquaredEuclidean;

    ",
            $doctest_build_tree,
            "

    let leaf_node_idx = tree.get_leaf_node_idx(&[1.0, 2.0, 5.1]);

    assert_eq!(leaf_node_idx, 0);
```"
        ));
    };
}

impl<A: Axis, T: Content, SO: StemStrategy, const K: usize, const B: usize>
    ImmutableKdTree<A, T, SO, K, B>
{
    generate_immutable_float_get_leaf_node_idx!(
        "let content: Vec<[f64; 3]> = vec!(
            [1.0, 2.0, 5.0],
            [2.0, 3.0, 6.0]
        );

        let tree: ImmutableKdTree<f64, 3> = ImmutableKdTree::new_from_slice(&content);"
    );
}

#[cfg(feature = "rkyv_08")]
impl<A, T, SO, const K: usize, const B: usize>
    crate::immutable::float::kdtree::ArchivedR8ImmutableKdTree<A, T, SO, K, B>
where
    A: Copy
        + Default
        + PartialOrd
        + Axis
        + LeafSliceFloat<T>
        + LeafSliceFloatChunk<T, K>
        + rkyv_08::Archive,
    T: Copy + Default + Content + rkyv_08::Archive,
    SO: StemStrategy,
    usize: Cast<T>,
{
    generate_immutable_float_get_leaf_node_idx!(
        "use std::fs::File;
    use memmap::MmapOptions;
    use rkyv_08::{access_unchecked, Archived};
    use kiddo::immutable::float::kdtree::ArchivedR8ImmutableKdTree;
    use kiddo::Eytzinger;

    let mmap = unsafe { MmapOptions::new().map(&File::open(\"./examples/immutable-doctest-tree_rkyv08.rkyv\").expect(\"./examples/immutable-doctest-tree_rkyv08.rkyv missing\")).unwrap() };
    let tree = unsafe { access_unchecked::<ArchivedR8ImmutableKdTree<f64, u32, Eytzinger, 3, 256>>(&mmap) };"
    );
}

#[cfg(test)]
mod tests {
    use crate::distance::float::Manhattan;
    use crate::immutable::float::kdtree::ImmutableKdTree;
    use crate::nearest_neighbour::NearestNeighbour;
    use crate::stem_strategies::{Donnelly, Eytzinger};

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

        let tree: ImmutableKdTree<AX, u32, Donnelly<4, 64, 4, 4>, 4, 4> =
            ImmutableKdTree::new_from_slice(&content_to_add);

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
