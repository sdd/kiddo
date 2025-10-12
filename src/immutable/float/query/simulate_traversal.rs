use std::sync::mpsc::Sender;

use crate::generate_immutable_simulate_traversal;
use crate::immutable::float::kdtree::ImmutableKdTree;
#[allow(unused_imports)]
use crate::leaf_slice::float::{LeafSliceFloat, LeafSliceFloatChunk};
use crate::traits::StemStrategy;
use crate::traits::{Axis, Content};
use az::Cast;

macro_rules! generate_immutable_float_simulate_traversal {
    ($doctest_build_tree:tt) => {
        generate_immutable_simulate_traversal!((
            "Queries the tree to find the index of the leaf node matching the query.

# Examples

```rust
    use kiddo::ImmutableKdTree;
    use kiddo::SquaredEuclidean;

    ",
            $doctest_build_tree,
            "

    let leaf_node_idx = tree.simulate_traversal(&[1.0, 2.0, 5.1]);

    assert_eq!(leaf_node_idx, 0);
```"
        ));
    };
}

impl<A: Axis, T: Content, SO: StemStrategy, const K: usize, const B: usize>
    ImmutableKdTree<A, T, SO, K, B>
{
    generate_immutable_float_simulate_traversal!(
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
    generate_immutable_float_simulate_traversal!(
        "use std::fs::File;
    use memmap::MmapOptions;
    use rkyv_08::{access_unchecked, Archived};
    use kiddo::immutable::float::kdtree::ArchivedR8ImmutableKdTree;
    use kiddo::Eytzinger;

    let mmap = unsafe { MmapOptions::new().map(&File::open(\"./examples/immutable-doctest-tree_rkyv08.rkyv\").expect(\"./examples/immutable-doctest-tree_rkyv08.rkyv missing\")).unwrap() };
    let tree = unsafe { access_unchecked::<ArchivedR8ImmutableKdTree<f64, u32, Eytzinger, 3, 256>>(&mmap) };"
    );
}
