use crate::distance_metric::DistanceMetric;
use crate::float::kdtree::Axis;
use crate::immutable::float::kdtree::ImmutableKdTree;
use crate::nearest_neighbour::NearestNeighbour;
use crate::types::Content;

use crate::generate_immutable_approx_nearest_one;

macro_rules! generate_immutable_approx_float_nearest_one {
    ($doctest_build_tree:tt) => {
        generate_immutable_approx_nearest_one!((
            "Queries the tree to find the approximate nearest element to `query`, using the specified
distance metric function.

Faster than querying for nearest_one(point) due
to not recursing up the tree to find potentially closer points in other branches.

# Examples

```rust
    use kiddo::immutable::float::kdtree::ImmutableKdTree;
    use kiddo::float::distance::SquaredEuclidean;

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

        let tree: ImmutableKdTree<f64, u32, 3, 32> = ImmutableKdTree::new_from_slice(&content);"
    );
}

#[cfg(feature = "rkyv")]
use crate::immutable::float::kdtree::ArchivedImmutableKdTree;
#[cfg(feature = "rkyv")]
impl<
        A: Axis + rkyv::Archive<Archived = A>,
        T: Content + rkyv::Archive<Archived = T>,
        const K: usize,
        const B: usize,
    > ArchivedImmutableKdTree<A, T, K, B>
{
    generate_immutable_approx_float_nearest_one!(
        "use std::fs::File;
    use memmap::MmapOptions;

    let mmap = unsafe { MmapOptions::new().map(&File::open(\"./examples/immutable-doctest-tree.rkyv\").unwrap()).unwrap() };
    let tree = unsafe { rkyv::archived_root::<ImmutableKdTree<f64, u32, 3, 32>>(&mmap) };"
    );
}

/*impl<A: Axis, T: Content, const K: usize, const B: usize> ImmutableKdTree<A, T, K, B> {
    /// Queries the tree to find the approximate nearest element to `query`, using the specified
    /// distance metric function.
    ///
    /// Faster than querying for nearest_one(point) due
    /// to not recursing up the tree to find potentially closer points in other branches.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kiddo::immutable::float::kdtree::ImmutableKdTree;
    /// use kiddo::float::distance::SquaredEuclidean;
    ///
    /// let content: Vec<[f64; 3]> = vec!(
    ///     [1.0, 2.0, 5.0],
    ///     [2.0, 3.0, 6.0]
    /// );
    ///
    /// let mut tree: ImmutableKdTree<f64, u32, 3, 32> = ImmutableKdTree::new_from_slice(&content);
    ///
    /// let nearest = tree.approx_nearest_one::<SquaredEuclidean>(&[1.0, 2.0, 5.1]);
    ///
    /// assert!((nearest.distance - 0.01f64).abs() < f64::EPSILON);
    /// assert_eq!(nearest.item, 0);
    /// ```
    #[inline]
    pub fn approx_nearest_one<F>(&self, query: &[A; K], distance_fn: &F) -> NearestNeighbour<A, T>
    where
        F: Fn(&[A; K], &[A; K]) -> A,
    {
        let mut split_dim = 0;
        let mut stem_idx = 1;
        let mut best_item = T::zero();
        let mut best_dist = A::max_value();

        let stem_len = self.stems.len();

        while stem_idx < stem_len {
            let left_child_idx = stem_idx << 1;
            self.prefetch_stems(left_child_idx);

            let val = *unsafe { self.stems.get_unchecked(stem_idx) };
            let is_right_child = usize::from(*unsafe { query.get_unchecked(split_dim) } >= val);

            stem_idx = left_child_idx + is_right_child;

            split_dim += 1;
            split_dim %= K;
        }

        let leaf_node = unsafe { self.leaves.get_unchecked(stem_idx - stem_len) };
        // let leaf_node = &self.leaves[leaf_idx];

        leaf_node
            .content_points
            .iter()
            .enumerate()
            .take(leaf_node.size)
            .for_each(|(idx, entry)| {
                let dist = distance_fn(query, entry);
                if dist < best_dist {
                    best_dist = dist;
                    best_item = unsafe { *leaf_node.content_items.get_unchecked(idx) };
                    // *best_item = leaf_node.content_items[idx]
                }
            });

        NearestNeighbour {
            distance: best_dist,
            item: best_item,
        }
    }
}
*/

#[cfg(test)]
mod tests {
    use crate::float::distance::Manhattan;
    use crate::immutable::float::kdtree::ImmutableKdTree;
    use crate::nearest_neighbour::NearestNeighbour;

    type AX = f32;

    #[test]
    fn can_query_approx_nearest_one_item() {
        let content_to_add: [[AX; 4]; 16] = [
            [0.9f32, 0.0f32, 0.9f32, 0.0f32],   // 1.34
            [0.4f32, 0.5f32, 0.4f32, 0.51f32],  // 0.86
            [0.12f32, 0.3f32, 0.12f32, 0.3f32], // 1.82
            [0.7f32, 0.2f32, 0.7f32, 0.22f32],  // 0.86
            [0.13f32, 0.4f32, 0.13f32, 0.4f32], // 1.56
            [0.6f32, 0.3f32, 0.6f32, 0.33f32],  // 0.86
            [0.2f32, 0.7f32, 0.2f32, 0.7f32],   // 1.46
            [0.14f32, 0.5f32, 0.14f32, 0.5f32], // 1.38
            [0.3f32, 0.6f32, 0.3f32, 0.6f32],   // 1.06
            [0.10f32, 0.1f32, 0.10f32, 0.1f32], // 2.26
            [0.16f32, 0.7f32, 0.16f32, 0.7f32], // 1.54
            [0.1f32, 0.8f32, 0.1f32, 0.8f32],   // 1.86
            [0.15f32, 0.6f32, 0.15f32, 0.6f32], // 1.36
            [0.5f32, 0.4f32, 0.5f32, 0.44f32],  // 0.86
            [0.8f32, 0.1f32, 0.8f32, 0.15f32],  // 0.86
            [0.11f32, 0.2f32, 0.11f32, 0.2f32], // 2.04
        ];

        let tree: ImmutableKdTree<AX, u32, 4, 4> = ImmutableKdTree::new_from_slice(&content_to_add);

        assert_eq!(tree.size(), 16);

        let query_point = [0.78f32, 0.55f32, 0.78f32, 0.55f32];

        let expected = NearestNeighbour {
            distance: 0.819999933,
            item: 13,
        };

        let result = tree.approx_nearest_one::<Manhattan>(&query_point);
        assert_eq!(result, expected);
    }
}
