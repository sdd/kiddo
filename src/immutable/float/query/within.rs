use crate::distance_metric::DistanceMetric;
use crate::float::kdtree::Axis;
use crate::immutable::float::kdtree::ImmutableKdTree;
use crate::nearest_neighbour::NearestNeighbour;
use crate::types::Content;

use crate::generate_immutable_within;

macro_rules! generate_immutable_float_within {
    ($doctest_build_tree:tt) => {
        generate_immutable_within!((
            "Finds all elements within `dist` of `query`, using the specified
distance metric function.

Results are returned sorted nearest-first

# Examples

```rust
    use kiddo::immutable::float::kdtree::ImmutableKdTree;
    use kiddo::float::distance::SquaredEuclidean;
    ",
            $doctest_build_tree,
            "

    let within = tree.within::<SquaredEuclidean>(&[1.0, 2.0, 5.0], 10f64);

    assert_eq!(within.len(), 2);
```"
        ));
    };
}

impl<A: Axis, T: Content, const K: usize, const B: usize> ImmutableKdTree<A, T, K, B> {
    generate_immutable_float_within!(
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
    generate_immutable_float_within!(
        "use std::fs::File;
use memmap::MmapOptions;

let mmap = unsafe { MmapOptions::new().map(&File::open(\"./examples/immutable-doctest-tree.rkyv\").unwrap()).unwrap() };
let tree = unsafe { rkyv::archived_root::<ImmutableKdTree<f64, u32, 3, 32>>(&mmap) };"
    );
}

/*impl<A: Axis, T: Content, const K: usize, const B: usize> ImmutableKdTree<A, T, K, B> {
    /// Finds all elements within `dist` of `query`, using the specified
    /// distance metric function.
    ///
    /// Results are returned sorted nearest-first
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kiddo::immutable::float::kdtree::ImmutableKdTree;
    /// use kiddo::float::distance::SquaredEuclidean;
    ///
    /// let content: Vec<[f64; 3]> = vec!(
    ///     [1.0, 2.0, 5.0],
    ///     [2.0, 3.0, 6.0],
    ///     [200.0, 300.0, 600.0],
    /// );
    ///
    /// let mut tree: ImmutableKdTree<f64, u32, 3, 32> = ImmutableKdTree::new_from_slice(&content);
    ///
    /// let within = tree.within::<SquaredEuclidean>(&[1.0, 2.0, 5.0], 10f64);
    ///
    /// assert_eq!(within.len(), 2);
    /// ```
    #[inline]
    pub fn within<D>(&self, query: &[A; K], dist: A) -> Vec<NearestNeighbour<A, T>>
    where
        D: DistanceMetric<A, K>,
    {
        let mut off = [A::zero(); K];
        let mut matching_items: BinaryHeap<NearestNeighbour<A, T>> = BinaryHeap::new();

        self.within_recurse::<D>(query, dist, 1, 0, &mut matching_items, &mut off, A::zero());

        matching_items.into_sorted_vec()
    }

    #[allow(clippy::too_many_arguments)]
    fn within_recurse<D>(
        &self,
        query: &[A; K],
        radius: A,
        stem_idx: usize,
        split_dim: usize,
        matching_items: &mut BinaryHeap<NearestNeighbour<A, T>>,
        off: &mut [A; K],
        rd: A,
    ) where
        D: DistanceMetric<A, K>,
    {
        if stem_idx >= self.stems.len() {
            let leaf_node = &self.leaves[stem_idx - self.stems.len()];

            leaf_node
                .content_points
                .iter()
                .enumerate()
                .take(leaf_node.size)
                .for_each(|(idx, entry)| {
                    let distance = D::dist(query, entry);

                    if distance < radius {
                        matching_items.push(NearestNeighbour {
                            distance,
                            item: *unsafe { leaf_node.content_items.get_unchecked(idx) },
                        })
                    }
                });

            return;
        }

        let left_child_idx = stem_idx << 1;
        self.prefetch_stems(left_child_idx);

        let val = *unsafe { self.stems.get_unchecked(stem_idx) };
        // let val = self.stems[stem_idx];

        let mut rd = rd;
        let old_off = off[split_dim];
        let new_off = query[split_dim].saturating_dist(val);

        let is_left_child = usize::from(*unsafe { query.get_unchecked(split_dim) } < val);
        // let is_left_child = usize::from(query[split_dim] < val);

        let closer_node_idx = left_child_idx + (1 - is_left_child);
        let further_node_idx = left_child_idx + is_left_child;

        let next_split_dim = (split_dim + 1).rem(K);

        self.within_recurse::<D>(
            query,
            radius,
            closer_node_idx,
            next_split_dim,
            matching_items,
            off,
            rd,
        );

        rd = Axis::rd_update(rd, D::dist1(new_off, old_off));

        if rd <= radius {
            off[split_dim] = new_off;
            self.within_recurse::<D>(
                query,
                radius,
                further_node_idx,
                next_split_dim,
                matching_items,
                off,
                rd,
            );
            off[split_dim] = old_off;
        }
    }
}*/

#[cfg(test)]
mod tests {
    use crate::distance_metric::DistanceMetric;
    use crate::float::distance::Manhattan;
    use crate::float::kdtree::Axis;
    use crate::immutable::float::kdtree::ImmutableKdTree;
    use rand::Rng;
    use std::cmp::Ordering;

    type AX = f32;

    #[test]
    fn can_query_items_within_radius() {
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

        let radius = 0.2;
        let expected = linear_search(&content_to_add, &query_point, radius);

        let mut result: Vec<_> = tree
            .within::<Manhattan>(&query_point, radius)
            .into_iter()
            .map(|n| (n.distance, n.item))
            .collect();
        stabilize_sort(&mut result);
        assert_eq!(result, expected);

        let mut rng = rand::thread_rng();
        for _i in 0..1000 {
            let query_point = [
                rng.gen_range(0f32..1f32),
                rng.gen_range(0f32..1f32),
                rng.gen_range(0f32..1f32),
                rng.gen_range(0f32..1f32),
            ];
            let radius: f32 = 2.0;
            let expected = linear_search(&content_to_add, &query_point, radius);

            let mut result: Vec<_> = tree
                .within::<Manhattan>(&query_point, radius)
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

        let content_to_add: Vec<[f32; 4]> =
            (0..TREE_SIZE).map(|_| rand::random::<[f32; 4]>()).collect();

        let tree: ImmutableKdTree<AX, u32, 4, 32> =
            ImmutableKdTree::new_from_slice(&content_to_add);
        assert_eq!(tree.size(), TREE_SIZE);

        let query_points: Vec<[f32; 4]> = (0..NUM_QUERIES)
            .map(|_| rand::random::<[f32; 4]>())
            .collect();

        for query_point in query_points {
            let expected = linear_search(&content_to_add, &query_point, RADIUS);

            let result: Vec<_> = tree
                .within::<Manhattan>(&query_point, RADIUS)
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();

            assert_eq!(result, expected);
        }
    }

    fn linear_search<A: Axis, const K: usize>(
        content: &[[A; K]],
        query_point: &[A; K],
        radius: A,
    ) -> Vec<(A, u32)> {
        let mut matching_items = vec![];

        for (idx, p) in content.iter().enumerate() {
            let dist = Manhattan::dist(query_point, &p);
            if dist < radius {
                matching_items.push((dist, idx as u32));
            }
        }

        stabilize_sort(&mut matching_items);

        matching_items
    }

    fn stabilize_sort<A: Axis>(matching_items: &mut Vec<(A, u32)>) {
        matching_items.sort_unstable_by(|a, b| {
            let dist_cmp = a.0.partial_cmp(&b.0).unwrap();
            if dist_cmp == Ordering::Equal {
                a.1.cmp(&b.1)
            } else {
                dist_cmp
            }
        });
    }
}
