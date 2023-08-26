use crate::distance_metric::DistanceMetric;
use crate::float::kdtree::Axis;
use crate::immutable::float::kdtree::ImmutableKdTree;
use crate::nearest_neighbour::NearestNeighbour;
use crate::types::Content;
use std::ops::Rem;

impl<A: Axis, T: Content, const K: usize, const B: usize> ImmutableKdTree<A, T, K, B> {
    /// Queries the tree to find the nearest item to the `query` point.
    ///
    /// Faster than querying for nearest_n(point, 1, ...) due
    /// to not needing to allocate memory or maintain sorted results.
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
    /// let nearest = tree.nearest_one::<SquaredEuclidean>(&[1.0, 2.0, 5.1]);
    ///
    /// assert!((nearest.distance - 0.01f64).abs() < f64::EPSILON);
    /// assert_eq!(nearest.item, 0);
    /// ```
    #[inline]
    pub fn nearest_one<D>(&self, query: &[A; K]) -> NearestNeighbour<A, T>
    where
        D: DistanceMetric<A, K>,
    {
        let mut off = [A::zero(); K];
        self.nearest_one_recurse::<D>(
            query,
            1,
            0,
            NearestNeighbour {
                distance: A::max_value(),
                item: T::zero(),
            },
            &mut off,
            A::zero(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn nearest_one_recurse<D>(
        &self,
        query: &[A; K],
        stem_idx: usize,
        split_dim: usize,
        mut nearest: NearestNeighbour<A, T>,
        off: &mut [A; K],
        rd: A,
    ) -> NearestNeighbour<A, T>
    where
        D: DistanceMetric<A, K>,
    {
        if stem_idx >= self.stems.len() {
            self.search_leaf_for_nearest::<D>(query, &mut nearest, stem_idx - self.stems.len());

            return nearest;
        }

        let left_child_idx = stem_idx << 1;
        self.prefetch_stems(left_child_idx);

        let val = *unsafe { self.stems.get_unchecked(stem_idx) };
        // let val = self.stems[stem_idx];

        let mut rd = rd;
        let old_off = off[split_dim];
        // let new_off = (query[split_dim] * query[split_dim]) - (val * val);
        let new_off = query[split_dim].saturating_dist(val);
        // let new_off = query[split_dim] - val;

        let is_left_child = usize::from(*unsafe { query.get_unchecked(split_dim) } < val);
        // let is_left_child = usize::from(query[split_dim] < val);

        let closer_node_idx = left_child_idx + (1 - is_left_child);
        let further_node_idx = left_child_idx + is_left_child;

        let next_split_dim = (split_dim + 1).rem(K);

        let nearest_neighbour =
            self.nearest_one_recurse::<D>(query, closer_node_idx, next_split_dim, nearest, off, rd);

        if nearest_neighbour < nearest {
            nearest = nearest_neighbour;
        }

        rd = Axis::rd_update(rd, D::dist1(new_off, old_off));

        if rd <= nearest.distance {
            off[split_dim] = new_off;
            let result = self.nearest_one_recurse::<D>(
                query,
                further_node_idx,
                next_split_dim,
                nearest,
                off,
                rd,
            );
            off[split_dim] = old_off;

            if result < nearest {
                nearest = result;
            }
        }

        nearest
    }

    fn search_leaf_for_nearest<D>(
        &self,
        query: &[A; K],
        nearest: &mut NearestNeighbour<A, T>,
        leaf_idx: usize,
    ) where
        D: DistanceMetric<A, K>,
    {
        let leaf_node = unsafe { self.leaves.get_unchecked(leaf_idx) };
        // let leaf_node = &self.leaves[leaf_idx];

        leaf_node
            .content_points
            .iter()
            .enumerate()
            .take(leaf_node.size)
            .for_each(|(idx, entry)| {
                let dist = D::dist(query, entry);
                if dist < nearest.distance {
                    nearest.distance = dist;
                    nearest.item = unsafe { *leaf_node.content_items.get_unchecked(idx) };
                    // nearest.item = leaf_node.content_items[idx]
                }
            });
    }
}

#[cfg(test)]
mod tests {
    use crate::distance_metric::DistanceMetric;
    use crate::float::distance::Manhattan;
    use crate::float::kdtree::Axis;
    use crate::immutable::float::kdtree::ImmutableKdTree;
    use crate::nearest_neighbour::NearestNeighbour;
    use rand::Rng;

    type AX = f32;

    #[test]
    fn can_query_nearest_one_item() {
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

        let result = tree.nearest_one::<Manhattan>(&query_point);
        assert_eq!(result, expected);

        let mut rng = rand::thread_rng();
        for _i in 0..1000 {
            let query_point = [
                rng.gen_range(0f32..1f32),
                rng.gen_range(0f32..1f32),
                rng.gen_range(0f32..1f32),
                rng.gen_range(0f32..1f32),
            ];
            let expected = linear_search(&content_to_add, &query_point);

            let result = tree.nearest_one::<Manhattan>(&query_point);

            // println!("#{}: {} == {}", _i, result.0, expected.0);
            assert_eq!(result.distance, expected.distance);
        }
    }

    #[test]
    fn can_query_nearest_one_item_large_scale() {
        const TREE_SIZE: usize = 100_000;
        const NUM_QUERIES: usize = 100;

        let content_to_add: Vec<[f32; 4]> =
            (0..TREE_SIZE).map(|_| rand::random::<[f32; 4]>()).collect();

        let tree: ImmutableKdTree<AX, u32, 4, 32> =
            ImmutableKdTree::new_from_slice(&content_to_add);

        assert_eq!(tree.size(), TREE_SIZE);

        let query_points: Vec<[f32; 4]> = (0..NUM_QUERIES)
            .map(|_| rand::random::<[f32; 4]>())
            .collect();

        for (_i, query_point) in query_points.iter().enumerate() {
            let expected = linear_search(&content_to_add, &query_point);

            let result = tree.nearest_one::<Manhattan>(&query_point);

            // println!("#{}: {} == {}", _i, result.0, expected.0);
            assert_eq!(result.distance, expected.distance);
            assert_eq!(result.item as usize, expected.item);
        }
    }

    fn linear_search<A: Axis, const K: usize>(
        content: &[[A; K]],
        query_point: &[A; K],
    ) -> NearestNeighbour<A, usize> {
        let mut best_dist: A = A::infinity();
        let mut best_item: usize = usize::MAX;

        for (idx, p) in content.iter().enumerate() {
            let dist = Manhattan::dist(query_point, p);
            if dist < best_dist {
                best_item = idx;
                best_dist = dist;
            }
        }

        NearestNeighbour {
            distance: best_dist,
            item: best_item,
        }
    }
}
