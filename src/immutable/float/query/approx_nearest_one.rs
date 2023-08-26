use crate::float::kdtree::Axis;
use crate::immutable::float::kdtree::ImmutableKdTree;
use crate::types::Content;

impl<A: Axis, T: Content, const K: usize, const B: usize> ImmutableKdTree<A, T, K, B> {
    /// Queries the tree to find the nearest element to `query`, using the specified
    /// distance metric function.
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
    /// assert!((nearest.0 - 0.01f64).abs() < f64::EPSILON);
    /// assert_eq!(nearest.1, 0);
    /// ```
    #[inline]
    pub fn approx_nearest_one<F>(&self, query: &[A; K], distance_fn: &F) -> (A, T)
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

        (best_dist, best_item)
    }
}

#[cfg(test)]
mod tests {
    use crate::distance_metric::DistanceMetric;
    use crate::float::distance::Manhattan;
    use crate::float::kdtree::Axis;
    use crate::immutable::float::kdtree::ImmutableKdTree;
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

        let expected = (0.819999933, 13);

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
            assert_eq!(result.0, expected.0);
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
            assert_eq!(result.0, expected.0);
            assert_eq!(result.1 as usize, expected.1);
        }
    }

    fn linear_search<A: Axis, const K: usize>(
        content: &[[A; K]],
        query_point: &[A; K],
    ) -> (A, usize) {
        let mut best_dist: A = A::infinity();
        let mut best_item: usize = usize::MAX;

        for (idx, p) in content.iter().enumerate() {
            let dist = Manhattan::dist(query_point, p);
            if dist < best_dist {
                best_item = idx;
                best_dist = dist;
            }
        }

        (best_dist, best_item)
    }
}
