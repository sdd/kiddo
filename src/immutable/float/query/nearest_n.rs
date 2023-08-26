use crate::distance_metric::DistanceMetric;
use crate::float::kdtree::Axis;
use crate::immutable::float::kdtree::ImmutableKdTree;
use crate::nearest_neighbour::NearestNeighbour;
use crate::types::Content;
use std::collections::BinaryHeap;
use std::ops::Rem;

impl<A: Axis, T: Content, const K: usize, const B: usize> ImmutableKdTree<A, T, K, B> {
    /// Finds the nearest `qty` elements to `query`, using the specified
    /// distance metric function.
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
    /// let nearest: Vec<_> = tree.nearest_n::<SquaredEuclidean>(&[1.0, 2.0, 5.1], 1);
    ///
    /// assert_eq!(nearest.len(), 1);
    /// assert!((nearest[0].distance - 0.01f64).abs() < f64::EPSILON);
    /// assert_eq!(nearest[0].item, 0);
    /// ```
    #[inline]
    pub fn nearest_n<D>(&self, query: &[A; K], qty: usize) -> Vec<NearestNeighbour<A, T>>
    where
        D: DistanceMetric<A, K>,
    {
        let mut off = [A::zero(); K];
        let mut result: BinaryHeap<NearestNeighbour<A, T>> = BinaryHeap::with_capacity(qty);

        self.nearest_n_recurse::<D>(query, 1, 0, &mut result, &mut off, A::zero());

        result.into_sorted_vec()
    }

    fn nearest_n_recurse<D>(
        &self,
        query: &[A; K],
        stem_idx: usize,
        split_dim: usize,
        results: &mut BinaryHeap<NearestNeighbour<A, T>>,
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
                .take(leaf_node.size)
                .enumerate()
                .for_each(|(idx, entry)| {
                    let distance: A = D::dist(query, entry);
                    if Self::dist_belongs_in_heap(distance, results) {
                        let item = unsafe { *leaf_node.content_items.get_unchecked(idx) };
                        let element = NearestNeighbour { distance, item };
                        if results.len() < results.capacity() {
                            results.push(element)
                        } else {
                            let mut top = results.peek_mut().unwrap();
                            if element.distance < top.distance {
                                *top = element;
                            }
                        }
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
        let new_off = query[split_dim] - val;

        let is_left_child = usize::from(*unsafe { query.get_unchecked(split_dim) } < val);
        // let is_left_child = usize::from(query[split_dim] < val);

        let closer_node_idx = left_child_idx + (1 - is_left_child);
        let further_node_idx = left_child_idx + is_left_child;

        let next_split_dim = (split_dim + 1).rem(K);

        self.nearest_n_recurse::<D>(query, closer_node_idx, next_split_dim, results, off, rd);

        // TODO: switch from dist_fn to a dist trait that can apply to 1D as well as KD
        //       so that updating rd is not hardcoded to sq euclidean
        rd = rd + new_off * new_off - old_off * old_off;
        if Self::dist_belongs_in_heap(rd, results) {
            off[split_dim] = new_off;
            self.nearest_n_recurse::<D>(query, further_node_idx, next_split_dim, results, off, rd);
            off[split_dim] = old_off;
        }
    }

    fn dist_belongs_in_heap(dist: A, heap: &BinaryHeap<NearestNeighbour<A, T>>) -> bool {
        heap.is_empty() || dist < heap.peek().unwrap().distance || heap.len() < heap.capacity()
    }
}

#[cfg(test)]
mod tests {
    use crate::distance_metric::DistanceMetric;
    use crate::float::distance::SquaredEuclidean;
    use crate::float::kdtree::Axis;
    use crate::immutable::float::kdtree::ImmutableKdTree;
    use rand::Rng;

    type AX = f32;

    #[test]
    fn can_query_nearest_n_item() {
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

        let expected = vec![(0.17569996, 5), (0.19139998, 13), (0.24420004, 3)];

        let result: Vec<_> = tree
            .nearest_n::<SquaredEuclidean>(&query_point, 3)
            .into_iter()
            .map(|n| (n.distance, n.item))
            .collect();
        assert_eq!(result, expected);

        let qty = 10;
        let mut rng = rand::thread_rng();
        for _i in 0..1000 {
            let query_point = [
                rng.gen_range(0f32..1f32),
                rng.gen_range(0f32..1f32),
                rng.gen_range(0f32..1f32),
                rng.gen_range(0f32..1f32),
            ];
            let expected = linear_search(&content_to_add, qty, &query_point);

            let result: Vec<_> = tree
                .nearest_n::<SquaredEuclidean>(&query_point, qty)
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();

            let result_dists: Vec<_> = result.iter().map(|(d, _)| d).collect();
            let expected_dists: Vec<_> = expected.iter().map(|(d, _)| d).collect();

            assert_eq!(result_dists, expected_dists);
        }
    }

    #[test]
    fn can_query_nearest_10_items_large_scale() {
        const TREE_SIZE: usize = 100_000;
        const NUM_QUERIES: usize = 100;
        const N: usize = 10;

        let content_to_add: Vec<[f32; 4]> =
            (0..TREE_SIZE).map(|_| rand::random::<[f32; 4]>()).collect();

        let tree: ImmutableKdTree<AX, u32, 4, 32> =
            ImmutableKdTree::new_from_slice(&content_to_add);

        assert_eq!(tree.size(), TREE_SIZE);

        let query_points: Vec<[f32; 4]> = (0..NUM_QUERIES)
            .map(|_| rand::random::<[f32; 4]>())
            .collect();

        for query_point in query_points {
            let expected = linear_search(&content_to_add, N, &query_point);

            let result: Vec<_> = tree
                .nearest_n::<SquaredEuclidean>(&query_point, N)
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();

            let result_dists: Vec<_> = result.iter().map(|(d, _)| d).collect();
            let expected_dists: Vec<_> = expected.iter().map(|(d, _)| d).collect();

            assert_eq!(result_dists, expected_dists);
        }
    }

    fn linear_search<A: Axis, const K: usize>(
        content: &[[A; K]],
        qty: usize,
        query_point: &[A; K],
    ) -> Vec<(A, usize)> {
        let mut results = vec![];

        for (idx, p) in content.iter().enumerate() {
            let dist = SquaredEuclidean::dist(query_point, &p);
            if results.len() < qty {
                results.push((dist, idx));
                results.sort_by(|(a_dist, _), (b_dist, _)| a_dist.partial_cmp(&b_dist).unwrap());
            } else if dist < results[qty - 1].0 {
                results[qty - 1] = (dist, idx);
                results.sort_by(|(a_dist, _), (b_dist, _)| a_dist.partial_cmp(&b_dist).unwrap());
            }
        }

        results
    }
}
