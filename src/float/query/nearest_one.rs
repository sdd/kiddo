use crate::neighbour::Neighbour;
use crate::float::kdtree::{Axis, KdTree, LeafNode};
use crate::types::{Content, Index};
use az::{Az, Cast};
use std::ops::Rem;
use crate::float::distance::DistanceMetric;

impl<A: Axis, T: Content, const K: usize, const B: usize, IDX: Index<T = IDX>>
    KdTree<A, T, K, B, IDX>
where
    usize: Cast<IDX>,
{
    /// Queries the tree to find the nearest element to `query`, using the specified
    /// distance metric function.
    ///
    /// Faster than querying for nearest_n(point, 1, ...) due
    /// to not needing to allocate memory or maintain sorted results.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kiddo::float::kdtree::KdTree;
    /// use kiddo::distance::squared_euclidean;
    /// use kiddo::float::distance::SquaredEuclidean;
    ///
    /// let mut tree: KdTree<f64, u32, 3, 32, u32> = KdTree::new();
    ///
    /// tree.add(&[1.0, 2.0, 5.0], 100);
    /// tree.add(&[2.0, 3.0, 6.0], 101);
    ///
    /// let nearest = tree.nearest_one::<SquaredEuclidean>(&[1.0, 2.0, 5.1]).unwrap();
    ///
    /// assert!((nearest.distance - 0.01f64).abs() < f64::EPSILON);
    /// assert_eq!(nearest.item, 100);
    /// ```
    #[inline]
    pub fn nearest_one<D>(&self, query: &[A; K]) -> Option<Neighbour<A, T>>
    where
        D: DistanceMetric<A, K>
    {
        if self.size() == T::zero() {
            return None;
        }
        let mut off = [A::zero(); K];
        Some(unsafe {
            self.nearest_one_recurse::<D>(
                query,
                self.root_index,
                0,
                Neighbour { item:  T::zero(), distance: A::max_value() },
                &mut off,
                A::zero(),
            )
        })
    }

    #[inline]
    unsafe fn nearest_one_recurse<D>(
        &self,
        query: &[A; K],
        curr_node_idx: IDX,
        split_dim: usize,
        mut best_neighbour: Neighbour<A, T>,
        off: &mut [A; K],
        rd: A,
    ) -> Neighbour<A, T>
    where
        D: DistanceMetric<A, K>
    {
        if KdTree::<A, T, K, B, IDX>::is_stem_index(curr_node_idx) {
            let node = &self.stems.get_unchecked(curr_node_idx.az::<usize>());

            let mut rd = rd;
            let old_off = off[split_dim];
            let new_off = query[split_dim] - node.split_val;

            let [closer_node_idx, further_node_idx] =
                if *query.get_unchecked(split_dim) < node.split_val {
                    [node.left, node.right]
                } else {
                    [node.right, node.left]
                };
            let next_split_dim = (split_dim + 1).rem(K);

            let neighbour = self.nearest_one_recurse::<D>(
                query,
                closer_node_idx,
                next_split_dim,
                best_neighbour,
                off,
                rd,
            );

            if neighbour.distance < best_neighbour.distance {
                best_neighbour = neighbour
            }

            rd = rd + D::dist1(old_off, new_off);

            if rd <= best_neighbour.distance {
                off[split_dim] = new_off;
                let neighbour = self.nearest_one_recurse::<D>(
                    query,
                    further_node_idx,
                    next_split_dim,
                    best_neighbour,
                    off,
                    rd,
                );
                off[split_dim] = old_off;

                if neighbour.distance < best_neighbour.distance {
                    best_neighbour = neighbour
                }
            }
        } else {
            let leaf_node = self
                .leaves
                .get_unchecked((curr_node_idx - IDX::leaf_offset()).az::<usize>());

            Self::search_content_for_best::<D>(
                query,
                &mut best_neighbour,
                leaf_node,
            );
        }

        best_neighbour
    }

    fn search_content_for_best<D>(
        query: &[A; K],
        best_neighbour: &mut Neighbour<A, T>,
        leaf_node: &LeafNode<A, T, K, B, IDX>,
    ) where
        D: DistanceMetric<A, K>
    {
        leaf_node
            .content_points
            .iter()
            .enumerate()
            .take(leaf_node.size.az::<usize>())
            .for_each(|(idx, entry)| {
                let dist = D::dist(query, entry);
                if dist < best_neighbour.distance {
                    best_neighbour.distance = dist;
                    best_neighbour.item = unsafe { *leaf_node.content_items.get_unchecked(idx) };
                }
            });
    }
}

#[cfg(test)]
mod tests {
    use crate::float::distance::Manhattan;
    use crate::float::kdtree::{Axis, KdTree};
    use crate::neighbour::Neighbour;
    use rand::Rng;
    use crate::float::distance::DistanceMetric;

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

        let expected = Neighbour { distance: 0.819999933, item: 5 };

        let result = tree.nearest_one::<Manhattan>(&query_point).unwrap();
        assert_eq!(result.distance, expected.distance);
        assert_eq!(result.item, expected.item);

        let mut rng = rand::thread_rng();
        for _i in 0..1000 {
            let query_point = [
                rng.gen_range(0f32..1f32),
                rng.gen_range(0f32..1f32),
                rng.gen_range(0f32..1f32),
                rng.gen_range(0f32..1f32),
            ];
            let expected = linear_search(&content_to_add, &query_point);

            let result = tree.nearest_one::<Manhattan>(&query_point).unwrap();

            assert_eq!(result.distance, expected.0);
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
        assert_eq!(tree.size(), TREE_SIZE as u32);

        let query_points: Vec<[f32; 4]> = (0..NUM_QUERIES)
            .map(|_| rand::random::<[f32; 4]>())
            .collect();

        for query_point in query_points {
            let expected = linear_search(&content_to_add, &query_point);

            let result = tree.nearest_one::<Manhattan>(&query_point).unwrap();

            assert_eq!(result.distance, expected.0);
            assert_eq!(result.item, expected.1);
        }
    }

    fn linear_search<A: Axis, const K: usize>(
        content: &[([A; K], u32)],
        query_point: &[A; K],
    ) -> (A, u32) {
        let mut best_dist: A = A::infinity();
        let mut best_item: u32 = u32::MAX;

        for &(p, item) in content {
            let dist = Manhattan::dist(query_point, &p);
            if dist < best_dist {
                best_item = item;
                best_dist = dist;
            }
        }

        (best_dist, best_item)
    }
}
