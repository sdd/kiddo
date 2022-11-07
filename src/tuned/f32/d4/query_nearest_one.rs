use std::ops::Rem;
// use crate::tuned::f32::d4::distance::squared_euclidean_simd_f32_d4_a_unaligned;
use crate::tuned::f32::d4::kdtree::{KdTree, LeafNode, A, K, LEAF_OFFSET, PT, T, IDX};

impl KdTree {
    #[inline]
    pub fn nearest_one<F>(&self, query: &[A; K], distance_fn: &F) -> (A, T)
    where
        F: Fn(&[A; K], &[A; K]) -> A,
    {
        self.nearest_one_recurse(query, distance_fn, self.root_index, 0, 0, f32::INFINITY)
    }

    fn nearest_one_recurse<F>(
        &self,
        query: &[A; K],
        distance_fn: &F,
        curr_node_idx: IDX,
        split_dim: usize,
        mut best_item: T,
        mut best_dist: A,
    ) -> (A, T)
    where
        F: Fn(&PT, &PT) -> A,
    {
        if KdTree::is_stem_index(curr_node_idx) {
            let node = &self.stems[curr_node_idx];

            let child_node_indices = if query[split_dim] < node.split_val {
                [node.left, node.right]
            } else {
                [node.right, node.left]
            };
            let next_split_dim = (split_dim + 1).rem(K);

            for node_idx in child_node_indices {
                let child_node_dist = self.child_dist_to_bounds(query, node_idx, distance_fn);
                if child_node_dist <= best_dist {
                    let (dist, item) = self.nearest_one_recurse(
                        query,
                        distance_fn,
                        node_idx,
                        next_split_dim,
                        best_item,
                        best_dist,
                    );

                    if dist < best_dist {
                        best_dist = dist;
                        best_item = item;
                    }
                }
            }
        } else {
            let leaf_node = &self.leaves[curr_node_idx - LEAF_OFFSET];

            /*TODO: it might be possible to get a SIMD speedup here
            by rearchitecting LeafNode so that points are moved out
            of content and made contiguous. then, each consecutive pair of points
            can be loaded into an __mm_256. It may be possible to unroll, loading
            several sets of __mm_256 at the same time and then processing them all
            consecutively.*/
            Self::search_content_for_best(
                query,
                distance_fn,
                &mut best_item,
                &mut best_dist,
                leaf_node,
            );
        }

        (best_dist, best_item)
    }

    // #[inline(never)]
    fn search_content_for_best<F>(
        query: &[A; 4],
        distance_fn: &F,
        best_item: &mut T,
        best_dist: &mut A,
        leaf_node: &LeafNode,
    ) where
        F: Fn(&PT, &PT) -> A,
    {
        leaf_node
            .content_points
            .iter()
            .enumerate()
            .take(leaf_node.size)
            .for_each(|(idx, entry)| {
                let dist = distance_fn(query, &entry);
                if dist < *best_dist {
                    *best_dist = dist;
                    *best_item = leaf_node.content_items[idx];
                }
            });
    }
}

#[cfg(test)]
mod tests {
    use crate::tuned::f32::d4::kdtree::{KdTree, A, PT, T};
    use rand::Rng;
    use crate::float::distance::manhattan;

    #[test]
    fn can_query_nearest_one_item() {
        let mut tree: KdTree = KdTree::with_capacity(4);

        let content_to_add: [(PT, T); 16] = [
            ([0.9f32, 0.0f32, 0.9f32, 0.0f32], 9),
            ([0.4f32, 0.5f32, 0.4f32, 0.5f32], 4),
            ([0.12f32, 0.3f32, 0.12f32, 0.3f32], 12),
            ([0.7f32, 0.2f32, 0.7f32, 0.2f32], 7),
            ([0.13f32, 0.4f32, 0.13f32, 0.4f32], 13),
            ([0.6f32, 0.3f32, 0.6f32, 0.3f32], 6),
            ([0.2f32, 0.7f32, 0.2f32, 0.7f32], 2),
            ([0.14f32, 0.5f32, 0.14f32, 0.5f32], 14),
            ([0.3f32, 0.6f32, 0.3f32, 0.6f32], 3),
            ([0.10f32, 0.1f32, 0.10f32, 0.1f32], 10),
            ([0.16f32, 0.7f32, 0.16f32, 0.7f32], 16),
            ([0.1f32, 0.8f32, 0.1f32, 0.8f32], 1),
            ([0.15f32, 0.6f32, 0.15f32, 0.6f32], 15),
            ([0.5f32, 0.4f32, 0.5f32, 0.4f32], 5),
            ([0.8f32, 0.1f32, 0.8f32, 0.1f32], 8),
            ([0.11f32, 0.2f32, 0.11f32, 0.2f32], 11),
        ];

        for (point, item) in content_to_add {
            tree.add(&point, item);
        }
        assert_eq!(tree.size(), 16);

        let query_point = [
            0.78f32,
            0.55f32,
            0.78f32,
            0.55f32,
        ];
        let expected = (0.8599999, 6);
        let result = tree.nearest_one(&query_point, &manhattan);
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

            let result = tree.nearest_one(&query_point, &manhattan);

            assert!((result.0 - expected.0).abs() < 0.25f32);
            assert_eq!(result.1, expected.1);
        }
    }

    fn linear_search(content: &[(PT, T)], query_point: &PT) -> (A, T) {
        let mut best_dist: A = A::INFINITY;
        let mut best_item: T = T::MAX;

        for &(p, item) in content {
            let dist = manhattan(query_point, &p);
            if dist < best_dist {
                best_item = item;
                best_dist = dist;
            }
        }

        (best_dist, best_item)
    }
}
