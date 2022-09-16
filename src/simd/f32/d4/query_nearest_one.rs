use std::ops::Rem;
// use crate::simd::f32::d4::distance::squared_euclidean_simd_f32_d4_a_unaligned;
use crate::simd::f32::d4::kdtree::{KdTree, LeafNode, A, K, LEAF_OFFSET, PT, T};

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
        curr_node_idx: usize,
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
            Self::search_content_for_best(query, distance_fn, &mut best_item, &mut best_dist, leaf_node);
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
    use crate::distance::squared_euclidean;
    use crate::simd::f32::d4::kdtree::{KdTree, A, PT, T};
    // use crate::simd::f32::d4::distance::squared_euclidean_simd_f32_d4;
    use rand::Rng;

    #[test]
    fn can_query_nearest_one_item() {
        let mut tree: KdTree = KdTree::new();

        let content_to_add: [(PT, T); 16] = [
            ([9f32, 0f32, 9f32, 0f32], 9),
            ([4f32, 500f32, 4f32, 500f32], 4),
            ([12f32, -300f32, 12f32, -300f32], 12),
            ([7f32, 200f32, 7f32, 200f32], 7),
            ([13f32, -400f32, 13f32, -400f32], 13),
            ([6f32, 300f32, 6f32, 300f32], 6),
            ([2f32, 700f32, 2f32, 700f32], 2),
            ([14f32, -500f32, 14f32, -500f32], 14),
            ([3f32, 600f32, 3f32, 600f32], 3),
            ([10f32, -100f32, 10f32, -100f32], 10),
            ([16f32, -700f32, 16f32, -700f32], 16),
            ([1f32, 800f32, 1f32, 800f32], 1),
            ([15f32, -600f32, 15f32, -600f32], 15),
            ([5f32, 400f32, 5f32, 400f32], 5),
            ([8f32, 100f32, 8f32, 100f32], 8),
            ([11f32, -200f32, 11f32, -200f32], 11),
        ];

        for (point, item) in content_to_add {
            tree.add(&point, item);
        }

        assert_eq!(tree.size(), 16);

        let query_point = [
            4.788542420397475f32,
            -780.5537885546596f32,
            4.788542420397475f32,
            -780.5537885546596f32,
        ];
        let expected = (13229.214, 16);

        let result = tree.nearest_one(&query_point, &squared_euclidean);
        assert_eq!(result, expected);

        let mut rng = rand::thread_rng();
        for i in 0..1000 {
            let query_point = [
                rng.gen_range(-10f32..20f32),
                rng.gen_range(-1000f32..1000f32),
                rng.gen_range(-10f32..20f32),
                rng.gen_range(-1000f32..1000f32),
            ];
            let expected = linear_search(&content_to_add, &query_point);

            let result = tree.nearest_one(&query_point, &squared_euclidean);

            if result.1 != expected.1 || (result.0 - expected.0).abs() > 0.25f32 {
                println!(
                    "Bad: #{:?}. Query: {:?}, Expected: {:?}, Actual: {:?}",
                    i, &query_point, &expected, &result
                );
            }

            assert!((result.0 - expected.0).abs() < 0.25f32);
            assert_eq!(result.1, expected.1);
            // println!("Good: {:?}", i);
        }
    }

    fn linear_search(content: &[(PT, T)], query_point: &PT) -> (A, T) {
        let mut best_dist: A = A::INFINITY;
        let mut best_item: T = T::MAX;

        for &(p, item) in content {
            let dist = squared_euclidean(query_point, &p);
            if dist < best_dist {
                best_item = item;
                best_dist = dist;
            }
        }

        (best_dist, best_item)
    }
}
