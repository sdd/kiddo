use crate::tuned::u16::d4::kdtree::{KdTree, LeafNode, A, IDX, K, LEAF_OFFSET, PT, T};
use std::ops::Rem;

impl KdTree {
    #[inline]
    pub fn nearest_one<F>(&self, query: &[A; K], distance_fn: &F) -> (A, T)
    where
        F: Fn(&[A; K], &[A; K]) -> A,
    {
        unsafe { self.nearest_one_recurse(query, distance_fn, self.root_index, 0, 0, A::MAX) }
    }

    #[inline]
    unsafe fn nearest_one_recurse<F>(
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
            let node = &self.stems.get_unchecked(curr_node_idx as usize);

            let child_node_indices = if *query.get_unchecked(split_dim) < node.split_val {
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
            let leaf_node = self.leaves.get_unchecked((curr_node_idx - LEAF_OFFSET) as usize);

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
            .take(leaf_node.size as usize)
            .for_each(|(idx, entry)| {
                let dist = distance_fn(query, &entry);
                if dist < *best_dist {
                    *best_dist = dist;
                    *best_item = unsafe { *leaf_node.content_items.get_unchecked(idx) };
                }
            });
    }
}

#[cfg(test)]
mod tests {
    use fixed::types::extra::U16;
    use fixed::FixedU16;
    use num_traits::real::Real;
    use crate::tuned::u16::d4::distance::squared_euclidean;
    use crate::tuned::u16::d4::kdtree::{KdTree, A, PT, T};
    use rand::Rng;

    fn n(num: f32) -> FixedU16<U16> {
        FixedU16::<U16>::from_num(num)
    }

    #[test]
    fn can_query_nearest_one_item() {
        let mut tree: KdTree = KdTree::new();

        let content_to_add: [(PT, T); 16] = [
            ([n(0.9f32), n(0.0f32), n(0.9f32), n(0.0f32)], 9),
            ([n(0.4f32), n(0.5f32), n(0.4f32), n(0.5f32)], 4),
            ([n(0.12f32), n(0.3f32), n(0.12f32), n(0.3f32)], 12),
            ([n(0.7f32), n(0.2f32), n(0.7f32), n(0.2f32)], 7),
            ([n(0.13f32), n(0.4f32), n(0.13f32), n(0.4f32)], 13),
            ([n(0.6f32), n(0.3f32), n(0.6f32), n(0.3f32)], 6),
            ([n(0.2f32), n(0.7f32), n(0.2f32), n(0.7f32)], 2),
            ([n(0.14f32), n(0.5f32), n(0.14f32), n(0.5f32)], 14),
            ([n(0.3f32), n(0.6f32), n(0.3f32), n(0.6f32)], 3),
            ([n(0.10f32), n(0.1f32), n(0.10f32), n(0.1f32)], 10),
            ([n(0.16f32), n(0.7f32), n(0.16f32), n(0.7f32)], 16),
            ([n(0.1f32), n(0.8f32), n(0.1f32), n(0.8f32)], 1),
            ([n(0.15f32), n(0.6f32), n(0.15f32), n(0.6f32)], 15),
            ([n(0.5f32), n(0.4f32), n(0.5f32), n(0.4f32)], 5),
            ([n(0.8f32), n(0.1f32), n(0.8f32), n(0.1f32)], 8),
            ([n(0.11f32), n(0.2f32), n(0.11f32), n(0.2f32)], 11),
        ];

        for (point, item) in content_to_add {
            tree.add(&point, item);
        }

        assert_eq!(tree.size(), 16);

        let query_point = [
            n(0.78f32),
            n(0.55f32),
            n(0.78f32),
            n(0.55f32),
        ];
        let expected = (n(0.1898), 6);

        let result = tree.nearest_one(&query_point, &squared_euclidean);
        assert_eq!(result, expected);

        let mut rng = rand::thread_rng();
        for i in 0..1000 {
            let query_point = [
                n(rng.gen_range(0f32..1f32)),
                n(rng.gen_range(0f32..1f32)),
                n(rng.gen_range(0f32..1f32)),
                n(rng.gen_range(0f32..1f32)),
            ];
            let expected = linear_search(&content_to_add, &query_point);

            let result = tree.nearest_one(&query_point, &squared_euclidean);

            if result.1 != expected.1 || result.0 != expected.0 {
                println!(
                    "Bad: #{:?}. Query: {:?}, Expected: {:?}, Actual: {:?}",
                    i, &query_point, &expected, &result
                );
            }

            assert_eq!(result.0, expected.0);
            assert_eq!(result.1, expected.1);
            // println!("Good: {:?}", i);
        }
    }

    fn linear_search(content: &[(PT, T)], query_point: &PT) -> (A, T) {
        let mut best_dist: A = A::MAX;
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
