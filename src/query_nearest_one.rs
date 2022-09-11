use crate::sok::{Axis, Content, LEAF_OFFSET, StemNode};
use crate::KdTree;
use std::ops::Rem;

impl<A: Axis, T: Content, const K: usize, const B: usize> KdTree<A, T, K, B> {
    #[inline]
    pub fn nearest_one<F>(&self, query: &[A; K], distance_fn: &F) -> (A, T)
    where
        F: Fn(&[A; K], &[A; K]) -> A,
    {
        self.nearest_one_recurse(
            query,
            distance_fn,
            self.root_index,
            0,
            T::default(),
            A::infinity(),
        )
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
            F: Fn(&[A; K], &[A; K]) -> A,
    {
        if KdTree::<A, T, K, B>::is_stem_index(curr_node_idx) {
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
            leaf_node.content.iter().for_each(|entry| {
                let dist = distance_fn(query, &entry.point);
                if dist < best_dist {
                    best_dist = dist;
                    best_item = entry.item;
                }
            });
        }

        (best_dist, best_item)
    }
}

#[cfg(test)]
mod tests {
    use crate::distance::squared_euclidean;
    use crate::KdTree;
    use num_traits::Float;
    use rand::Rng;

    #[test]
    fn can_query_nearest_one_item() {
        let mut tree: KdTree<f64, i32, 2, 4> = KdTree::new();

        let content_to_add = [
            ([9f64, 0f64], 9),
            ([4f64, 500f64], 4),
            ([12f64, -300f64], 12),
            ([7f64, 200f64], 7),
            ([13f64, -400f64], 13),
            ([6f64, 300f64], 6),
            ([2f64, 700f64], 2),
            ([14f64, -500f64], 14),
            ([3f64, 600f64], 3),
            ([10f64, -100f64], 10),
            ([16f64, -700f64], 16),
            ([1f64, 800f64], 1),
            ([15f64, -600f64], 15),
            ([5f64, 400f64], 5),
            ([8f64, 100f64], 8),
            ([11f64, -200f64], 11),
        ];

        for (point, item) in content_to_add {
            tree.add(&point, item);
        }

        assert_eq!(tree.size(), 16);

        let query_point = [4.788542420397475f64, -780.5537885546596f64];
        let expected = (6614.609631568035, 16);

        let result = tree.nearest_one(&query_point, &squared_euclidean);
        assert_eq!(result, expected);

        let mut rng = rand::thread_rng();
        for _i in 0..1000 {
            let query_point = [
                rng.gen_range(-10f64..20f64),
                rng.gen_range(-1000f64..1000f64),
            ];
            let expected = linear_search(&content_to_add, &query_point);

            let result = tree.nearest_one(&query_point, &squared_euclidean);

            // if result != expected {
            //     println!("Bad: #{:?}. Query: {:?}, Expected: {:?}, Actual: {:?}", i, &query_point, &expected, &result);
            // }

            assert_eq!(result, expected);
            // println!("Good: {:?}", i);
        }
    }

    fn linear_search(content: &[([f64; 2], i32)], query_point: &[f64; 2]) -> (f64, i32) {
        let mut best_dist: f64 = f64::infinity();
        let mut best_item: i32 = i32::MAX;

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
