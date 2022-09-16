use crate::heap_element::HeapElement;
use crate::sok::{Axis, Content, LEAF_OFFSET};
use crate::KdTree;
use std::collections::BinaryHeap;
use std::ops::Rem;
//use log::debug;

impl<A: Axis, T: Content, const K: usize, const B: usize> KdTree<A, T, K, B> {
    #[inline]
    pub fn within<F>(&self, query: &[A; K], radius: A, distance_fn: &F) -> Vec<(A, T)>
    where
        F: Fn(&[A; K], &[A; K]) -> A,
    {
        let mut matching_items: BinaryHeap<HeapElement<A, T>> = BinaryHeap::new();

        self.within_recurse(
            query,
            radius,
            distance_fn,
            self.root_index,
            0,
            &mut matching_items,
        );

        matching_items
            .into_sorted_vec()
            .into_iter()
            .map(Into::into)
            .collect()
    }

    fn within_recurse<F>(
        &self,
        query: &[A; K],
        radius: A,
        distance_fn: &F,
        curr_node_idx: usize,
        split_dim: usize,
        matching_items: &mut BinaryHeap<HeapElement<A, T>>,
    ) where
        F: Fn(&[A; K], &[A; K]) -> A,
    {
        if KdTree::<A, T, K, B>::is_stem_index(curr_node_idx) {
            let node = &self.stems[curr_node_idx];
            //debug!("stem node: {:?}", &node);

            let child_node_indices = if query[split_dim] < node.split_val {
                [node.left, node.right]
            } else {
                [node.right, node.left]
            };
            let next_split_dim = (split_dim + 1).rem(K);

            for node_idx in child_node_indices {
                let child_node_dist = self.child_dist_to_bounds(query, node_idx, distance_fn);
                if child_node_dist <= radius {
                    self.within_recurse(
                        query,
                        radius,
                        distance_fn,
                        node_idx,
                        next_split_dim,
                        matching_items,
                    );
                }
            }
        } else {
            let leaf_node = &self.leaves[curr_node_idx - LEAF_OFFSET];

            leaf_node
                .content
                .iter()
                .take(leaf_node.size)
                .for_each(|entry| {
                    let distance = distance_fn(query, &entry.point);
                    if distance < radius {
                        matching_items.push(HeapElement {
                            distance,
                            item: entry.item,
                        })
                    }
                });
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::distance::squared_euclidean;
    use crate::KdTree;
    use rand::Rng;
    use std::cmp::Ordering;

    #[test]
    fn can_query_items_within_radius() {
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

        let query_point = [9f64, 0f64];
        let radius = 20000f64;
        let expected = linear_search(&content_to_add, &query_point, radius);

        let result = tree.within(&query_point, radius, &squared_euclidean);
        assert_eq!(result, expected);

        let mut rng = rand::thread_rng();
        for _i in 0..1000 {
            let query_point = [
                rng.gen_range(-10f64..20f64),
                rng.gen_range(-1000f64..1000f64),
            ];
            let radius = 10000f64;
            let expected = linear_search(&content_to_add, &query_point, radius);

            let result = tree.within(&query_point, radius, &squared_euclidean);
            assert_eq!(result, expected);
        }
    }

    fn linear_search(
        content: &[([f64; 2], i32)],
        query_point: &[f64; 2],
        radius: f64,
    ) -> Vec<(f64, i32)> {
        let mut matching_items = vec![];

        for &(p, item) in content {
            let dist = squared_euclidean(query_point, &p);
            if dist < radius {
                matching_items.push((dist, item));
            }
        }

        matching_items.sort_unstable_by(|a, b| {
            let dist_cmp = a.0.partial_cmp(&b.0).unwrap();
            if dist_cmp == Ordering::Equal {
                a.1.cmp(&b.1)
            } else {
                dist_cmp
            }
        });

        matching_items
    }
}
