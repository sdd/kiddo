use crate::float::kdtree::{Axis, KdTree, LeafNode};

use std::collections::BinaryHeap;
use std::ops::Rem;
use az::{Az, Cast};
use crate::types::{Content, Index};

impl<A: Axis, T: Content, const K: usize, const B: usize, IDX: Index<T = IDX>> KdTree<A, T, K, B, IDX> where usize: Cast<IDX> {
    #[inline]
    pub fn best_n_within_into_iter<F>(
        &self,
        query: &[A; K],
        radius: A,
        max_qty: usize,
        distance_fn: &F,
    ) -> impl Iterator<Item = T>
    where
        F: Fn(&[A; K], &[A; K]) -> A,
    {
        let mut best_items: BinaryHeap<T> = BinaryHeap::new();

        unsafe {
            self.best_n_within_recurse(
                query,
                radius,
                max_qty,
                distance_fn,
                self.root_index,
                0,
                &mut best_items,
            );
        }

        best_items.into_iter()
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn best_n_within_recurse<F>(
        &self,
        query: &[A; K],
        radius: A,
        max_qty: usize,
        distance_fn: &F,
        curr_node_idx: IDX,
        split_dim: usize,
        best_items: &mut BinaryHeap<T>,
    ) where
        F: Fn(&[A; K], &[A; K]) -> A,
    {
        if KdTree::<A, T, K, B, IDX>::is_stem_index(curr_node_idx) {
            let node = self.stems.get_unchecked(curr_node_idx.az::<usize>());

            let child_node_indices = if *query.get_unchecked(split_dim) < node.split_val {
                [node.left, node.right]
            } else {
                [node.right, node.left]
            };
            let next_split_dim = (split_dim + 1).rem(K);

            for node_idx in child_node_indices {
                let child_node_dist = self.child_dist_to_bounds(query, node_idx, distance_fn);
                if child_node_dist <= radius {
                    self.best_n_within_recurse(
                        query,
                        radius,
                        max_qty,
                        distance_fn,
                        node_idx,
                        next_split_dim,
                        best_items,
                    );
                }
            }
        } else {
            let leaf_node = self.leaves.get_unchecked((curr_node_idx - IDX::leaf_offset()).az::<usize>());

            Self::process_leaf_node(query, radius, max_qty, distance_fn, best_items, leaf_node);
        }
    }

    // #[inline(never)]
    unsafe fn process_leaf_node<F>(
        query: &[A; K],
        radius: A,
        max_qty: usize,
        distance_fn: &F,
        best_items: &mut BinaryHeap<T>,
        leaf_node: &LeafNode<A, T, K, B, IDX>
    ) where
        F: Fn(&[A; K], &[A; K]) -> A, {
        leaf_node
            .content_points
            .iter()
            .take(leaf_node.size.az::<usize>())
            .map(|entry| {
                distance_fn(query, &entry)
            })
            .enumerate()
            .filter(|(_, distance)| *distance <= radius)
            .for_each(|(idx, _)| {
                Self::get_item_and_add_if_good(max_qty, best_items, leaf_node, idx)
            });
    }

    // #[inline(never)]
    unsafe fn get_item_and_add_if_good(max_qty: usize, best_items: &mut BinaryHeap<T>, leaf_node: &LeafNode<A, T, K, B, IDX>, idx: usize) {
        let item = *leaf_node.content_items.get_unchecked(idx.az::<usize>());
        if best_items.len() < max_qty {
            best_items.push(item);
        } else {
            let mut top = best_items.peek_mut().unwrap();
            if item < *top {
                *top = item;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::Rng;
    use crate::float::distance::squared_euclidean;
    use crate::float::kdtree::KdTree;

    type AX = f64;

    #[test]
    fn can_query_best_n_items_within_radius() {
        let mut tree: KdTree<AX, i32, 2, 4, u32> = KdTree::new();

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

        let query = [9f64, 0f64];
        let radius = 20000f64;
        let max_qty = 3;
        let expected = vec![10, 9, 8];

        let result: Vec<_> = tree
            .best_n_within_into_iter(&query, radius, max_qty, &squared_euclidean)
            .collect();
        assert_eq!(result, expected);

        let max_qty = 2;

        let mut rng = rand::thread_rng();
        for _i in 0..1000 {
            let query = [
                rng.gen_range(-10f64..20f64),
                rng.gen_range(-1000f64..1000f64),
            ];
            let radius = 100000f64;
            let expected = linear_search(&content_to_add, &query, radius, max_qty);
            println!("{}, {}", query[0].to_string(), query[1].to_string());

            let result: Vec<_> = tree
                .best_n_within_into_iter(&query, radius, max_qty, &squared_euclidean)
                .collect();
            assert_eq!(result, expected);
        }
    }

    fn linear_search(
        content: &[([f64; 2], i32)],
        query: &[f64; 2],
        radius: f64,
        max_qty: usize,
    ) -> Vec<i32> {
        let mut best_items = Vec::with_capacity(max_qty);

        for &(p, item) in content {
            let dist = squared_euclidean(query, &p);
            if dist <= radius {
                if best_items.len() < max_qty {
                    best_items.push(item);
                } else {
                    if item < *best_items.last().unwrap() {
                        best_items.pop().unwrap();
                        best_items.push(item);
                    }
                }
            }
            best_items.sort_unstable();
        }
        best_items.reverse();

        best_items
    }
}
