use az::{Az, Cast};
use std::collections::BinaryHeap;
use std::ops::Rem;

use crate::fixed::kdtree::{Axis, KdTree, LeafNode};
use crate::types::{Content, Index};

impl<A: Axis, T: Content, const K: usize, const B: usize, IDX: Index<T = IDX>>
    KdTree<A, T, K, B, IDX>
where
    usize: Cast<IDX>,
{
    #[inline]
    pub fn best_n_within<F>(
        &self,
        query: &[A; K],
        radius: A,
        max_qty: usize,
        distance_fn: &F,
    ) -> impl Iterator<Item = T>
    where
        F: Fn(&[A; K], &[A; K]) -> A,
    {
        // TODO: switch to https://docs.rs/min-max-heap/1.3.0/min_max_heap/struct.MinMaxHeap.html
        let mut best_items: BinaryHeap<T> = BinaryHeap::new();

        self.best_n_within_recurse(
            query,
            radius,
            max_qty,
            distance_fn,
            self.root_index,
            0,
            &mut best_items,
        );

        best_items.into_iter()
    }

    #[allow(clippy::too_many_arguments)]
    fn best_n_within_recurse<F>(
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
            let node = unsafe { self.stems.get_unchecked(curr_node_idx.az::<usize>()) };

            let child_node_indices = if unsafe { *query.get_unchecked(split_dim) } < node.split_val {
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
            let leaf_node = unsafe {
                self
                    .leaves
                    .get_unchecked((curr_node_idx - IDX::leaf_offset()).az::<usize>())
            };

            Self::process_leaf_node(query, radius, max_qty, distance_fn, best_items, leaf_node);
        }
    }

    fn process_leaf_node<F>(
        query: &[A; K],
        radius: A,
        max_qty: usize,
        distance_fn: &F,
        best_items: &mut BinaryHeap<T>,
        leaf_node: &LeafNode<A, T, K, B, IDX>,
    ) where
        F: Fn(&[A; K], &[A; K]) -> A,
    {
        leaf_node
            .content_points
            .iter()
            .take(leaf_node.size.az::<usize>())
            .map(|entry| distance_fn(query, entry))
            .enumerate()
            .filter(|(_, distance)| *distance <= radius)
            .for_each(|(idx, _)| {
                unsafe {
                    Self::get_item_and_add_if_good(max_qty, best_items, leaf_node, idx)
                }
            });
    }

    unsafe fn get_item_and_add_if_good(
        max_qty: usize,
        best_items: &mut BinaryHeap<T>,
        leaf_node: &LeafNode<A, T, K, B, IDX>,
        idx: usize,
    ) {
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
    use crate::fixed::distance::squared_euclidean;
    use crate::fixed::kdtree::{Axis, KdTree};
    use fixed::types::extra::U14;
    use fixed::FixedU16;
    use rand::Rng;

    type FXD = FixedU16<U14>;

    fn n(num: f32) -> FXD {
        FXD::from_num(num)
    }

    #[test]
    fn can_query_best_n_items_within_radius() {
        let mut tree: KdTree<FXD, u32, 2, 4, u32> = KdTree::new();

        let content_to_add: [([FXD; 2], u32); 16] = [
            ([n(0.9f32), n(0.0f32)], 9),
            ([n(0.4f32), n(0.5f32)], 4),
            ([n(0.12f32), n(0.3f32)], 12),
            ([n(0.7f32), n(0.2f32)], 7),
            ([n(0.13f32), n(0.4f32)], 13),
            ([n(0.6f32), n(0.3f32)], 6),
            ([n(0.2f32), n(0.7f32)], 2),
            ([n(0.14f32), n(0.5f32)], 14),
            ([n(0.3f32), n(0.6f32)], 3),
            ([n(0.10f32), n(0.1f32)], 10),
            ([n(0.16f32), n(0.7f32)], 16),
            ([n(0.1f32), n(0.8f32)], 1),
            ([n(0.15f32), n(0.6f32)], 15),
            ([n(0.5f32), n(0.4f32)], 5),
            ([n(0.8f32), n(0.1f32)], 8),
            ([n(0.11f32), n(0.2f32)], 11),
        ];

        for (point, item) in content_to_add {
            tree.add(&point, item);
        }

        assert_eq!(tree.size(), 16);
        let max_qty = 5;

        let query = [n(0.9f32), n(0.7f32)];
        let radius = n(0.6f32);
        let expected = vec![6, 5, 3, 2, 4];

        let result: Vec<_> = tree
            .best_n_within(&query, radius, max_qty, &squared_euclidean)
            .collect();
        assert_eq!(result, expected);

        let mut rng = rand::thread_rng();
        for _i in 0..1000 {
            let query = [
                n(rng.gen_range(0.0f32..0.9f32)),
                n(rng.gen_range(0.0f32..0.9f32)),
            ];
            println!("{}: {}, {}", _i, query[0].to_string(), query[1].to_string());
            let radius = n(0.1f32);
            let expected = linear_search(&content_to_add, &query, radius, max_qty);

            let mut result: Vec<_> = tree
                .best_n_within(&query, radius, max_qty, &squared_euclidean)
                .collect();
            result.sort_unstable();
            assert_eq!(result, expected);
        }
    }

    fn linear_search<A: Axis, const K: usize>(
        content: &[([A; K], u32)],
        query_point: &[A; K],
        radius: A,
        max_qty: usize,
    ) -> Vec<u32> {
        let mut best_items = Vec::with_capacity(max_qty);

        for &(p, item) in content {
            let dist = squared_euclidean(query_point, &p);
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

        best_items
    }
}
