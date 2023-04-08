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
    /// Queries the tree to find the best `n` elements within `radius` of `point`, using the specified
    /// distance metric function. Results are returned in arbitrary order. 'Best' is determined by
    /// performing a comparison of the elements using < (ie, std::ord::lt). Returns an iterator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fixed::FixedU16;
    /// use fixed::types::extra::U0;
    /// use kiddo::fixed::kdtree::KdTree;
    /// use kiddo::fixed::distance::squared_euclidean;
    ///
    /// type FXD = FixedU16<U0>;
    ///
    /// let mut tree: KdTree<FXD, u32, 3, 32, u32> = KdTree::new();
    ///
    /// tree.add(&[FXD::from_num(1), FXD::from_num(2), FXD::from_num(5)], 100);
    /// tree.add(&[FXD::from_num(2), FXD::from_num(3), FXD::from_num(6)], 1);
    /// tree.add(&[FXD::from_num(20), FXD::from_num(30), FXD::from_num(60)], 102);
    ///
    /// let mut best_n_within_iter = tree.best_n_within(&[FXD::from_num(1), FXD::from_num(2), FXD::from_num(5)], FXD::from_num(10), 1, &squared_euclidean);
    /// let first = best_n_within_iter.next().unwrap();
    ///
    /// assert_eq!(first, 1);
    /// ```
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
        let mut off = [A::ZERO; K];
        // TODO: switch to https://docs.rs/min-max-heap/1.3.0/min_max_heap/struct.MinMaxHeap.html
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
                &mut off,
                A::ZERO,
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
        off: &mut [A; K],
        rd: A,
    ) where
        F: Fn(&[A; K], &[A; K]) -> A,
    {
        if KdTree::<A, T, K, B, IDX>::is_stem_index(curr_node_idx) {
            let node = unsafe { self.stems.get_unchecked(curr_node_idx.az::<usize>()) };

            let mut rd = rd;
            let old_off = off[split_dim];
            let new_off = query[split_dim].dist(node.split_val);

            let [closer_node_idx, further_node_idx] =
                if *query.get_unchecked(split_dim) < node.split_val {
                    [node.left, node.right]
                } else {
                    [node.right, node.left]
                };
            let next_split_dim = (split_dim + 1).rem(K);

            self.best_n_within_recurse(
                query,
                radius,
                max_qty,
                distance_fn,
                closer_node_idx,
                next_split_dim,
                best_items,
                off,
                rd,
            );

            // TODO: switch from dist_fn to a dist trait that can apply to 1D as well as KD
            //       so that updating rd is not hardcoded to sq euclidean
            rd = rd.saturating_add(
                (new_off.saturating_mul(new_off)).saturating_sub(old_off.saturating_mul(old_off)),
            );

            if rd <= radius {
                off[split_dim] = new_off;
                self.best_n_within_recurse(
                    query,
                    radius,
                    max_qty,
                    distance_fn,
                    further_node_idx,
                    next_split_dim,
                    best_items,
                    off,
                    rd,
                );
                off[split_dim] = old_off;
            }
        } else {
            let leaf_node = unsafe {
                self.leaves
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
            .for_each(|(idx, _)| unsafe {
                Self::get_item_and_add_if_good(max_qty, best_items, leaf_node, idx)
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
    use crate::fixed::distance::manhattan;
    use crate::fixed::kdtree::{Axis, KdTree};
    use crate::test_utils::{rand_data_fixed_u16_entry, rand_data_fixed_u16_point};
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
        let radius = n(0.8f32);
        let expected = vec![6, 5, 3, 2, 4];

        let result: Vec<_> = tree
            .best_n_within(&query, radius, max_qty, &manhattan)
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
                .best_n_within(&query, radius, max_qty, &manhattan)
                .collect();

            result.sort_unstable();
            assert_eq!(result, expected);
        }
    }

    #[test]
    fn can_query_best_items_within_radius_large_scale() {
        const TREE_SIZE: usize = 100_000;
        const NUM_QUERIES: usize = 100;
        let radius: FXD = n(0.6);
        let max_qty = 5;

        let content_to_add: Vec<([FXD; 4], u32)> = (0..TREE_SIZE)
            .map(|_| rand_data_fixed_u16_entry::<U14, u32, 4>())
            .collect();

        let mut tree: KdTree<FXD, u32, 4, 4, u32> = KdTree::with_capacity(TREE_SIZE);
        content_to_add
            .iter()
            .for_each(|(point, content)| tree.add(point, *content));
        assert_eq!(tree.size(), TREE_SIZE as u32);

        let query_points: Vec<[FXD; 4]> = (0..NUM_QUERIES)
            .map(|_| rand_data_fixed_u16_point::<U14, 4>())
            .collect();

        for query_point in query_points {
            let expected = linear_search(&content_to_add, &query_point, radius, max_qty);

            let mut result: Vec<_> = tree
                .best_n_within(&query_point, radius, max_qty, &manhattan)
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
            let dist = manhattan(query_point, &p);
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
