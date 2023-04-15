use az::{Az, Cast};
use std::collections::BinaryHeap;
use std::ops::Rem;
use crate::best_neighbour::BestNeighbour;
use crate::fixed::distance::DistanceMetric;

use crate::fixed::kdtree::{Axis, KdTree, LeafNode};
use crate::types::{Content, Index};

impl<A: Axis, T: Content, const K: usize, const B: usize, IDX: Index<T = IDX>>
    KdTree<A, T, K, B, IDX>
where
    usize: Cast<IDX>,
{
    /// Queries the tree to find the best `n` elements within `dist` of `point`, using the specified
    /// distance metric function. Results are returned in arbitrary order. 'Best' is determined by
    /// performing a comparison of the elements using < (ie, std::ord::lt). Returns an iterator.
    ///
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fixed::FixedU16;
    /// use fixed::types::extra::U0;
    /// use kiddo::fixed::distance::SquaredEuclidean;
    /// use kiddo::fixed::kdtree::KdTree;
    ///
    /// type FXD = FixedU16<U0>;
    ///
    /// let mut tree: KdTree<FXD, u32, 3, 32, u32> = KdTree::new();
    ///
    /// tree.add(&[FXD::from_num(1), FXD::from_num(2), FXD::from_num(5)], 100);
    /// tree.add(&[FXD::from_num(2), FXD::from_num(3), FXD::from_num(6)], 1);
    /// tree.add(&[FXD::from_num(20), FXD::from_num(30), FXD::from_num(60)], 102);
    ///
    /// let mut best_n_within = tree.best_n_within::<SquaredEuclidean>(&[FXD::from_num(1), FXD::from_num(2), FXD::from_num(5)], FXD::from_num(10), 1);
    ///
    /// assert_eq!(best_n_within[0].item, 1);
    /// ```
    #[inline]
    pub fn best_n_within<D>(
        &self,
        query: &[A; K],
        dist: A,
        max_qty: usize,
    ) -> Vec<BestNeighbour<A, T>>
    where
        D: DistanceMetric<A, K>
    {
        let mut off = [A::ZERO; K];
        let mut best_items: BinaryHeap<BestNeighbour<A, T>> = BinaryHeap::new();

        unsafe {
            self.best_n_within_recurse::<D>(
                query,
                dist,
                max_qty,
                self.root_index,
                0,
                &mut best_items,
                &mut off,
                A::ZERO,
            );
        }

        best_items.into_sorted_vec()
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn best_n_within_recurse<D>(
        &self,
        query: &[A; K],
        radius: A,
        max_qty: usize,
        curr_node_idx: IDX,
        split_dim: usize,
        best_items: &mut BinaryHeap<BestNeighbour<A, T>>,
        off: &mut [A; K],
        rd: A,
    ) where
        D: DistanceMetric<A, K>
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

            self.best_n_within_recurse::<D>(
                query,
                radius,
                max_qty,
                closer_node_idx,
                next_split_dim,
                best_items,
                off,
                rd,
            );

            rd = rd.saturating_add(D::dist1(old_off, new_off));

            if rd <= radius {
                off[split_dim] = new_off;
                self.best_n_within_recurse::<D>(
                    query,
                    radius,
                    max_qty,
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

            Self::process_leaf_node::<D>(query, radius, max_qty, best_items, leaf_node);
        }
    }

    unsafe fn process_leaf_node<D>(
        query: &[A; K],
        radius: A,
        max_qty: usize,
        best_items: &mut BinaryHeap<BestNeighbour<A, T>>,
        leaf_node: &LeafNode<A, T, K, B, IDX>,
    ) where
        D: DistanceMetric<A, K>
    {
        leaf_node
            .content_points
            .iter()
            .take(leaf_node.size.az::<usize>())
            .map(|entry| D::dist(query, entry))
            .enumerate()
            .filter(|(_, distance)| *distance <= radius)
            .for_each(|(idx, dist)| {
                Self::get_item_and_add_if_good(max_qty, best_items, leaf_node, idx, dist)
            });
    }

    unsafe fn get_item_and_add_if_good(
        max_qty: usize,
        best_items: &mut BinaryHeap<BestNeighbour<A, T>>,
        leaf_node: &LeafNode<A, T, K, B, IDX>,
        idx: usize,
        distance: A,
    ) {
        let item = *leaf_node.content_items.get_unchecked(idx.az::<usize>());
        if best_items.len() < max_qty {
            best_items.push(BestNeighbour { item, distance });
        } else {
            let mut top = best_items.peek_mut().unwrap();
            if item < top.item {
                top.distance = distance;
                top.item = item;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::fixed::distance::{Manhattan, SquaredEuclidean};
    use crate::fixed::kdtree::{Axis, KdTree};
    use crate::test_utils::{rand_data_fixed_u16_entry, rand_data_fixed_u16_point};
    use fixed::types::extra::U14;
    use fixed::FixedU16;
    use rand::Rng;
    use crate::best_neighbour::BestNeighbour;
    use crate::fixed::distance::DistanceMetric;

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
        let expected = vec![
            BestNeighbour { item: 1, distance: n(0.65f32) },
            BestNeighbour { item: 2, distance: n(0.49f32) },
            BestNeighbour { item: 3, distance: n(0.36993f32) },
            BestNeighbour { item: 4, distance: n(0.29f32) },
            BestNeighbour { item: 5, distance: n(0.24994f32) },
        ];

        let result: Vec<_> = tree
            .best_n_within::<SquaredEuclidean>(&query, radius, max_qty);
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

            let result: Vec<_> = tree
                .best_n_within::<Manhattan>(&query, radius, max_qty);
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

            let result: Vec<_> = tree
                .best_n_within::<Manhattan>(&query_point, radius, max_qty);
            assert_eq!(result, expected);
        }
    }

    fn linear_search<A: Axis, const K: usize>(
        content: &[([A; K], u32)],
        query_point: &[A; K],
        radius: A,
        max_qty: usize,
    ) -> Vec<BestNeighbour<A, u32>> {
        let mut best_items = Vec::with_capacity(max_qty);

        for &(p, item) in content {
            let distance = Manhattan::dist(query_point, &p);
            if distance <= radius {
                if best_items.len() < max_qty {
                    best_items.push(BestNeighbour{ item, distance });
                } else {
                    if item < best_items.last().unwrap().item {
                        best_items.pop().unwrap();
                        best_items.push(BestNeighbour{ item, distance });
                    }
                }
            }
            best_items.sort_unstable();
        }

        best_items
    }
}
