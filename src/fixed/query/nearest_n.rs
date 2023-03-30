use crate::fixed::heap_element::HeapElement;
use crate::fixed::kdtree::{Axis, KdTree};
use crate::types::{Content, Index};
use az::{Az, Cast};
use min_max_heap::MinMaxHeap;
use std::ops::Rem;

pub struct NearestIter<A: Axis, T: Content> {
    result: MinMaxHeap<HeapElement<A, T>>,
}

impl<A: Axis, T: Content> Iterator for NearestIter<A, T> {
    type Item = (A, T);

    fn next(&mut self) -> Option<(A, T)> {
        self.result.pop_min().map(|a| (a.distance, a.item))
    }
}

impl<A: Axis, T: Content, const K: usize, const B: usize, IDX: Index<T = IDX>>
    KdTree<A, T, K, B, IDX>
where
    usize: Cast<IDX>,
{
    /// Finds the nearest `qty` elements to `query`, using the specified
    /// distance metric function.
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
    /// tree.add(&[FXD::from_num(2), FXD::from_num(3), FXD::from_num(6)], 101);
    ///
    /// let nearest: Vec<_> = tree.nearest_n(&[FXD::from_num(1), FXD::from_num(2), FXD::from_num(5)], 1, &squared_euclidean).collect();
    ///
    /// assert_eq!(nearest.len(), 1);
    /// assert_eq!(nearest[0].0, FXD::from_num(0));
    /// assert_eq!(nearest[0].1, 100);
    /// ```
    #[inline]
    pub fn nearest_n<F>(
        &self,
        query: &[A; K],
        qty: usize,
        distance_fn: &F,
    ) -> impl Iterator<Item = (A, T)>
    where
        F: Fn(&[A; K], &[A; K]) -> A,
    {
        let mut off = [A::ZERO; K];
        let mut result: MinMaxHeap<HeapElement<A, T>> = MinMaxHeap::with_capacity(qty);

        unsafe {
            self.nearest_n_recurse(
                query,
                distance_fn,
                self.root_index,
                0,
                &mut result,
                &mut off,
                A::ZERO,
            )
        }

        NearestIter { result }
    }

    unsafe fn nearest_n_recurse<F>(
        &self,
        query: &[A; K],
        distance_fn: &F,
        curr_node_idx: IDX,
        split_dim: usize,
        results: &mut MinMaxHeap<HeapElement<A, T>>,
        off: &mut [A; K],
        rd: A,
    ) where
        F: Fn(&[A; K], &[A; K]) -> A,
    {
        if KdTree::<A, T, K, B, IDX>::is_stem_index(curr_node_idx) {
            let node = &self.stems.get_unchecked(curr_node_idx.az::<usize>());

            let mut rd = rd;
            let old_off = off[split_dim];
            let new_off = query[split_dim].saturating_sub(node.split_val);

            let [closer_node_idx, further_node_idx] =
                if *query.get_unchecked(split_dim) < node.split_val {
                    [node.left, node.right]
                } else {
                    [node.right, node.left]
                };
            let next_split_dim = (split_dim + 1).rem(K);

            self.nearest_n_recurse(
                query,
                distance_fn,
                closer_node_idx,
                next_split_dim,
                results,
                off,
                rd,
            );

            // TODO: switch from dist_fn to a dist trait that can apply to 1D as well as KD
            //       so that updating rd is not hardcoded to sq euclidean
            rd = rd.saturating_add(
                (new_off.saturating_mul(new_off)).saturating_sub(old_off.saturating_mul(old_off)),
            );

            if Self::dist_belongs_in_heap(rd, results) {
                off[split_dim] = new_off;
                self.nearest_n_recurse(
                    query,
                    distance_fn,
                    further_node_idx,
                    next_split_dim,
                    results,
                    off,
                    rd,
                );
                off[split_dim] = old_off;
            }
        } else {
            let leaf_node = self
                .leaves
                .get_unchecked((curr_node_idx - IDX::leaf_offset()).az::<usize>());

            leaf_node
                .content_points
                .iter()
                .take(leaf_node.size.az::<usize>())
                .enumerate()
                .for_each(|(idx, entry)| {
                    let distance: A = distance_fn(query, entry);
                    if Self::dist_belongs_in_heap(distance, results) {
                        let item = unsafe { *leaf_node.content_items.get_unchecked(idx) };
                        let element = HeapElement { distance, item };
                        if results.len() < results.capacity() {
                            results.push(element)
                        } else {
                            results.replace_max(element);
                        }
                    }
                });
        }
    }

    fn dist_belongs_in_heap(dist: A, heap: &MinMaxHeap<HeapElement<A, T>>) -> bool {
        heap.is_empty() || dist < heap.peek_max().unwrap().distance || heap.len() < heap.capacity()
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
    fn can_query_nearest_n_items() {
        let mut tree: KdTree<FXD, u32, 4, 4, u32> = KdTree::new();

        let content_to_add: [([FXD; 4], u32); 16] = [
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

        let query_point = [n(0.78f32), n(0.55f32), n(0.78f32), n(0.55f32)];

        let expected = vec![(n(0.86), 7), (n(0.86), 5), (n(0.86), 4)];

        let result: Vec<_> = tree.nearest_n(&query_point, 3, &manhattan).collect();
        assert_eq!(result, expected);

        let qty = 10;
        let mut rng = rand::thread_rng();
        for _i in 0..1000 {
            let query_point = [
                n(rng.gen_range(0f32..1f32)),
                n(rng.gen_range(0f32..1f32)),
                n(rng.gen_range(0f32..1f32)),
                n(rng.gen_range(0f32..1f32)),
            ];
            let expected = linear_search(&content_to_add, qty, &query_point);

            let result: Vec<_> = tree.nearest_n(&query_point, qty, &manhattan).collect();

            let result_dists: Vec<_> = result.iter().map(|(d, _)| d).collect();
            let expected_dists: Vec<_> = expected.iter().map(|(d, _)| d).collect();

            assert_eq!(result_dists, expected_dists);
        }
    }

    #[test]
    fn can_query_nearest_n_items_large_scale() {
        const TREE_SIZE: usize = 100_000;
        const NUM_QUERIES: usize = 100;
        const N: usize = 10;

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
            let expected = linear_search(&content_to_add, N, &query_point);

            let result: Vec<_> = tree.nearest_n(&query_point, N, &manhattan).collect();

            let result_dists: Vec<_> = result.iter().map(|(d, _)| d).collect();
            let expected_dists: Vec<_> = expected.iter().map(|(d, _)| d).collect();

            assert_eq!(result_dists, expected_dists);
        }
    }

    fn linear_search<A: Axis, const K: usize>(
        content: &[([A; K], u32)],
        qty: usize,
        query_point: &[A; K],
    ) -> Vec<(A, u32)> {
        let mut results = vec![];

        for &(p, item) in content {
            let dist = manhattan(query_point, &p);
            if results.len() < qty {
                results.push((dist, item));
                results.sort_by(|(a_dist, _), (b_dist, _)| a_dist.partial_cmp(b_dist).unwrap());
            } else if dist < results[qty - 1].0 {
                results[qty - 1] = (dist, item);
                results.sort_by(|(a_dist, _), (b_dist, _)| a_dist.partial_cmp(b_dist).unwrap());
            }
        }

        results
    }
}
