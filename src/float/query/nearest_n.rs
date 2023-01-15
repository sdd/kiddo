use az::{Az, Cast};
use std::ops::Rem;
use min_max_heap::MinMaxHeap;
use crate::float::kdtree::{Axis, KdTree};
use crate::float::heap_element::HeapElement;
use crate::types::{Content, Index};

pub struct NearestIter<
    A: Axis,
    T: Content
> {
    result: MinMaxHeap<HeapElement<A, T>>,
}

impl<A: Axis, T: Content> Iterator
for NearestIter<A, T> {
    type Item = (A, T);

    fn next(&mut self) -> Option<(A, T)> {
        self.result.pop_min().map(|a|(a.distance, a.item))
    }
}

impl<A: Axis, T: Content, const K: usize, const B: usize, IDX: Index<T = IDX>> KdTree<A, T, K, B, IDX> where usize: Cast<IDX>  {
    #[inline]
    pub fn nearest_n<F>(&self, query: &[A; K], qty: usize, distance_fn: &F) -> impl Iterator<Item = (A, T)>
    where
        F: Fn(&[A; K], &[A; K]) -> A,
    {
        let mut result: MinMaxHeap<HeapElement<A, T>> = MinMaxHeap::with_capacity(qty);

        unsafe { self.nearest_n_recurse(query, distance_fn, self.root_index, 0, &mut result) }

        NearestIter {
            result
        }
    }

    unsafe fn nearest_n_recurse<F>(
        &self,
        query: &[A; K],
        distance_fn: &F,
        curr_node_idx: IDX,
        split_dim: usize,
        results: &mut MinMaxHeap<HeapElement<A, T>>,
    )
    where
        F: Fn(&[A; K], &[A; K]) -> A,
    {
        if KdTree::<A, T, K, B, IDX>::is_stem_index(curr_node_idx) {
            let node = &self.stems.get_unchecked(curr_node_idx.az::<usize>());

            let child_node_indices = if *query.get_unchecked(split_dim) < node.split_val {
                [node.left, node.right]
            } else {
                [node.right, node.left]
            };
            let next_split_dim = (split_dim + 1).rem(K);

            for node_idx in child_node_indices {
                let child_node_dist = self.child_dist_to_bounds(query, node_idx, distance_fn);
                if Self::dist_belongs_in_heap(child_node_dist, results) {
                    self.nearest_n_recurse(
                        query,
                        distance_fn,
                        node_idx,
                        next_split_dim,
                        results,
                    );
                }
            }
        } else {
            let leaf_node = self.leaves.get_unchecked((curr_node_idx - IDX::leaf_offset()).az::<usize>());

            leaf_node
                .content_points
                .iter()
                .take(leaf_node.size.az::<usize>())
                .enumerate()
                .for_each(|(idx, entry)| {
                    let distance: A = distance_fn(query, &entry);
                    if Self::dist_belongs_in_heap(distance, results) {
                        let item = unsafe { *leaf_node.content_items.get_unchecked(idx) };
                        let element = HeapElement {
                            distance,
                            item
                        };
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
        heap.len() == 0 || dist < heap.peek_max().unwrap().distance || heap.len() < heap.capacity()
    }
}

#[cfg(test)]
mod tests {
    use rand::Rng;
    use crate::float::distance::manhattan;
    use crate::float::kdtree::{Axis, KdTree};

    type AX = f32;

    #[test]
    fn can_query_nearest_one_item() {
        let mut tree: KdTree<AX, u32, 4, 8, u32> = KdTree::new();

        let content_to_add: [([AX; 4], u32); 16] = [
            ([0.9f32, 0.0f32, 0.9f32, 0.0f32], 9), // 1.34
            ([0.4f32, 0.5f32, 0.4f32, 0.51f32], 4), // 0.86
            ([0.12f32, 0.3f32, 0.12f32, 0.3f32], 12), // 1.82
            ([0.7f32, 0.2f32, 0.7f32, 0.22f32], 7), // 0.86
            ([0.13f32, 0.4f32, 0.13f32, 0.4f32], 13), // 1.56
            ([0.6f32, 0.3f32, 0.6f32, 0.33f32], 6), // 0.86
            ([0.2f32, 0.7f32, 0.2f32, 0.7f32], 2), // 1.46
            ([0.14f32, 0.5f32, 0.14f32, 0.5f32], 14), // 1.38
            ([0.3f32, 0.6f32, 0.3f32, 0.6f32], 3), // 1.06
            ([0.10f32, 0.1f32, 0.10f32, 0.1f32], 10), // 2.26
            ([0.16f32, 0.7f32, 0.16f32, 0.7f32], 16), // 1.54
            ([0.1f32, 0.8f32, 0.1f32, 0.8f32], 1), // 1.86
            ([0.15f32, 0.6f32, 0.15f32, 0.6f32], 15), // 1.36
            ([0.5f32, 0.4f32, 0.5f32, 0.44f32], 5), // 0.86
            ([0.8f32, 0.1f32, 0.8f32, 0.15f32], 8), // 0.86
            ([0.11f32, 0.2f32, 0.11f32, 0.2f32], 11), // 2.04
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

        let expected = vec![(0.81999993, 5), (0.8299999, 6), (0.84000003, 7)];

        let result: Vec<_> = tree.nearest_n(&query_point, 3, &manhattan).collect();
        assert_eq!(result, expected);

        let qty = 10;
        let mut rng = rand::thread_rng();
        for _i in 0..1000 {
            let query_point = [
                rng.gen_range(0f32..1f32),
                rng.gen_range(0f32..1f32),
                rng.gen_range(0f32..1f32),
                rng.gen_range(0f32..1f32),
            ];
            let expected = linear_search(&content_to_add, qty, &query_point);

            let result: Vec<_> = tree.nearest_n(&query_point, qty, &manhattan).collect();

            let result_dists:  Vec<_> = result.iter().map(|(d, _)|d).collect();
            let expected_dists:  Vec<_> = expected.iter().map(|(d, _)|d).collect();

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
                results.sort_by(|(a_dist, _), (b_dist, _)| a_dist.partial_cmp(&b_dist).unwrap());
            } else if dist < results[qty-1].0 {
                results[qty-1] = (dist, item);
                results.sort_by(|(a_dist, _), (b_dist, _)| a_dist.partial_cmp(&b_dist).unwrap());
            }
        }

        results
    }
}
