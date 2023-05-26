use crate::float::kdtree::{Axis, KdTree};
use crate::float::neighbour::Neighbour;
use crate::types::{Content, Index};
use az::{Az, Cast};
use std::collections::BinaryHeap;
use std::ops::Rem;


macro_rules! genreate_nearest_n {
    ($kdtree:ident, $doctest_build_tree:tt) => {
    doc_comment! {
    concat!("Finds the nearest `qty` elements to `query`, using the specified
distance metric function.
# Examples

```rust
    use kiddo::float::kdtree::KdTree;
    use kiddo::distance::squared_euclidean;

    ",  $doctest_build_tree, "

    let nearest: Vec<_> = tree.nearest_n(&[1.0, 2.0, 5.1], 1, &squared_euclidean);

    assert_eq!(nearest.len(), 1);
    assert!((nearest[0].distance - 0.01f64).abs() < f64::EPSILON);
    assert_eq!(nearest[0].item, 100);
```"),
    #[inline]
    pub fn nearest_n<F>(&self, query: &[A; K], qty: usize, distance_fn: &F) -> Vec<Neighbour<A, T>>
    where
        F: Fn(&[A; K], &[A; K]) -> A,
    {
        let mut off = [A::zero(); K];
        let mut result: BinaryHeap<Neighbour<A, T>> = BinaryHeap::with_capacity(qty);

        unsafe {
            self.nearest_n_recurse(
                query,
                distance_fn,
                self.root_index,
                0,
                &mut result,
                &mut off,
                A::zero(),
            )
        }

        result.into_sorted_vec()
    }

    unsafe fn nearest_n_recurse<F>(
        &self,
        query: &[A; K],
        distance_fn: &F,
        curr_node_idx: IDX,
        split_dim: usize,
        results: &mut BinaryHeap<Neighbour<A, T>>,
        off: &mut [A; K],
        rd: A,
    ) where
        F: Fn(&[A; K], &[A; K]) -> A,
    {
        if $kdtree::<A, T, K, B, IDX>::is_stem_index(curr_node_idx) {
            let node = &self.stems.get_unchecked(curr_node_idx.az::<usize>());

            let mut rd = rd;
            let old_off = off[split_dim];
            let new_off = query[split_dim] - node.split_val;

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
            rd = rd + new_off * new_off - old_off * old_off;
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
                        let element = Neighbour { distance, item };
                        if results.len() < results.capacity() {
                            results.push(element)
                        } else {
                            let mut top = results.peek_mut().unwrap();
                            if element.distance < top.distance {
                                *top = element;
                            }
                        }
                    }
                });
        }
    }

    fn dist_belongs_in_heap(dist: A, heap: &BinaryHeap<Neighbour<A, T>>) -> bool {
        heap.is_empty() || dist < heap.peek().unwrap().distance || heap.len() < heap.capacity()
    }
}}}


impl<A: Axis, T: Content, const K: usize, const B: usize, IDX: Index<T = IDX>>
    KdTree<A, T, K, B, IDX>
where
    usize: Cast<IDX>,
{
    genreate_nearest_n!(
        KdTree,
        "let mut tree: KdTree<f64, u32, 3, 32, u32> = KdTree::new();
    tree.add(&[1.0, 2.0, 5.0], 100);
    tree.add(&[2.0, 3.0, 6.0], 101);"
    );
}

#[cfg(feature = "rkyv")]
use crate::float::kdtree::ArchivedKdTree;
#[cfg(feature = "rkyv")]
impl<A: Axis + rkyv::Archive<Archived = A>, T: Content + rkyv::Archive<Archived = T>, const K: usize, const B: usize, IDX: Index<T = IDX> + rkyv::Archive<Archived = IDX>>
ArchivedKdTree<A, T, K, B, IDX>
    where
        usize: Cast<IDX>,
{
    genreate_nearest_n!(
        ArchivedKdTree,
        "use std::fs::File;
    use memmap::MmapOptions;

    let mmap = unsafe { MmapOptions::new().map(&File::open(\"./examples/test-tree.rkyv\").unwrap()).unwrap() };
    let tree = unsafe { rkyv::archived_root::<KdTree<f64, u32, 3, 32, u32>>(&mmap) };"
    );
}

#[cfg(test)]
mod tests {
    use crate::float::distance::squared_euclidean;
    use crate::float::kdtree::{Axis, KdTree};
    use rand::Rng;

    type AX = f32;

    #[test]
    fn can_query_nearest_n_item() {
        let mut tree: KdTree<AX, u32, 4, 8, u32> = KdTree::new();

        let content_to_add: [([AX; 4], u32); 16] = [
            ([0.9f32, 0.0f32, 0.9f32, 0.0f32], 9),    // 1.34
            ([0.4f32, 0.5f32, 0.4f32, 0.51f32], 4),   // 0.86
            ([0.12f32, 0.3f32, 0.12f32, 0.3f32], 12), // 1.82
            ([0.7f32, 0.2f32, 0.7f32, 0.22f32], 7),   // 0.86
            ([0.13f32, 0.4f32, 0.13f32, 0.4f32], 13), // 1.56
            ([0.6f32, 0.3f32, 0.6f32, 0.33f32], 6),   // 0.86
            ([0.2f32, 0.7f32, 0.2f32, 0.7f32], 2),    // 1.46
            ([0.14f32, 0.5f32, 0.14f32, 0.5f32], 14), // 1.38
            ([0.3f32, 0.6f32, 0.3f32, 0.6f32], 3),    // 1.06
            ([0.10f32, 0.1f32, 0.10f32, 0.1f32], 10), // 2.26
            ([0.16f32, 0.7f32, 0.16f32, 0.7f32], 16), // 1.54
            ([0.1f32, 0.8f32, 0.1f32, 0.8f32], 1),    // 1.86
            ([0.15f32, 0.6f32, 0.15f32, 0.6f32], 15), // 1.36
            ([0.5f32, 0.4f32, 0.5f32, 0.44f32], 5),   // 0.86
            ([0.8f32, 0.1f32, 0.8f32, 0.15f32], 8),   // 0.86
            ([0.11f32, 0.2f32, 0.11f32, 0.2f32], 11), // 2.04
        ];

        for (point, item) in content_to_add {
            tree.add(&point, item);
        }

        assert_eq!(tree.size(), 16);

        let query_point = [0.78f32, 0.55f32, 0.78f32, 0.55f32];

        let expected = vec![(0.17569996, 6), (0.19139998, 5), (0.24420004, 7)];

        let result: Vec<_> = tree
            .nearest_n(&query_point, 3, &squared_euclidean)
            .into_iter()
            .map(|n| (n.distance, n.item))
            .collect();
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

            let result: Vec<_> = tree
                .nearest_n(&query_point, qty, &squared_euclidean)
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();

            let result_dists: Vec<_> = result.iter().map(|(d, _)| d).collect();
            let expected_dists: Vec<_> = expected.iter().map(|(d, _)| d).collect();

            assert_eq!(result_dists, expected_dists);
        }
    }

    #[test]
    fn can_query_nearest_10_items_large_scale() {
        const TREE_SIZE: usize = 100_000;
        const NUM_QUERIES: usize = 100;
        const N: usize = 10;

        let content_to_add: Vec<([f32; 4], u32)> = (0..TREE_SIZE)
            .map(|_| rand::random::<([f32; 4], u32)>())
            .collect();

        let mut tree: KdTree<AX, u32, 4, 32, u32> = KdTree::with_capacity(TREE_SIZE);
        content_to_add
            .iter()
            .for_each(|(point, content)| tree.add(point, *content));
        assert_eq!(tree.size(), TREE_SIZE as u32);

        let query_points: Vec<[f32; 4]> = (0..NUM_QUERIES)
            .map(|_| rand::random::<[f32; 4]>())
            .collect();

        for query_point in query_points {
            let expected = linear_search(&content_to_add, N, &query_point);

            let result: Vec<_> = tree
                .nearest_n(&query_point, N, &squared_euclidean)
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();

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
            let dist = squared_euclidean(query_point, &p);
            if results.len() < qty {
                results.push((dist, item));
                results.sort_by(|(a_dist, _), (b_dist, _)| a_dist.partial_cmp(&b_dist).unwrap());
            } else if dist < results[qty - 1].0 {
                results[qty - 1] = (dist, item);
                results.sort_by(|(a_dist, _), (b_dist, _)| a_dist.partial_cmp(&b_dist).unwrap());
            }
        }

        results
    }
}
