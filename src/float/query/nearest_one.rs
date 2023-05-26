use crate::float::kdtree::{Axis, KdTree, LeafNode};
use crate::types::{Content, Index};
use az::{Az, Cast};
use std::ops::Rem;

macro_rules! genreate_nearest_one {
    ($kdtree:ident, $leafnode:ident, $doctest_build_tree:tt) => {
    doc_comment! {
    concat!("Finds the nearest element to `query`, using the specified
distance metric function.

Faster than querying for nearest_n(point, 1, ...) due
to not needing to allocate memory or maintain sorted results.

# Examples

```rust
    use kiddo::float::kdtree::KdTree;
    use kiddo::distance::squared_euclidean;

    ",  $doctest_build_tree, "

    let nearest = tree.nearest_one(&[1.0, 2.0, 5.1], &squared_euclidean);

    assert!((nearest.0 - 0.01f64).abs() < f64::EPSILON);
    assert_eq!(nearest.1, 100);
```"),
    #[inline]
    pub fn nearest_one<F>(&self, query: &[A; K], distance_fn: &F) -> (A, T)
        where
            F: Fn(&[A; K], &[A; K]) -> A,
    {
        let mut off = [A::zero(); K];
        unsafe {
            self.nearest_one_recurse(
                query,
                distance_fn,
                self.root_index,
                0,
                T::zero(),
                A::max_value(),
                &mut off,
                A::zero(),
            )
        }
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
        off: &mut [A; K],
        rd: A,
    ) -> (A, T)
        where
            F: Fn(&[A; K], &[A; K]) -> A,
    {
        if KdTree::<A, T, K, B, IDX>::is_stem_index(curr_node_idx) {
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

            let (dist, item) = self.nearest_one_recurse(
                query,
                distance_fn,
                closer_node_idx,
                next_split_dim,
                best_item,
                best_dist,
                off,
                rd,
            );

            if dist < best_dist {
                best_dist = dist;
                best_item = item;
            }

            // TODO: switch from dist_fn to a dist trait that can apply to 1D as well as KD
            //       so that updating rd is not hardcoded to sq euclidean
            rd = rd + new_off * new_off - old_off * old_off;
            if rd <= best_dist {
                off[split_dim] = new_off;
                let (dist, item) = self.nearest_one_recurse(
                    query,
                    distance_fn,
                    further_node_idx,
                    next_split_dim,
                    best_item,
                    best_dist,
                    off,
                    rd,
                );
                off[split_dim] = old_off;

                if dist < best_dist {
                    best_dist = dist;
                    best_item = item;
                }
            }
        } else {
            let leaf_node = self
                .leaves
                .get_unchecked((curr_node_idx - IDX::leaf_offset()).az::<usize>());

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
        query: &[A; K],
        distance_fn: &F,
        best_item: &mut T,
        best_dist: &mut A,
        leaf_node: &$leafnode<A, T, K, B, IDX>,
    ) where
        F: Fn(&[A; K], &[A; K]) -> A,
    {
        leaf_node
            .content_points
            .iter()
            .enumerate()
            .take(leaf_node.size.az::<usize>())
            .for_each(|(idx, entry)| {
                let dist = distance_fn(query, entry);
                if dist < *best_dist {
                    *best_dist = dist;
                    *best_item = unsafe { *leaf_node.content_items.get_unchecked(idx) };
                }
            });
    }
}}}

impl<A: Axis, T: Content, const K: usize, const B: usize, IDX: Index<T = IDX>>
    KdTree<A, T, K, B, IDX>
where
    usize: Cast<IDX>,
{
    genreate_nearest_one!(
        KdTree,
        LeafNode,
        "let mut tree: KdTree<f64, u32, 3, 32, u32> = KdTree::new();
    tree.add(&[1.0, 2.0, 5.0], 100);
    tree.add(&[2.0, 3.0, 6.0], 101);"
    );
}

#[cfg(feature = "rkyv")]
use crate::float::kdtree::{ArchivedKdTree,ArchivedLeafNode};
#[cfg(feature = "rkyv")]
impl<A: Axis + rkyv::Archive<Archived = A>, T: Content + rkyv::Archive<Archived = T>, const K: usize, const B: usize, IDX: Index<T = IDX> + rkyv::Archive<Archived = IDX>>
ArchivedKdTree<A, T, K, B, IDX>
    where
        usize: Cast<IDX>,
{
    genreate_nearest_one!(
        ArchivedKdTree,
        ArchivedLeafNode,
        "use std::fs::File;
    use memmap::MmapOptions;

    let mmap = unsafe { MmapOptions::new().map(&File::open(\"./examples/test-tree.rkyv\").unwrap()).unwrap() };
    let tree = unsafe { rkyv::archived_root::<KdTree<f64, u32, 3, 32, u32>>(&mmap) };"
    );
}

#[cfg(test)]
mod tests {
    use crate::float::distance::manhattan;
    use crate::float::kdtree::{Axis, KdTree};
    use rand::Rng;

    type AX = f32;

    #[test]
    fn can_query_nearest_one_item() {
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

        let expected = (0.819999933, 5);

        let result = tree.nearest_one(&query_point, &manhattan);
        assert_eq!(result, expected);

        let mut rng = rand::thread_rng();
        for _i in 0..1000 {
            let query_point = [
                rng.gen_range(0f32..1f32),
                rng.gen_range(0f32..1f32),
                rng.gen_range(0f32..1f32),
                rng.gen_range(0f32..1f32),
            ];
            let expected = linear_search(&content_to_add, &query_point);

            let result = tree.nearest_one(&query_point, &manhattan);

            assert_eq!(result.0, expected.0);
        }
    }

    #[test]
    fn can_query_nearest_one_item_large_scale() {
        const TREE_SIZE: usize = 100_000;
        const NUM_QUERIES: usize = 100;

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
            let expected = linear_search(&content_to_add, &query_point);

            let result = tree.nearest_one(&query_point, &manhattan);

            assert_eq!(result.0, expected.0);
            assert_eq!(result.1, expected.1);
        }
    }

    fn linear_search<A: Axis, const K: usize>(
        content: &[([A; K], u32)],
        query_point: &[A; K],
    ) -> (A, u32) {
        let mut best_dist: A = A::infinity();
        let mut best_item: u32 = u32::MAX;

        for &(p, item) in content {
            let dist = manhattan(query_point, &p);
            if dist < best_dist {
                best_item = item;
                best_dist = dist;
            }
        }

        (best_dist, best_item)
    }
}
