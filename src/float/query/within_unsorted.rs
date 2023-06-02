use crate::float::neighbour::Neighbour;
use az::{Az, Cast};
use std::ops::Rem;

use crate::float::kdtree::{Axis, KdTree};
use crate::types::{is_stem_index, Content, Index};

use crate::generate_within_unsorted;

macro_rules! generate_float_within_unsorted {
    ($doctest_build_tree:tt) => {
        generate_within_unsorted!((
            "Finds all elements within `dist` of `query`, using the specified
distance metric function.

Results are returned in arbitrary order. Faster than `within`.

# Examples

```rust
use kiddo::float::kdtree::KdTree;
use kiddo::distance::squared_euclidean;
",
            $doctest_build_tree,
            "

let within = tree.within_unsorted(&[1.0, 2.0, 5.0], 10f64, &squared_euclidean);

assert_eq!(within.len(), 2);
```"
        ));
    };
}

/*
macro_rules! generate_within_unsorted {
    ($kdtree:ident, $doctest_build_tree:tt) => {
        doc_comment! {
            concat!("Finds all elements within `dist` of `query`, using the specified
distance metric function.

Results are returned in arbitrary order. Faster than `within`.

# Examples

```rust
use kiddo::float::kdtree::KdTree;
use kiddo::distance::squared_euclidean;
",  $doctest_build_tree, "

let within = tree.within_unsorted(&[1.0, 2.0, 5.0], 10f64, &squared_euclidean);

assert_eq!(within.len(), 2);
```"),
            #[inline]
            pub fn within_unsorted<F>(
                &self,
                query: &[A; K],
                dist: A,
                distance_fn: &F,
            ) -> Vec<Neighbour<A, T>>
            where
                F: Fn(&[A; K], &[A; K]) -> A,
            {
                let mut off = [A::zero(); K];
                let mut matching_items = Vec::new();

                unsafe {
                    self.within_unsorted_recurse(
                        query,
                        dist,
                        distance_fn,
                        self.root_index,
                        0,
                        &mut matching_items,
                        &mut off,
                        A::zero(),
                    );
                }

                matching_items
            }

            unsafe fn within_unsorted_recurse<F>(
                &self,
                query: &[A; K],
                radius: A,
                distance_fn: &F,
                curr_node_idx: IDX,
                split_dim: usize,
                matching_items: &mut Vec<Neighbour<A, T>>,
                off: &mut [A; K],
                rd: A,
            ) where
                F: Fn(&[A; K], &[A; K]) -> A,
            {
                if KdTree::<A, T, K, B, IDX>::is_stem_index(curr_node_idx) {
                    let node = self.stems.get_unchecked(curr_node_idx.az::<usize>());

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

                    self.within_unsorted_recurse(
                        query,
                        radius,
                        distance_fn,
                        closer_node_idx,
                        next_split_dim,
                        matching_items,
                        off,
                        rd,
                    );

                    // TODO: switch from dist_fn to a dist trait that can apply to 1D as well as KD
                    //       so that updating rd is not hardcoded to sq euclidean
                    rd = rd + new_off * new_off - old_off * old_off;

                    if rd <= radius {
                        off[split_dim] = new_off;
                        self.within_unsorted_recurse(
                            query,
                            radius,
                            distance_fn,
                            further_node_idx,
                            next_split_dim,
                            matching_items,
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
                        .enumerate()
                        .take(leaf_node.size.az::<usize>())
                        .for_each(|(idx, entry)| {
                            let distance = distance_fn(query, entry);

                            if distance < radius {
                                matching_items.push(Neighbour {
                                    distance,
                                    item: *leaf_node.content_items.get_unchecked(idx.az::<usize>()),
                                });
                            }
                        });
                }
            }
        }
    };
}
 */

impl<A: Axis, T: Content, const K: usize, const B: usize, IDX: Index<T = IDX>>
    KdTree<A, T, K, B, IDX>
where
    usize: Cast<IDX>,
{
    generate_float_within_unsorted!(
        "
let mut tree: KdTree<f64, u32, 3, 32, u32> = KdTree::new();
tree.add(&[1.0, 2.0, 5.0], 100);
tree.add(&[2.0, 3.0, 6.0], 101);"
    );
}

#[cfg(feature = "rkyv")]
use crate::float::kdtree::ArchivedKdTree;
#[cfg(feature = "rkyv")]
impl<
        A: Axis + rkyv::Archive<Archived = A>,
        T: Content + rkyv::Archive<Archived = T>,
        const K: usize,
        const B: usize,
        IDX: Index<T = IDX> + rkyv::Archive<Archived = IDX>,
    > ArchivedKdTree<A, T, K, B, IDX>
where
    usize: Cast<IDX>,
{
    generate_float_within_unsorted!(
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
    use std::cmp::Ordering;

    type AX = f32;

    #[test]
    fn can_query_items_within_radius() {
        let mut tree: KdTree<AX, u32, 4, 4, u32> = KdTree::new();

        let content_to_add: [([AX; 4], u32); 16] = [
            ([0.9f32, 0.0f32, 0.9f32, 0.0f32], 9),
            ([0.4f32, 0.5f32, 0.4f32, 0.5f32], 4),
            ([0.12f32, 0.3f32, 0.12f32, 0.3f32], 12),
            ([0.7f32, 0.2f32, 0.7f32, 0.2f32], 7),
            ([0.13f32, 0.4f32, 0.13f32, 0.4f32], 13),
            ([0.6f32, 0.3f32, 0.6f32, 0.3f32], 6),
            ([0.2f32, 0.7f32, 0.2f32, 0.7f32], 2),
            ([0.14f32, 0.5f32, 0.14f32, 0.5f32], 14),
            ([0.3f32, 0.6f32, 0.3f32, 0.6f32], 3),
            ([0.10f32, 0.1f32, 0.10f32, 0.1f32], 10),
            ([0.16f32, 0.7f32, 0.16f32, 0.7f32], 16),
            ([0.1f32, 0.8f32, 0.1f32, 0.8f32], 1),
            ([0.15f32, 0.6f32, 0.15f32, 0.6f32], 15),
            ([0.5f32, 0.4f32, 0.5f32, 0.4f32], 5),
            ([0.8f32, 0.1f32, 0.8f32, 0.1f32], 8),
            ([0.11f32, 0.2f32, 0.11f32, 0.2f32], 11),
        ];

        for (point, item) in content_to_add {
            tree.add(&point, item);
        }

        assert_eq!(tree.size(), 16);

        let query_point = [0.78f32, 0.55f32, 0.78f32, 0.55f32];

        let radius = 0.2;
        let expected = linear_search(&content_to_add, &query_point, radius);

        let result: Vec<_> = tree
            .within_unsorted(&query_point, radius, &squared_euclidean)
            .into_iter()
            .map(|n| (n.distance, n.item))
            .collect();
        assert_eq!(result, expected);

        let mut rng = rand::thread_rng();
        for _i in 0..1000 {
            let query_point = [
                rng.gen_range(0f32..1f32),
                rng.gen_range(0f32..1f32),
                rng.gen_range(0f32..1f32),
                rng.gen_range(0f32..1f32),
            ];
            let radius = 0.2;
            let expected = linear_search(&content_to_add, &query_point, radius);

            let mut result: Vec<_> = tree
                .within_unsorted(&query_point, radius, &squared_euclidean)
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();
            stabilize_sort(&mut result);

            assert_eq!(result, expected);
        }
    }

    #[test]
    fn can_query_items_unsorted_within_radius_large_scale() {
        const TREE_SIZE: usize = 100_000;
        const NUM_QUERIES: usize = 100;
        const RADIUS: f32 = 0.2;

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
            let expected = linear_search(&content_to_add, &query_point, RADIUS);

            let mut result: Vec<_> = tree
                .within_unsorted(&query_point, RADIUS, &squared_euclidean)
                .into_iter()
                .map(|n| (n.distance, n.item))
                .collect();

            stabilize_sort(&mut result);
            assert_eq!(result, expected);
        }
    }

    fn linear_search<A: Axis, const K: usize>(
        content: &[([A; K], u32)],
        query_point: &[A; K],
        radius: A,
    ) -> Vec<(A, u32)> {
        let mut matching_items = vec![];

        for &(p, item) in content {
            let dist = squared_euclidean(query_point, &p);
            if dist < radius {
                matching_items.push((dist, item));
            }
        }

        stabilize_sort(&mut matching_items);

        matching_items
    }

    fn stabilize_sort<A: Axis>(matching_items: &mut Vec<(A, u32)>) {
        matching_items.sort_unstable_by(|a, b| {
            let dist_cmp = a.0.partial_cmp(&b.0).unwrap();
            if dist_cmp == Ordering::Equal {
                a.1.cmp(&b.1)
            } else {
                dist_cmp
            }
        });
    }
}
