use crate::float_sss::kdtree::{Axis, KdTree};
use crate::types::{Content, Index};
use az::{Az, Cast};
use std::ops::Rem;
use std::ptr;

enum StemIdx<IDX> {
    Stem(IDX),
    DStem(IDX),
    Leaf(IDX),
}

impl<A: Axis, T: Content, const K: usize, const B: usize, IDX: Index<T = IDX>>
    KdTree<A, T, K, B, IDX>
where
    usize: Cast<IDX>,
{
    /// Queries the tree to find the nearest element to `query`, using the specified
    /// distance metric function.
    ///
    /// Faster than querying for nearest_n(point, 1, ...) due
    /// to not needing to allocate memory or maintain sorted results.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kiddo::float::kdtree::KdTree;
    /// use kiddo::distance::squared_euclidean;
    ///
    /// let mut tree: KdTree<f64, u32, 3, 32, u32> = KdTree::new();
    ///
    /// tree.add(&[1.0, 2.0, 5.0], 100);
    /// tree.add(&[2.0, 3.0, 6.0], 101);
    ///
    /// let nearest = tree.nearest_one(&[1.0, 2.0, 5.1], &squared_euclidean);
    ///
    /// assert!((nearest.0 - 0.01f64).abs() < f64::EPSILON);
    /// assert_eq!(nearest.1, 100);
    /// ```
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
                StemIdx::Stem(IDX::one()),
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
        stem_idx: StemIdx<IDX>,
        split_dim: usize,
        mut best_item: T,
        mut best_dist: A,
        off: &mut [A; K],
        rd: A,
    ) -> (A, T)
    where
        F: Fn(&[A; K], &[A; K]) -> A,
    {
        let mut rd = rd;
        let old_off = off[split_dim];
        let new_off: A;

        let [closer_node_idx, further_node_idx] = match stem_idx {
            StemIdx::Stem(mut stem_idx) => {
                let val = *unsafe { self.stems.get_unchecked(stem_idx.az::<usize>()) };

                if val.is_nan() {
                    // if bottom-level stem
                    // corresponding leaf node will be leftmost child
                    while stem_idx < self.stems.capacity().div_ceil(2).az::<IDX>() {
                        stem_idx = stem_idx << 1;
                    }

                    let leaf_idx: IDX =
                        stem_idx * 2.az::<IDX>() - self.stems.capacity().az::<IDX>();

                    self.search_leaf_for_best(
                        query,
                        distance_fn,
                        &mut best_item,
                        &mut best_dist,
                        leaf_idx,
                    );

                    return (best_dist, best_item);
                }

                new_off = query[split_dim] - val;

                let left_child_idx = stem_idx << 1;
                let right_child_idx = (stem_idx << 1) + IDX::one();

                #[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "aarch64")))]
                self.prefetch_stems(left_child_idx.az::<usize>());

                let is_left_child =
                    usize::from(*unsafe { query.get_unchecked(split_dim) } < val).az::<IDX>();

                if right_child_idx < self.stems.capacity().az::<IDX>() {
                    [
                        StemIdx::Stem(left_child_idx + (IDX::one() - is_left_child)),
                        StemIdx::Stem(left_child_idx + is_left_child),
                    ]
                    // if *query.get_unchecked(split_dim) < val {
                    //     [ StemIdx::Stem(left_child_idx), StemIdx::Stem(right_child_idx) ]
                    // } else {
                    //     [ StemIdx::Stem(right_child_idx), StemIdx::Stem(left_child_idx) ]
                    // }
                } else {
                    let left_child = if val.is_lsb_set() {
                        StemIdx::DStem(left_child_idx - self.stems.capacity().az::<IDX>())
                    } else {
                        StemIdx::Leaf(left_child_idx - self.stems.capacity().az::<IDX>())
                    };

                    let right_child = if val.is_2lsb_set() {
                        StemIdx::DStem(right_child_idx - self.stems.capacity().az::<IDX>())
                    } else {
                        StemIdx::Leaf(right_child_idx - self.stems.capacity().az::<IDX>())
                    };

                    if *query.get_unchecked(split_dim) < val {
                        [left_child, right_child]
                    } else {
                        [right_child, left_child]
                    }
                }
            }

            StemIdx::DStem(stem_idx) => {
                let node = unsafe { self.dstems.get_unchecked(stem_idx.az::<usize>()) };

                new_off = query[split_dim] - node.split_val;

                let left_child = if KdTree::<A, T, K, B, IDX>::is_stem_index(node.children[0]) {
                    StemIdx::DStem(node.children[0])
                } else {
                    StemIdx::Leaf(node.children[0] - IDX::leaf_offset())
                };

                let right_child = if KdTree::<A, T, K, B, IDX>::is_stem_index(node.children[1]) {
                    StemIdx::DStem(node.children[1])
                } else {
                    StemIdx::Leaf(node.children[1] - IDX::leaf_offset())
                };

                if *unsafe { query.get_unchecked(split_dim) } < node.split_val {
                    [left_child, right_child]
                } else {
                    [right_child, left_child]
                }
            }

            StemIdx::Leaf(leaf_idx) => {
                self.search_leaf_for_best(
                    query,
                    distance_fn,
                    &mut best_item,
                    &mut best_dist,
                    leaf_idx,
                );

                return (best_dist, best_item);
            }
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
        // rd = rd.rd_update(old_off, new_off);

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

        (best_dist, best_item)
    }

    #[inline]
    #[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "aarch64")))]
    fn prefetch_stems(&self, idx: usize) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let prefetch = self.stems.as_ptr().wrapping_offset(2 * idx as isize);
            std::arch::x86_64::_mm_prefetch::<{ core::arch::x86_64::_MM_HINT_T0 }>(ptr::addr_of!(
                prefetch
            )
                as *const i8);
        }

        #[cfg(target_arch = "aarch64")]
        unsafe {
            let prefetch = self.stems.as_ptr().wrapping_offset(2 * idx as isize);
            core::arch::aarch64::_prefetch(
                ptr::addr_of!(prefetch) as *const i8,
                core::arch::aarch64::_PREFETCH_READ,
                core::arch::aarch64::_PREFETCH_LOCALITY3,
            );
        }
    }

    fn search_leaf_for_best<F>(
        &self,
        query: &[A; K],
        distance_fn: &F,
        best_item: &mut T,
        best_dist: &mut A,
        leaf_idx: IDX,
    ) where
        F: Fn(&[A; K], &[A; K]) -> A,
    {
        let leaf_node = unsafe { self.leaves.get_unchecked(leaf_idx.az::<usize>()) };

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
}

#[cfg(test)]
mod tests {
    use crate::float_sss::distance::manhattan;
    use crate::float_sss::kdtree::{Axis, KdTree};
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

        let mut rng = rand::rng();
        for _i in 0..1000 {
            let query_point = [
                rng.random_range(0f32..1f32),
                rng.random_range(0f32..1f32),
                rng.random_range(0f32..1f32),
                rng.random_range(0f32..1f32),
            ];
            let expected = linear_search(&content_to_add, &query_point);

            let result = tree.nearest_one(&query_point, &manhattan);

            // println!("#{}: {} == {}", _i, result.0, expected.0);
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
        assert_eq!(tree.size(), TREE_SIZE);

        let query_points: Vec<[f32; 4]> = (0..NUM_QUERIES)
            .map(|_| rand::random::<[f32; 4]>())
            .collect();

        for (_i, query_point) in query_points.iter().enumerate() {
            let expected = linear_search(&content_to_add, &query_point);

            let result = tree.nearest_one(&query_point, &manhattan);

            // println!("#{}: {} == {}", _i, result.0, expected.0);
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
