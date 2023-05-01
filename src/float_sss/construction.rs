use crate::float_sss::kdtree::{Axis, KdTree, StemNode};
use crate::mirror_select_nth_unstable_by::mirror_select_nth_unstable_by;
use crate::types::{Content, Index};
use az::{Az, Cast};
use std::ops::Rem;


// TODO: change from usize to IDX
enum LeafParent<T> {
    Stem(T, bool),
    DStem(T, bool)
}

// TODO: change from usizes to IDX?
enum TraverseState<T> {
    ValidStem(T, LeafParent<T>),
    DStem(T),
    Leaf(T, LeafParent<T>)
}


impl<A: Axis, T: Content, const K: usize, const B: usize, IDX: Index<T = IDX>>
    KdTree<A, T, K, B, IDX>
where
    usize: Cast<IDX>,
{
    /// Adds an item to the tree.
    ///
    /// The first argument specifies co-ordinates of the point where the item is located.
    /// The second argument is an integer identifier / index for the item being stored.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kiddo::float::kdtree::KdTree;
    ///
    /// let mut tree: KdTree<f64, u32, 3, 32, u32> = KdTree::new();
    ///
    /// tree.add(&[1.0, 2.0, 5.0], 100);
    ///
    /// assert_eq!(tree.size(), 1);
    /// ```
    #[inline]
    pub fn add(&mut self, query: &[A; K], item: T) {
        let mut trav_state = TraverseState::ValidStem(IDX::one(), LeafParent::Stem(IDX::one(), true));
        let mut split_dim = 0;
        let mut is_left_child: bool = false;

        loop {
            match trav_state {
                TraverseState::ValidStem(mut idx, mut parent) => {
                    let val = self.stems[idx.az::<usize>() - 1];

                    if val.is_nan() || idx > (self.stems.capacity() / 2 + 1).az::<IDX>() {
                        // if this stem value is NaN, we are a bottom-level stem

                        // corresponding leaf node will be leftmost child
                        while idx < self.stems.capacity().az::<IDX>().div_ceil(2.az::<IDX>()) {
                            idx = idx * 2.az::<IDX>();
                        }

                        trav_state = TraverseState::Leaf(idx * 2.az::<IDX>() - IDX::one() - self.stems.capacity().az::<IDX>(), parent);
                    } else {

                        let next = if query[split_dim] < val {
                            is_left_child = true;
                            idx * 2.az::<IDX>()
                        } else {
                            is_left_child = false;
                            idx * 2.az::<IDX>() + IDX::one()
                        };
                        split_dim = (split_dim + 1).rem(K);

                        if idx < (self.stems.capacity() / 2 + 1).az::<IDX>() {
                            // non-nan stem val, not on bottom layer:

                            trav_state = TraverseState::ValidStem(next, LeafParent::Stem(idx, is_left_child));
                        } else {
                            // non-NaN stem val on bottom layer
                            let next = next - self.stems.capacity().az::<IDX>();

                            if (is_left_child && val.is_lsb_set()) || (!is_left_child && val.is_2lsb_set()) {
                                trav_state = TraverseState::DStem(next - IDX::one());
                            } else {
                                trav_state = TraverseState::Leaf(next - IDX::one(), LeafParent::Stem(idx, is_left_child));
                            }
                        }
                    }
                },

                TraverseState::DStem(mut idx) => {
                    let mut parent_idx = idx;

                    while KdTree::<A, T, K, B, IDX>::is_stem_index(idx) {
                        let node = &self.dstems[idx.az::<usize>()];

                        parent_idx = idx;
                        idx = if query[split_dim] <= node.split_val {
                            is_left_child = true;
                            node.left
                        } else {
                            is_left_child = false;
                            node.right
                        };

                        split_dim = (split_dim + 1).rem(K);
                    }

                    trav_state = TraverseState::Leaf(idx - IDX::leaf_offset(), LeafParent::DStem(parent_idx, is_left_child));
                },

                TraverseState::Leaf(mut idx, parent) => {
                    if idx >= IDX::leaf_offset() {
                        idx = idx - IDX::leaf_offset();
                    }

                    let mut node = unsafe { self.leaves.get_unchecked_mut(idx.az::<usize>()) };

                    if node.size == B.az::<IDX>() {
                        trav_state = self.split(idx, split_dim, parent);
                    } else {
                        node.content_points[node.size.az::<usize>()] = *query;
                        node.content_items[node.size.az::<usize>()] = item;

                        node.size = node.size + IDX::one();
                        self.size = self.size + T::one();

                        return;
                    }
                },
            }
        }
    }

    /// Removes an item from the tree.
    ///
    /// The first argument specifies co-ordinates of the point where the item is located.
    /// The second argument is the integer identifier / index for the stored item.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kiddo::float::kdtree::KdTree;
    ///
    /// let mut tree: KdTree<f64, u32, 3, 32, u32> = KdTree::new();
    ///
    /// tree.add(&[1.0, 2.0, 5.0], 100);
    /// tree.add(&[1.0, 2.0, 5.0], 200);
    /// assert_eq!(tree.size(), 2);
    ///
    /// tree.remove(&[1.0, 2.0, 5.0], 100);
    /// assert_eq!(tree.size(), 1);
    ///
    /// tree.remove(&[1.0, 2.0, 5.0], 200);
    /// assert_eq!(tree.size(), 0);
    /// ```
/*     #[inline]
    pub fn remove(&mut self, query: &[A; K], item: T) -> usize {
        let mut stem_idx = self.root_index;
        let mut split_dim = 0;
        let mut removed: usize = 0;

        while KdTree::<A, T, K, B, IDX>::is_stem_index(stem_idx) {
            let Some(stem_node) = self.stems.get_mut(stem_idx.az::<usize>()) else {
                return removed;
            };

            stem_idx = if query[split_dim] <= stem_node.split_val {
                stem_node.left
            } else {
                stem_node.right
            };

            split_dim = (split_dim + 1).rem(K);
        }

        let leaf_idx = stem_idx - IDX::leaf_offset();

        if let Some(mut leaf_node) = self.leaves.get_mut(leaf_idx.az::<usize>()) {
            let mut p_index = 0;
            while p_index < leaf_node.size.az::<usize>() {
                if &leaf_node.content_points[p_index] == query
                    && leaf_node.content_items[p_index] == item
                {
                    leaf_node.content_points[p_index] =
                        leaf_node.content_points[leaf_node.size.az::<usize>() - 1];
                    leaf_node.content_items[p_index] =
                        leaf_node.content_items[leaf_node.size.az::<usize>() - 1];

                    self.size -= T::one();
                    removed += 1;
                    leaf_node.size = leaf_node.size - IDX::one();
                } else {
                    p_index += 1;
                }
            }
        }

        removed
    } */

    fn split(
        &mut self,
        leaf_idx: IDX,
        split_dim: usize,
        parent: LeafParent<IDX>,
    ) -> TraverseState<IDX> {

        let mut split_val: A;
        let pivot_idx: IDX;
        {
            let orig = unsafe {
                self.leaves.get_unchecked_mut(leaf_idx.az::<usize>())
            };
            pivot_idx = (B / 2).az::<IDX>();

            // partially sort original leaf so that first half of content
            // is sorted ascending
            mirror_select_nth_unstable_by(
                &mut orig.content_points,
                &mut orig.content_items,
                pivot_idx.az::<usize>(),
                |a, b| unsafe {
                    a.get_unchecked(split_dim)
                    .partial_cmp(b.get_unchecked(split_dim))
                    .expect("Leaf node sort failed.")
                },
            );

            split_val = unsafe { *orig
                .content_points
                .get_unchecked(pivot_idx.az::<usize>())
                .get_unchecked(split_dim) };
        }

        // determine where to copy the upper half of points to
        let mut right_idx;
        let result: TraverseState<IDX>;

        match parent {
            LeafParent::Stem(parent_idx , _) if parent_idx == IDX::one() && self.stems[0].is_nan() => {
                // parent is a static stem, root level, first split:

                // 1) update stems to use the root static stem
                //    as the first split plane
                self.stems[0] = split_val;
                self.stems[1] = A::nan();
                self.stems[2] = A::nan();

                // 2) update the destination to copy the split
                //    points to be the left leaf of the left-most
                //    final layer idx of the right child of the
                //    root static stem
                right_idx = 3.az::<IDX>();
                while right_idx <= self.stems.capacity().az::<IDX>() {
                    right_idx = right_idx * 2.az::<IDX>();
                }
                right_idx = right_idx - (self.stems.capacity() + 1).az::<IDX>();

                result = TraverseState::ValidStem(IDX::one(), LeafParent::Stem(IDX::one(), true));
            },

            LeafParent::Stem(parent_idx, is_left_child ) if parent_idx < (self.stems.capacity().div_ceil(2)).az::<IDX>() => {
                // parent is a static stem, not first split, not bottom layer:

                // 1) if the new stem is at base level, adjust the split
                //    plane value to have it's LSB clear
                if parent_idx >= (self.stems.capacity() / 4).az::<IDX>() {
                    split_val = split_val.with_lsb_clear().with_2lsb_clear();
                }

                let new_stem_idx =  if is_left_child {
                    // 2) if leaf was parent's left descendant, set the parents
                    //    left child as a newly active stem node
                    parent_idx.az::<usize>() * 2

                } else {
                    // 3) alternatively, set the parents right child as a newly
                    //    active stem node
                    parent_idx.az::<usize>() * 2 + 1
                };
                self.stems[new_stem_idx - 1] = split_val;

                // if new stem is not bottom layer, add mode NaNs
                if new_stem_idx < self.stems.capacity().div_ceil(2) {
                    self.stems[new_stem_idx * 2 - 1] = A::nan();
                    self.stems[new_stem_idx * 2] = A::nan();
                }

                // 4) determine the index of the leaf node we need to
                //    populate with the split-off points
                right_idx = (new_stem_idx * 2 + 1).az::<IDX>();
                while right_idx <= self.stems.capacity().az::<IDX>() {
                    right_idx = right_idx *  2.az::<IDX>();
                }
                right_idx = right_idx - (self.stems.capacity() + 1).az::<IDX>();

                result = TraverseState::ValidStem(new_stem_idx.az::<IDX>(), LeafParent::Stem(new_stem_idx.az::<IDX>(), is_left_child));
            },
            LeafParent::Stem(parent_idx, is_left_child) => {
                // parent is a static stem, bottom level:

                // check to see if the stem is already used:
                if self.stems[parent_idx.az::<usize>() - 1].is_nan() {
                    // 1) if it isn't, we will use it. Clear the split value
                    //    LSB.

                    split_val = split_val.with_lsb_clear().with_2lsb_clear();
                    self.stems[parent_idx.az::<usize>() - 1] = split_val;

                    // 4) determine the index of the leaf node we need to
                    //    populate with the split-off points
                    right_idx = (parent_idx.az::<usize>() * 2 + 1).az::<IDX>();
                    right_idx = right_idx - (self.stems.capacity() + 1).az::<IDX>();

                    result = TraverseState::ValidStem(parent_idx, LeafParent::Stem(parent_idx, is_left_child));
                } else {

                    // 1) allocate space for twice as many dstem nodes
                    //    as there are bottom-level stem values
                    if self.dstems.capacity() == 0 {
                        self.initialise_dstems()
                    }

                    // 2) update the parent's split value to set one of the lowest 2 bits as reqd
                    let existing_split_val = self.stems[parent_idx.az::<usize>() - 1];
                    if is_left_child {
                        self.stems[parent_idx.az::<usize>() - 1] = existing_split_val.with_lsb_set();
                    } else {
                        self.stems[parent_idx.az::<usize>() - 1] = existing_split_val.with_2lsb_set();
                    }
                    //    TODO: check to see if this alters the pivot
                    //    position (perhaps re-run mirror-select-unstable)

                    // 3) insert a dstem at the same index as the leaf
                    //    that we came from. It's left child should be
                    //    the leaf index & MSB, and the right child should
                    //    be the index of a new leaf node & MSB.
                    let new_dstem = unsafe { self.dstems.get_unchecked_mut(leaf_idx.az::<usize>()) };
                    new_dstem.split_val = split_val;
                    new_dstem.left = leaf_idx + IDX::leaf_offset();

                    if self.leaves.capacity() < self.unreserved_leaf_idx + 1 {
                        self.leaves.reserve(1);
                    }
                    right_idx = self.unreserved_leaf_idx.az::<IDX>();
                    self.unreserved_leaf_idx += 1;

                    new_dstem.right = right_idx + IDX::leaf_offset();

                    result = TraverseState::DStem(leaf_idx);
                }
            },
            LeafParent::DStem(parent_idx, was_parents_left) => {
                // parent is a dynamic stem:

                // TODO: should be self.unreserved_leaf_idx
                // let leaf_len = self.leaves.len();
                let leaf_len = self.unreserved_leaf_idx;
                let leaf_cap = self.leaves.capacity();

                // 1) insert a new dstem
                self.dstems.push(StemNode {
                    left: leaf_idx + IDX::leaf_offset(),
                    right: (leaf_len.az::<IDX>()) /*- IDX::one()*/ + IDX::leaf_offset(),
                    split_val,
                });
                let new_stem_index: IDX = (self.dstems.len() - 1).az::<IDX>();

                let parent_node = unsafe { self.dstems.get_unchecked_mut(parent_idx.az::<usize>()) };
                if was_parents_left {
                    parent_node.left = new_stem_index;
                } else {
                    parent_node.right = new_stem_index;
                }

                // 2) insert a new leaf, after any reserved or existing positions.
                if leaf_cap < self.unreserved_leaf_idx + 1 {
                    self.leaves.reserve(1);
                }
                right_idx = self.unreserved_leaf_idx.az::<IDX>();
                self.unreserved_leaf_idx += 1;

                // 3) update right_idx to the new leaf.
                result = TraverseState::DStem(new_stem_index)
            }
        }

        let [mut right, mut orig] =  unsafe { self.leaves.get_many_unchecked_mut([right_idx.az::<usize>(), leaf_idx.az::<usize>()]) };

        unsafe {
            if B.rem(2) == 1 {

                right
                    .content_points
                    .get_unchecked_mut(..((pivot_idx + IDX::one()).az::<usize>()))
                    .copy_from_slice(
                        orig.content_points
                            .get_unchecked((pivot_idx.az::<usize>())..),
                    );
                right
                    .content_items
                    .get_unchecked_mut(..((pivot_idx + IDX::one()).az::<usize>()))
                    .copy_from_slice(
                        orig.content_items
                            .get_unchecked((pivot_idx.az::<usize>())..),
                    );

            } else {

                right
                .content_points
                .get_unchecked_mut(..(pivot_idx.az::<usize>()))
                .copy_from_slice(
                    orig.content_points
                    .get_unchecked((pivot_idx.az::<usize>())..),
                );
                right
                .content_items
                .get_unchecked_mut(..(pivot_idx.az::<usize>()))
                .copy_from_slice(
                    orig.content_items
                    .get_unchecked((pivot_idx.az::<usize>())..),
                );
            }
        }

        right.size = (B.az::<IDX>()) - pivot_idx;
        orig.size = pivot_idx;

        result
    }
}

#[cfg(test)]
mod tests {
    use crate::float_sss::kdtree::KdTree;
    use rand::Rng;

    type FLT = f32;

    fn n(num: FLT) -> FLT {
        num
    }

    #[test]
    fn can_add_an_item() {
        let mut tree: KdTree<FLT, u32, 4, 32, u32> = KdTree::new();

        let point: [FLT; 4] = [n(0.1f32), n(0.2f32), n(0.3f32), n(0.4f32)];
        let item = 123;

        tree.add(&point, item);

        assert_eq!(tree.size(), 1);
    }

    #[test]
    fn can_add_enough_items_to_cause_a_split() {
        let mut tree: KdTree<FLT, u32, 4, 4, u32> = KdTree::with_capacity(16);

        let content_to_add: [([FLT; 4], u32); 16] = [
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
    }

    /* #[test]
    fn can_remove_an_item() {
        let mut tree: KdTree<FLT, u32, 4, 4, u32> = KdTree::new();

        let content_to_add: [([FLT; 4], u32); 16] = [
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

        let removed = tree.remove(&[n(0.9f32), n(0.0f32), n(0.9f32), n(0.0f32)], 9);

        assert_eq!(removed, 1);
        assert_eq!(tree.size(), 15);
    } */

    #[test]
    fn can_add_shitloads_of_points() {
        let mut tree: KdTree<FLT, u32, 4, 4, u32> = KdTree::with_capacity(1000);

        let mut rng = rand::thread_rng();
        for i in 0..1000 {
            let point = [
                n(rng.gen_range(0f32..0.99998f32)),
                n(rng.gen_range(0f32..0.99998f32)),
                n(rng.gen_range(0f32..0.99998f32)),
                n(rng.gen_range(0f32..0.99998f32)),
            ];

            tree.add(&point, i);
        }

        assert_eq!(tree.size(), 1000);
    }

    #[test]
    fn can_add_shitloads_of_random_points() {
        fn rand_data_2d() -> ([f64; 2], u32) {
            rand::random()
        }

        let points_to_add: Vec<([f64; 2], u32)> =
            (0..100_000).into_iter().map(|_| rand_data_2d()).collect();

        let mut points = vec![];
        let mut kdtree = KdTree::<f64, u32, 2, 32, u32>::with_capacity(200_000);
        for _ in 0..100_000 {
            points.push(rand_data_2d());
        }
        for i in 0..points.len() {
            kdtree.add(&points[i].0, points[i].1);
        }

        points_to_add
            .iter()
            .for_each(|point| kdtree.add(&point.0, point.1));

            assert_eq!(kdtree.size(), 200_000);
    }

    /* #[test]
    fn test_can_handle_remove_edge_case_from_issue_12() {
        // See: https://github.com/sdd/kiddo/issues/12
        let pts = vec![
            [19.2023, 7.1812],
            [7.6427, 22.5779],
            [26.6314, 34.8920],
            [36.7890, 27.2663],
            [28.3226, 8.5047],
            [5.3914, 28.1360],
            [5.0978, 3.6814],
            [0.5114, 11.6552],
            [4.7981, 21.6210],
            [29.0030, 9.6799],
            [35.5580, 1.8891],
            [3.9160, 25.5702],
            [22.2497, 31.6140],
            [30.7110, 36.7514],
            [0.2828, 12.4298],
            [20.0206, 3.0635],
            [30.6153, 2.8582],
            [23.7179, 6.2048],
            [13.0438, 4.2319],
            [4.6433, 30.9660],
            [5.0588, 5.2028],
            [19.2023, 23.7406],
            [37.3171, 32.7523],
            [12.6957, 15.7080],
            [15.6001, 14.3995],
            [36.0203, 21.0366],
            [6.3956, 2.7644],
            [3.1719, 8.7039],
            [0.9159, 12.2299],
            [23.8157, 14.0699],
            [27.7757, 7.3597],
            [28.4198, 31.3427],
            [2.3290, 6.2364],
            [10.1126, 7.7009],
        ];

        let mut tree = KdTree::<f64, usize, 2, 32, u32>::new();

        for (i, pt) in pts.iter().enumerate() {
            tree.add(pt, i);
        }

        assert_eq!(tree.remove(&pts[0], 0), 1);
    } */
}
