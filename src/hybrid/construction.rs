use crate::float_sss::kdtree::{Axis, KdTree, StemNode};
use crate::mirror_select_nth_unstable_by::mirror_select_nth_unstable_by;
use crate::types::{Content, Index};
use az::{Az, Cast};
use std::ops::Rem;

#[derive(Clone)]
enum LeafParent<T> {
    Stem(usize, bool),
    DStem(T, bool),
}

impl<A: Axis, T: Content, const K: usize, const B: usize, IDX: Index<T = IDX>>
    KdTree<A, T, K, B, IDX>
where
    usize: Cast<IDX>,
{
    /// Adds an item to the tree
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
        let mut split_dim = 0;
        let mut is_right_child: bool = false;
        let mut stem_idx: usize = 1;
        let mut parent_idx = 1;
        let mut val: A = A::zero();

        while stem_idx < self.stems.capacity() {
            val = *unsafe { self.stems.get_unchecked(stem_idx) };

            if val.is_nan() {
                // if bottom-level stem
                // corresponding leaf node will be leftmost child
                while stem_idx < self.stems.capacity().div_ceil(2) {
                    stem_idx = stem_idx << 1;
                }

                let leaf_idx: IDX = (stem_idx * 2 - self.stems.capacity()).az::<IDX>();
                return self.add_to_leaf(
                    query,
                    item,
                    split_dim,
                    leaf_idx,
                    LeafParent::Stem(parent_idx, is_right_child),
                );
            }

            parent_idx = stem_idx;
            is_right_child = *unsafe { query.get_unchecked(split_dim) } > val;
            stem_idx = (stem_idx << 1) + usize::from(is_right_child);
            split_dim = (split_dim + 1).rem(K);
        }

        // non-NaN stem val on bottom layer
        if (!is_right_child && val.is_lsb_set()) || (is_right_child && val.is_2lsb_set()) {
            // we're moving to the dstems
            stem_idx = stem_idx - self.stems.capacity();

            while KdTree::<A, T, K, B, IDX>::is_stem_index(stem_idx.az::<IDX>()) {
                let node = unsafe { self.dstems.get_unchecked(stem_idx) };

                parent_idx = stem_idx;
                is_right_child = *unsafe { query.get_unchecked(split_dim) } > val;

                stem_idx = (*unsafe { node.children.get_unchecked(usize::from(is_right_child)) })
                    .az::<usize>();
                split_dim = (split_dim + 1).rem(K);
            }
            let leaf_idx: IDX = stem_idx.az::<IDX>() - IDX::leaf_offset();
            return self.add_to_leaf(
                query,
                item,
                split_dim,
                leaf_idx,
                LeafParent::DStem(parent_idx.az::<IDX>(), is_right_child),
            );
        } else {
            let leaf_idx: IDX = (stem_idx - self.stems.capacity()).az::<IDX>();
            return self.add_to_leaf(
                query,
                item,
                split_dim,
                leaf_idx,
                LeafParent::Stem(parent_idx, is_right_child),
            );
        }
    }

    #[inline]
    pub(crate) fn add_to_optimized(&mut self, query: &[A; K], item: T) {
        assert!(self.optimized_read_only);

        let mut dim = 0;
        let mut idx: usize = 1;
        let mut val: A;

        while idx < self.stems.capacity() {
            val = *unsafe { self.stems.get_unchecked(idx) };

            let is_right_child = *unsafe { query.get_unchecked(dim) } >= val;
            idx = (idx << 1) + usize::from(is_right_child);
            dim = (dim + 1).rem(K);
        }
        idx -= self.stems.len();

        let node_size = (unsafe { self.leaves.get_unchecked_mut(idx) })
            .size
            .az::<usize>();

        if node_size == B {
            println!("Tree Stats: {:?}", self.generate_stats())
        }

        let node = unsafe { self.leaves.get_unchecked_mut(idx) };
        debug_assert!(node.size.az::<usize>() < B);

        *unsafe {
            node.content_points
                .get_unchecked_mut(node.size.az::<usize>())
        } = *query;
        *unsafe {
            node.content_items
                .get_unchecked_mut(node.size.az::<usize>())
        } = item;

        node.size = node.size + IDX::one();
        self.size += 1;
    }

    fn add_to_leaf(
        &mut self,
        query: &[A; K],
        item: T,
        split_dim: usize,
        leaf_idx: IDX,
        parent: LeafParent<IDX>,
    ) {
        let node: &mut super::kdtree::LeafNode<A, T, K, B, IDX>;

        let node_size = unsafe { self.leaves.get_unchecked_mut(leaf_idx.az::<usize>()) }.size;

        if node_size == B.az::<IDX>() {
            let (right_leaf_idx, split_val) = self.split(leaf_idx, split_dim, parent);
            if query[split_dim] > split_val {
                node = unsafe { self.leaves.get_unchecked_mut(right_leaf_idx.az::<usize>()) };
            } else {
                node = unsafe { self.leaves.get_unchecked_mut(leaf_idx.az::<usize>()) };
            }
        } else {
            node = unsafe { self.leaves.get_unchecked_mut(leaf_idx.az::<usize>()) };
        }

        *unsafe {
            node.content_points
                .get_unchecked_mut(node.size.az::<usize>())
        } = *query;
        *unsafe {
            node.content_items
                .get_unchecked_mut(node.size.az::<usize>())
        } = item;

        node.size = node.size + IDX::one();
        self.size += 1;
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

    fn split(&mut self, leaf_idx: IDX, split_dim: usize, parent: LeafParent<IDX>) -> (IDX, A) {
        let mut split_val: A;
        let pivot_idx: IDX;
        {
            let orig = unsafe { self.leaves.get_unchecked_mut(leaf_idx.az::<usize>()) };
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

            split_val = unsafe {
                *orig
                    .content_points
                    .get_unchecked(pivot_idx.az::<usize>())
                    .get_unchecked(split_dim)
            };
        }

        // determine where to copy the upper half of points to
        let mut right_idx;

        match parent {
            LeafParent::Stem(parent_idx, _) if parent_idx == 1 && self.stems[1].is_nan() => {
                // parent is a static stem, root level, first split:

                // 1) update stems to use the root static stem
                //    as the first split plane
                self.stems[1] = split_val;
                self.stems[2] = A::nan();
                self.stems[3] = A::nan();

                // 2) update the destination to copy the split
                //    points to be the left leaf of the left-most
                //    final layer idx of the right child of the
                //    root static stem
                right_idx = 3.az::<IDX>();
                while right_idx <= self.stems.capacity().az::<IDX>() {
                    right_idx = right_idx * 2.az::<IDX>();
                }
                right_idx = right_idx - self.stems.capacity().az::<IDX>();
            }
            LeafParent::Stem(parent_idx, is_right_child)
                if parent_idx < self.stems.capacity().div_ceil(2) =>
            {
                // parent is a static stem, not first split, not bottom layer:

                // 1) if the new stem is at base level, adjust the split
                //    plane value to have it's LSB clear
                if parent_idx >= self.stems.capacity() / 4 {
                    split_val = split_val.with_lsb_clear().with_2lsb_clear();
                }

                let new_stem_idx = (parent_idx << 1) + usize::from(is_right_child);
                self.stems[new_stem_idx] = split_val;

                // if new stem is not bottom layer, add more NaNs
                if new_stem_idx < self.stems.capacity().div_ceil(2) {
                    self.stems[new_stem_idx * 2] = A::nan();
                    self.stems[new_stem_idx * 2 + 1] = A::nan();
                }

                // 4) determine the index of the leaf node we need to
                //    populate with the split-off points
                right_idx = (new_stem_idx * 2 + 1).az::<IDX>();
                while right_idx <= self.stems.capacity().az::<IDX>() {
                    right_idx = right_idx * 2.az::<IDX>();
                }
                right_idx = right_idx - (self.stems.capacity()).az::<IDX>();
            }
            LeafParent::Stem(parent_idx, is_right_child) => {
                // parent is a static stem, bottom level:

                // check to see if the stem is already used:
                if self.stems[parent_idx].is_nan() {
                    // 1) if it isn't, we will use it. Clear the split value
                    //    LSB.

                    split_val = split_val.with_lsb_clear().with_2lsb_clear();
                    self.stems[parent_idx] = split_val;

                    // 4) determine the index of the leaf node we need to
                    //    populate with the split-off points
                    right_idx = (parent_idx * 2 + 1).az::<IDX>();
                    right_idx = right_idx - (self.stems.capacity() + 1).az::<IDX>();
                } else {
                    // 1) allocate space for twice as many dstem nodes
                    //    as there are bottom-level stem values
                    if self.dstems.capacity() == 0 {
                        self.initialise_dstems()
                    }

                    // 2) update the parent's split value to set one of the lowest 2 bits as reqd
                    let existing_split_val = self.stems[parent_idx];
                    if is_right_child {
                        self.stems[parent_idx] = existing_split_val.with_2lsb_set();
                    } else {
                        self.stems[parent_idx] = existing_split_val.with_lsb_set();
                    }
                    //    TODO: check to see if this alters the pivot
                    //    position (perhaps re-run mirror-select-unstable)

                    // 3) insert a dstem at the same index as the leaf
                    //    that we came from. It's left child should be
                    //    the leaf index & MSB, and the right child should
                    //    be the index of a new leaf node & MSB.
                    let new_dstem =
                        unsafe { self.dstems.get_unchecked_mut(leaf_idx.az::<usize>()) };
                    new_dstem.split_val = split_val;
                    new_dstem.children[0] = leaf_idx + IDX::leaf_offset();

                    if self.leaves.capacity() < self.unreserved_leaf_idx + 1 {
                        self.leaves.reserve(1);
                    }
                    right_idx = self.unreserved_leaf_idx.az::<IDX>();
                    self.unreserved_leaf_idx += 1;

                    new_dstem.children[1] = right_idx + IDX::leaf_offset();
                }
            }
            LeafParent::DStem(parent_idx, is_right_child) => {
                // parent is a dynamic stem:

                let leaf_len = self.unreserved_leaf_idx;
                let leaf_cap = self.leaves.capacity();

                // 1) insert a new dstem
                self.dstems.push(StemNode {
                    children: [
                        leaf_idx + IDX::leaf_offset(),
                        (leaf_len.az::<IDX>()) /*- IDX::one()*/ + IDX::leaf_offset(),
                    ],
                    split_val,
                });
                let new_stem_index: IDX = (self.dstems.len() - 1).az::<IDX>();

                let parent_node =
                    unsafe { self.dstems.get_unchecked_mut(parent_idx.az::<usize>()) };
                if is_right_child {
                    parent_node.children[1] = new_stem_index;
                } else {
                    parent_node.children[0] = new_stem_index;
                }

                // 2) insert a new leaf, after any reserved or existing positions.
                if leaf_cap < self.unreserved_leaf_idx + 1 {
                    self.leaves.reserve(1);
                }
                right_idx = self.unreserved_leaf_idx.az::<IDX>();
                self.unreserved_leaf_idx += 1;
            }
        }

        let [right, orig] = unsafe {
            self.leaves
                .get_many_unchecked_mut([right_idx.az::<usize>(), leaf_idx.az::<usize>()])
        };

        unsafe {
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

        right.size = (B.az::<IDX>()) - pivot_idx;
        orig.size = pivot_idx;

        (right_idx, split_val)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        float_sss::kdtree::{FloatLSB, KdTree},
        types::Index,
    };
    use rand::Rng;

    type Flt = f32;

    fn n(num: Flt) -> Flt {
        num
    }

    #[test]
    fn can_add_an_item() {
        let mut tree: KdTree<Flt, u32, 4, 32, u32> = KdTree::new();

        let point: [Flt; 4] = [n(0.1f32), n(0.2f32), n(0.3f32), n(0.4f32)];
        let item = 123;

        tree.add(&point, item);

        assert_eq!(tree.size(), 1);

        // adding to an empty tree with >0 stem nodes will
        // always drop the point in leaf 0
        assert_eq!(tree.leaves[0].content_items[0], 123);
        assert_eq!(&tree.leaves[0].content_points[0], &point);
    }

    #[test]
    fn can_add_items_root_stem_specified() {
        let mut tree: KdTree<Flt, u32, 4, 4, u32> = KdTree::with_capacity(16);

        //      01      Stems
        //   02    03   Stems
        //  0  1  2  3  Leaves

        let point_1: [Flt; 4] = [n(0.1f32), n(0.6f32), n(0.6f32), n(0.6f32)];
        let item_1 = 123;

        let point_2: [Flt; 4] = [n(0.6f32), n(0.1f32), n(0.1f32), n(0.1f32)];
        let item_2 = 456;

        tree.stems[1] = n(0.5f32);
        tree.stems[2] = f32::NAN;
        tree.stems[3] = f32::NAN;

        tree.add(&point_1, item_1);
        assert_eq!(tree.size(), 1);

        // adding to a tree with a val in stem[1] and NaNs in
        // the root's children will put the val in leaf[0] if
        // query[0] < split_val
        assert_eq!(tree.leaves[0].content_items[0], 123);
        assert_eq!(&tree.leaves[0].content_points[0], &point_1);

        tree.add(&point_2, item_2);
        assert_eq!(tree.size(), 2);

        // adding to a tree with a val in stem[1] and NaNs in
        // the root's children will put the val in leaf[2] if
        // query[0] > split_val
        assert_eq!(tree.leaves[2].content_items[0], 456);
        assert_eq!(&tree.leaves[2].content_points[0], &point_2);
    }

    #[test]
    fn can_add_items_root_and_one_base_stem_specified() {
        let mut tree: KdTree<Flt, u32, 4, 4, u32> = KdTree::with_capacity(16);

        //      01      Stems
        //   02    03   Stems
        //  0  1  2  3  Leaves

        let point_1: [Flt; 4] = [n(0.1f32), n(0.6f32), n(0.6f32), n(0.6f32)];
        let item_1 = 123;

        let point_2: [Flt; 4] = [n(0.6f32), n(0.1f32), n(0.1f32), n(0.1f32)];
        let item_2 = 456;

        let point_3: [Flt; 4] = [n(0.6f32), n(0.6f32), n(0.1f32), n(0.1f32)];
        let item_3 = 789;

        tree.stems[1] = n(0.5f32);
        tree.stems[2] = f32::NAN;

        // clearing the 2 LSBs indicates that both the left child and the right
        // child are leaves rather than dstems
        tree.stems[3] = 0.5f32.with_lsb_clear().with_2lsb_clear();

        tree.add(&point_1, item_1);
        assert_eq!(tree.size(), 1);

        // adding to a tree with a val in stem[1] and NaN in
        // the root's left child will put the val in leaf[0] if
        // query[0] < split_val
        assert_eq!(tree.leaves[0].content_items[0], 123);
        assert_eq!(&tree.leaves[0].content_points[0], &point_1);

        tree.add(&point_2, item_2);
        assert_eq!(tree.size(), 2);

        // adding to a tree with a val in stem[1] and a val
        // with clear LSBs in thr root's right child
        // will put the val in leaf[2] if
        // query[0] > stem[1] and query[1] < stem[3]
        assert_eq!(tree.leaves[2].content_items[0], 456);
        assert_eq!(&tree.leaves[2].content_points[0], &point_2);

        tree.add(&point_3, item_3);
        assert_eq!(tree.size(), 3);

        // adding to a tree with a val in stem[1] and a val
        // with clear LSBs in thr root's right child
        // will put the val in leaf[3] if
        // query[0] > stem[1] and query[1] > stem[3]
        assert_eq!(tree.leaves[3].content_items[0], 789);
        assert_eq!(&tree.leaves[3].content_points[0], &point_3);
    }

    #[test]
    fn can_add_items_root_and_both_base_stem_specified_plus_dstems() {
        let mut tree: KdTree<Flt, u32, 4, 4, u32> = KdTree::with_capacity(16);

        //      01      Stems
        //   02    03   Stems
        //  0  1  2  3  Leaves

        let point_1: [Flt; 4] = [n(0.1f32), n(0.1f32), n(0.1f32), n(0.6f32)];
        let item_1 = 111;

        let point_2: [Flt; 4] = [n(0.1f32), n(0.1f32), n(0.6f32), n(0.1f32)];
        let item_2 = 222;

        let point_3: [Flt; 4] = [n(0.1f32), n(0.6f32), n(0.1f32), n(0.1f32)];
        let item_3 = 333;

        let point_4: [Flt; 4] = [n(0.6f32), n(0.6f32), n(0.1f32), n(0.6f32)];
        let item_4 = 444;

        let point_5: [Flt; 4] = [n(0.6f32), n(0.6f32), n(0.6f32), n(0.1f32)];
        let item_5 = 555;

        tree.stems[1] = n(0.5f32);

        // children: left -> dstems, right -> leaves
        tree.stems[2] = 0.5f32.with_lsb_set().with_2lsb_clear();

        // children: left -> leaves, right -> dstems
        tree.stems[3] = 0.5f32.with_lsb_clear().with_2lsb_set();

        // TODO: set up the dstems
        tree.initialise_dstems();
        tree.dstems[0].split_val = 0.5f32;
        tree.dstems[0].children = [
            0 + <u32 as Index>::leaf_offset(),
            5 + <u32 as Index>::leaf_offset(),
        ];

        tree.dstems[3].split_val = 0.5f32;
        tree.dstems[3].children = [2 + <u32 as Index>::leaf_offset(), 2];

        tree.dstems[2].split_val = 0.5f32;
        tree.dstems[2].children = [
            6 + <u32 as Index>::leaf_offset(),
            7 + <u32 as Index>::leaf_offset(),
        ];

        tree.add(&point_1, item_1);
        assert_eq!(tree.size(), 1);

        assert_eq!(tree.leaves[0].content_items[0], 111);
        assert_eq!(&tree.leaves[0].content_points[0], &point_1);

        tree.leaves.reserve(1);
        tree.unreserved_leaf_idx += 1;

        tree.add(&point_2, item_2);
        assert_eq!(tree.size(), 2);

        assert_eq!(
            unsafe { tree.leaves.get_unchecked(5) }.content_items[0],
            222
        );
        assert_eq!(
            &unsafe { tree.leaves.get_unchecked(5) }.content_points[0],
            &point_2
        );

        tree.add(&point_3, item_3);
        assert_eq!(tree.size(), 3);

        assert_eq!(tree.leaves[1].content_items[0], 333);
        assert_eq!(&tree.leaves[1].content_points[0], &point_3);

        tree.add(&point_4, item_4);
        assert_eq!(tree.size(), 4);

        assert_eq!(tree.leaves[2].content_items[0], 444);
        assert_eq!(&tree.leaves[2].content_points[0], &point_4);

        tree.add(&point_5, item_5);
        assert_eq!(tree.size(), 5);

        tree.leaves.reserve(1);
        tree.unreserved_leaf_idx += 1;

        assert_eq!(
            unsafe { tree.leaves.get_unchecked(6) }.content_items[0],
            555
        );
        assert_eq!(
            &unsafe { tree.leaves.get_unchecked(6) }.content_points[0],
            &point_5
        );
    }

    #[test]
    fn can_handle_initial_split_new_item_to_original_bucket() {
        let mut tree: KdTree<Flt, u32, 4, 4, u32> = KdTree::with_capacity(16);

        //      01      Stems
        //   02    03   Stems
        //  0  1  2  3  Leaves

        let point_1: [Flt; 4] = [n(0.12f32), n(0.1f32), n(0.1f32), n(0.6f32)];
        let item_1 = 111;

        let point_2: [Flt; 4] = [n(0.11f32), n(0.1f32), n(0.6f32), n(0.1f32)];
        let item_2 = 222;

        let point_3: [Flt; 4] = [n(0.22f32), n(0.6f32), n(0.1f32), n(0.1f32)];
        let item_3 = 333;

        let point_4: [Flt; 4] = [n(0.21f32), n(0.6f32), n(0.1f32), n(0.6f32)];
        let item_4 = 444;

        // to be added
        let point_5: [Flt; 4] = [n(0.13f32), n(0.6f32), n(0.1f32), n(0.6f32)];
        let item_5 = 555;

        tree.leaves[0].content_items[0] = item_1;
        tree.leaves[0].content_items[1] = item_2;
        tree.leaves[0].content_items[2] = item_3;
        tree.leaves[0].content_items[3] = item_4;
        tree.leaves[0].content_points[0] = point_1.clone();
        tree.leaves[0].content_points[1] = point_2.clone();
        tree.leaves[0].content_points[2] = point_3.clone();
        tree.leaves[0].content_points[3] = point_4.clone();
        tree.leaves[0].size = 4;
        tree.size = 4;

        tree.add(&point_5, item_5);

        assert_eq!(tree.size, 5);

        assert_eq!(tree.stems[1], 0.21f32);
        assert!(tree.stems[2].is_nan());
        assert!(tree.stems[3].is_nan());

        assert_eq!(tree.leaves[0].content_items[0], item_1);
        assert_eq!(tree.leaves[0].content_items[1], item_2);
        assert_eq!(tree.leaves[0].content_items[2], item_5);
        assert_eq!(&tree.leaves[0].content_points[0], &point_1);
        assert_eq!(&tree.leaves[0].content_points[1], &point_2);
        assert_eq!(&tree.leaves[0].content_points[2], &point_5);
        assert_eq!(tree.leaves[0].size, 3);

        assert_eq!(tree.leaves[2].content_items[0], item_4);
        assert_eq!(tree.leaves[2].content_items[1], item_3);
        assert_eq!(&tree.leaves[2].content_points[0], &point_4);
        assert_eq!(&tree.leaves[2].content_points[1], &point_3);
        assert_eq!(tree.leaves[2].size, 2);
    }

    #[test]
    fn can_handle_initial_split_new_item_to_right_bucket() {
        let mut tree: KdTree<Flt, u32, 4, 4, u32> = KdTree::with_capacity(16);

        //      01      Stems
        //   02    03   Stems
        //  0  1  2  3  Leaves

        let point_1: [Flt; 4] = [n(0.12f32), n(0.1f32), n(0.1f32), n(0.6f32)];
        let item_1 = 111;

        let point_2: [Flt; 4] = [n(0.11f32), n(0.1f32), n(0.6f32), n(0.1f32)];
        let item_2 = 222;

        let point_3: [Flt; 4] = [n(0.22f32), n(0.6f32), n(0.1f32), n(0.1f32)];
        let item_3 = 333;

        let point_4: [Flt; 4] = [n(0.21f32), n(0.6f32), n(0.1f32), n(0.6f32)];
        let item_4 = 444;

        // to be added
        let point_5: [Flt; 4] = [n(0.23f32), n(0.6f32), n(0.1f32), n(0.6f32)];
        let item_5 = 555;

        tree.leaves[0].content_items[0] = item_1;
        tree.leaves[0].content_items[1] = item_2;
        tree.leaves[0].content_items[2] = item_3;
        tree.leaves[0].content_items[3] = item_4;
        tree.leaves[0].content_points[0] = point_1.clone();
        tree.leaves[0].content_points[1] = point_2.clone();
        tree.leaves[0].content_points[2] = point_3.clone();
        tree.leaves[0].content_points[3] = point_4.clone();
        tree.leaves[0].size = 4;
        tree.size = 4;

        tree.add(&point_5, item_5);

        assert_eq!(tree.size, 5);

        assert_eq!(tree.stems[1], 0.21f32);
        assert!(tree.stems[2].is_nan());
        assert!(tree.stems[3].is_nan());

        assert_eq!(tree.leaves[0].content_items[0], item_1);
        assert_eq!(tree.leaves[0].content_items[1], item_2);
        assert_eq!(&tree.leaves[0].content_points[0], &point_1);
        assert_eq!(&tree.leaves[0].content_points[1], &point_2);
        assert_eq!(tree.leaves[0].size, 2);

        assert_eq!(tree.leaves[2].content_items[0], item_4);
        assert_eq!(tree.leaves[2].content_items[1], item_3);
        assert_eq!(tree.leaves[2].content_items[2], item_5);
        assert_eq!(&tree.leaves[2].content_points[0], &point_4);
        assert_eq!(&tree.leaves[2].content_points[1], &point_3);
        assert_eq!(&tree.leaves[2].content_points[2], &point_5);
        assert_eq!(tree.leaves[2].size, 3);
    }

    #[test]
    fn can_handle_root_split_new_item_to_right_bucket() {
        let mut tree: KdTree<Flt, u32, 4, 4, u32> = KdTree::with_capacity(16);

        //      01      Stems
        //   02    03   Stems
        //  0  1  2  3  Leaves

        let point_1: [Flt; 4] = [n(0.12f32), n(0.2f32), n(0.1f32), n(0.6f32)];
        let item_1 = 111;

        let point_2: [Flt; 4] = [n(0.11f32), n(0.1f32), n(0.6f32), n(0.1f32)];
        let item_2 = 222;

        let point_3: [Flt; 4] = [n(0.16f32), n(0.6f32), n(0.1f32), n(0.1f32)];
        let item_3 = 333;

        let point_4: [Flt; 4] = [n(0.17f32), n(0.5f32), n(0.1f32), n(0.6f32)];
        let item_4 = 444;

        // to be added
        let point_5: [Flt; 4] = [n(0.18f32), n(0.7f32), n(0.1f32), n(0.6f32)];
        let item_5 = 555;

        tree.stems[1] = 0.21f32;
        tree.stems[2] = f32::NAN;
        tree.stems[3] = f32::NAN;

        tree.leaves[0].content_items[0] = item_1;
        tree.leaves[0].content_items[1] = item_2;
        tree.leaves[0].content_items[2] = item_3;
        tree.leaves[0].content_items[3] = item_4;
        tree.leaves[0].content_points[0] = point_1.clone();
        tree.leaves[0].content_points[1] = point_2.clone();
        tree.leaves[0].content_points[2] = point_3.clone();
        tree.leaves[0].content_points[3] = point_4.clone();
        tree.leaves[0].size = 4;
        tree.size = 4;

        tree.add(&point_5, item_5);

        assert_eq!(tree.size, 5);

        assert_eq!(tree.stems[1], 0.21f32);
        assert_eq!(tree.stems[2], 0.5f32.with_lsb_clear().with_2lsb_clear());
        assert!(tree.stems[3].is_nan());

        assert_eq!(tree.leaves[0].content_items[0], item_1);
        assert_eq!(tree.leaves[0].content_items[1], item_2);
        assert_eq!(&tree.leaves[0].content_points[0], &point_1);
        assert_eq!(&tree.leaves[0].content_points[1], &point_2);
        assert_eq!(tree.leaves[0].size, 2);

        assert_eq!(tree.leaves[1].content_items[0], item_4);
        assert_eq!(tree.leaves[1].content_items[1], item_3);
        assert_eq!(tree.leaves[1].content_items[2], item_5);
        assert_eq!(&tree.leaves[1].content_points[0], &point_4);
        assert_eq!(&tree.leaves[1].content_points[1], &point_3);
        assert_eq!(&tree.leaves[1].content_points[2], &point_5);
        assert_eq!(tree.leaves[1].size, 3);
    }

    #[test]
    fn can_handle_base_split_new_item_to_right_bucket() {
        let mut tree: KdTree<Flt, u32, 4, 4, u32> = KdTree::with_capacity(16);

        //      01      Stems
        //   02    03   Stems
        //  0  1  2  3  Leaves

        let point_1: [Flt; 4] = [n(0.12f32), n(0.2f32), n(0.2f32), n(0.6f32)];
        let item_1 = 111;

        let point_2: [Flt; 4] = [n(0.11f32), n(0.1f32), n(0.1f32), n(0.1f32)];
        let item_2 = 222;

        let point_3: [Flt; 4] = [n(0.16f32), n(0.4f32), n(0.4f32), n(0.1f32)];
        let item_3 = 333;

        let point_4: [Flt; 4] = [n(0.17f32), n(0.3f32), n(0.3f32), n(0.6f32)];
        let item_4 = 444;

        // to be added
        let point_5: [Flt; 4] = [n(0.18f32), n(0.45f32), n(0.7f32), n(0.6f32)];
        let item_5 = 555;

        tree.stems[1] = 0.21f32;
        tree.stems[2] = 0.5f32.with_lsb_clear().with_2lsb_clear();
        tree.stems[3] = f32::NAN;

        tree.leaves[0].content_items[0] = item_1;
        tree.leaves[0].content_items[1] = item_2;
        tree.leaves[0].content_items[2] = item_3;
        tree.leaves[0].content_items[3] = item_4;
        tree.leaves[0].content_points[0] = point_1.clone();
        tree.leaves[0].content_points[1] = point_2.clone();
        tree.leaves[0].content_points[2] = point_3.clone();
        tree.leaves[0].content_points[3] = point_4.clone();
        tree.leaves[0].size = 4;
        tree.size = 4;

        tree.add(&point_5, item_5);

        assert_eq!(tree.size, 5);

        assert_eq!(tree.stems[1], 0.21f32);
        assert_eq!(tree.stems[2], 0.5f32.with_lsb_set().with_2lsb_clear());
        assert!(tree.stems[3].is_nan());

        assert_eq!(tree.dstems[0].split_val, 0.3f32);
        assert_eq!(
            tree.dstems[0].children,
            [
                0 + <u32 as Index>::leaf_offset(),
                4 + <u32 as Index>::leaf_offset()
            ]
        );

        assert_eq!(tree.leaves[0].content_items[0], item_1);
        assert_eq!(tree.leaves[0].content_items[1], item_2);
        assert_eq!(&tree.leaves[0].content_points[0], &point_1);
        assert_eq!(&tree.leaves[0].content_points[1], &point_2);
        assert_eq!(tree.leaves[0].size, 2);

        assert_eq!(
            unsafe { tree.leaves.get_unchecked(4) }.content_items[0],
            item_4
        );
        assert_eq!(
            unsafe { tree.leaves.get_unchecked(4) }.content_items[1],
            item_3
        );
        assert_eq!(
            unsafe { tree.leaves.get_unchecked(4) }.content_items[2],
            item_5
        );
        assert_eq!(
            &unsafe { tree.leaves.get_unchecked(4) }.content_points[0],
            &point_4
        );
        assert_eq!(
            &unsafe { tree.leaves.get_unchecked(4) }.content_points[1],
            &point_3
        );
        assert_eq!(
            &unsafe { tree.leaves.get_unchecked(4) }.content_points[2],
            &point_5
        );
        assert_eq!(unsafe { tree.leaves.get_unchecked(4) }.size, 3);
    }

    #[test]
    fn can_handle_dstem_split_new_item_to_right_bucket() {
        let mut tree: KdTree<Flt, u32, 4, 4, u32> = KdTree::with_capacity(16);
        ///// BEFORE /////////////////////////
        //       01       Stems
        //   02      03   Stems
        //  D0   1  2  3  Dstems / Leaves
        // 0  4           Leaves

        ////// AFTER /////////////////////////
        //           01       Stems
        //       02      03   Stems
        //    D0    1   2  3  Dstems / Leaves
        //  D4   4            DStems / Leaves (D1-D3 reserved)
        // 0  5               Leaves

        let point_1: [Flt; 4] = [n(0.12f32), n(0.2f32), n(0.2f32), n(0.102f32)];
        let item_1 = 111;

        let point_2: [Flt; 4] = [n(0.11f32), n(0.1f32), n(0.1f32), n(0.101f32)];
        let item_2 = 222;

        let point_3: [Flt; 4] = [n(0.16f32), n(0.4f32), n(0.4f32), n(0.106f32)];
        let item_3 = 333;

        let point_4: [Flt; 4] = [n(0.17f32), n(0.3f32), n(0.3f32), n(0.107f32)];
        let item_4 = 444;

        // to be added
        let point_5: [Flt; 4] = [n(0.18f32), n(0.45f32), n(0.45f32), n(0.108f32)];
        let item_5 = 555;

        tree.stems[1] = 0.21f32;
        tree.stems[2] = 0.5f32.with_lsb_set().with_2lsb_clear();
        tree.stems[3] = f32::NAN;

        tree.initialise_dstems();
        tree.dstems[0].split_val = 0.3f32;
        tree.dstems[0].children = [
            0 + <u32 as Index>::leaf_offset(),
            4 + <u32 as Index>::leaf_offset(),
        ];

        tree.leaves.reserve(1);
        tree.unreserved_leaf_idx += 1;

        tree.leaves[0].content_items[0] = item_1;
        tree.leaves[0].content_items[1] = item_2;
        tree.leaves[0].content_items[2] = item_3;
        tree.leaves[0].content_items[3] = item_4;
        tree.leaves[0].content_points[0] = point_1.clone();
        tree.leaves[0].content_points[1] = point_2.clone();
        tree.leaves[0].content_points[2] = point_3.clone();
        tree.leaves[0].content_points[3] = point_4.clone();
        tree.leaves[0].size = 4;
        tree.size = 4;

        tree.add(&point_5, item_5);

        assert_eq!(tree.size, 5);

        assert_eq!(tree.stems[1], 0.21f32);
        assert_eq!(tree.stems[2], 0.5f32.with_lsb_set().with_2lsb_clear());
        assert!(tree.stems[3].is_nan());

        assert_eq!(tree.dstems[0].split_val, 0.3f32);
        assert_eq!(
            tree.dstems[0].children,
            [4, 4 + <u32 as Index>::leaf_offset()]
        );

        assert_eq!(tree.dstems[4].split_val, 0.106f32);
        assert_eq!(
            tree.dstems[4].children,
            [
                0 + <u32 as Index>::leaf_offset(),
                5 + <u32 as Index>::leaf_offset()
            ]
        );

        assert_eq!(tree.leaves[0].content_items[0], item_1);
        assert_eq!(tree.leaves[0].content_items[1], item_2);
        assert_eq!(&tree.leaves[0].content_points[0], &point_1);
        assert_eq!(&tree.leaves[0].content_points[1], &point_2);
        assert_eq!(tree.leaves[0].size, 2);

        assert_eq!(
            unsafe { tree.leaves.get_unchecked(5) }.content_items[0],
            item_3
        );
        assert_eq!(
            unsafe { tree.leaves.get_unchecked(5) }.content_items[1],
            item_4
        );
        assert_eq!(
            unsafe { tree.leaves.get_unchecked(5) }.content_items[2],
            item_5
        );
        assert_eq!(
            &unsafe { tree.leaves.get_unchecked(5) }.content_points[0],
            &point_3
        );
        assert_eq!(
            &unsafe { tree.leaves.get_unchecked(5) }.content_points[1],
            &point_4
        );
        assert_eq!(
            &unsafe { tree.leaves.get_unchecked(5) }.content_points[2],
            &point_5
        );
        assert_eq!(unsafe { tree.leaves.get_unchecked(5) }.size, 3);
    }

    #[test]
    fn can_handle_interior_stem_split_new_item_to_right_bucket() {
        let mut tree: KdTree<Flt, u32, 4, 4, u32> = KdTree::with_capacity(32);

        //             01              Stems
        //      02            03       Stems
        //  04     05     06     07    Stems
        // 0  1   2  3   4  5   6  7   Leaves

        let point_1: [Flt; 4] = [n(0.12f32), n(0.2f32), n(0.2f32), n(0.6f32)];
        let item_1 = 111;

        let point_2: [Flt; 4] = [n(0.11f32), n(0.1f32), n(0.1f32), n(0.1f32)];
        let item_2 = 222;

        let point_3: [Flt; 4] = [n(0.16f32), n(0.4f32), n(0.4f32), n(0.1f32)];
        let item_3 = 333;

        let point_4: [Flt; 4] = [n(0.17f32), n(0.3f32), n(0.3f32), n(0.6f32)];
        let item_4 = 444;

        // to be added
        let point_5: [Flt; 4] = [n(0.18f32), n(0.45f32), n(0.7f32), n(0.6f32)];
        let item_5 = 555;

        tree.stems[1] = 0.21f32;
        tree.stems[2] = 0.5f32;
        tree.stems[3] = f32::NAN;
        tree.stems[4] = f32::NAN;
        tree.stems[5] = f32::NAN;

        tree.leaves[0].content_items[0] = item_1;
        tree.leaves[0].content_items[1] = item_2;
        tree.leaves[0].content_items[2] = item_3;
        tree.leaves[0].content_items[3] = item_4;
        tree.leaves[0].content_points[0] = point_1.clone();
        tree.leaves[0].content_points[1] = point_2.clone();
        tree.leaves[0].content_points[2] = point_3.clone();
        tree.leaves[0].content_points[3] = point_4.clone();
        tree.leaves[0].size = 4;
        tree.size = 4;

        tree.add(&point_5, item_5);

        assert_eq!(tree.size, 5);

        assert_eq!(tree.stems[1], 0.21f32);
        assert_eq!(tree.stems[2], 0.5f32);
        assert!(tree.stems[3].is_nan());
        assert_eq!(tree.stems[4], 0.3f32.with_lsb_clear().with_2lsb_clear());
        assert!(tree.stems[5].is_nan());

        assert_eq!(tree.leaves[0].content_items[0], item_1);
        assert_eq!(tree.leaves[0].content_items[1], item_2);
        assert_eq!(&tree.leaves[0].content_points[0], &point_1);
        assert_eq!(&tree.leaves[0].content_points[1], &point_2);
        assert_eq!(tree.leaves[0].size, 2);

        assert_eq!(tree.leaves[1].content_items[0], item_4);
        assert_eq!(tree.leaves[1].content_items[1], item_3);
        assert_eq!(tree.leaves[1].content_items[2], item_5);
        assert_eq!(&tree.leaves[1].content_points[0], &point_4);
        assert_eq!(&tree.leaves[1].content_points[1], &point_3);
        assert_eq!(&tree.leaves[1].content_points[2], &point_5);
        assert_eq!(tree.leaves[1].size, 3);
    }

    #[test]
    fn can_add_enough_items_to_cause_a_split() {
        let mut tree: KdTree<Flt, u32, 4, 4, u32> = KdTree::with_capacity(16);

        let content_to_add: [([Flt; 4], u32); 16] = [
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
        let mut tree: KdTree<Flt, u32, 4, 4, u32> = KdTree::new();

        let content_to_add: [([Flt; 4], u32); 16] = [
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
        let mut tree: KdTree<Flt, u32, 4, 4, u32> = KdTree::with_capacity(1000);

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
