use crate::fixed::kdtree::{Axis, KdTree, LeafNode, StemNode};
use crate::mirror_select_nth_unstable_by::mirror_select_nth_unstable_by;
use crate::types::{Content, Index};
use az::{Az, Cast};
use std::ops::Rem;

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
    /// use fixed::FixedU16;
    /// use fixed::types::extra::U0;
    /// use kiddo::fixed::kdtree::KdTree;
    ///
    /// type FXD = FixedU16<U0>;
    ///
    /// let mut tree: KdTree<FXD, u32, 3, 32, u32> = KdTree::with_capacity(1_000_000);
    ///
    /// tree.add(&[FXD::from_num(1), FXD::from_num(2), FXD::from_num(5)], 100);
    /// tree.add(&[FXD::from_num(2), FXD::from_num(3), FXD::from_num(6)], 101);
    ///
    /// assert_eq!(tree.size(), 2);
    /// ```
    #[inline]
    pub fn add(&mut self, query: &[A; K], item: T) {
        unsafe {
            let mut stem_idx = self.root_index;
            let mut split_dim = 0;
            let mut stem_node;
            let mut parent_idx = <IDX as Index>::max();
            let mut is_left_child: bool = false;

            while KdTree::<A, T, K, B, IDX>::is_stem_index(stem_idx) {
                parent_idx = stem_idx;
                stem_node = self.stems.get_unchecked_mut(stem_idx.az::<usize>());

                stem_idx = if *query.get_unchecked(split_dim) < stem_node.split_val {
                    is_left_child = true;
                    stem_node.left
                } else {
                    is_left_child = false;
                    stem_node.right
                };

                split_dim = (split_dim + 1).rem(K);
            }

            let mut leaf_idx = stem_idx - IDX::leaf_offset();
            let mut leaf_node = self.leaves.get_unchecked_mut(leaf_idx.az::<usize>());

            if leaf_node.size == B.az::<IDX>() {
                stem_idx = self.split(leaf_idx, split_dim, parent_idx, is_left_child);
                let node = self.stems.get_unchecked_mut(stem_idx.az::<usize>());

                leaf_idx = (if *query.get_unchecked(split_dim) < node.split_val {
                    node.left
                } else {
                    node.right
                } - IDX::leaf_offset());

                leaf_node = self.leaves.get_unchecked_mut(leaf_idx.az::<usize>());
            }

            *leaf_node
                .content_points
                .get_unchecked_mut(leaf_node.size.az::<usize>()) = *query;
            *leaf_node
                .content_items
                .get_unchecked_mut(leaf_node.size.az::<usize>()) = item;

            leaf_node.size = leaf_node.size + IDX::one();
        }
        self.size = self.size + T::one();
    }

    /// Removes an item from the tree.
    ///
    /// The first argument specifies co-ordinates of the point where the item is located.
    /// The second argument is the integer identifier / index for the stored item.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fixed::FixedU16;
    /// use fixed::types::extra::U0;
    /// use kiddo::fixed::kdtree::KdTree;
    ///
    /// type FXD = FixedU16<U0>;
    ///
    /// let mut tree: KdTree<FXD, u32, 3, 32, u32> = KdTree::with_capacity(1_000_000);
    ///
    /// tree.add(&[FXD::from_num(1), FXD::from_num(2), FXD::from_num(5)], 100);
    /// tree.add(&[FXD::from_num(1), FXD::from_num(2), FXD::from_num(5)], 101);
    /// assert_eq!(tree.size(), 2);
    ///
    /// tree.remove(&[FXD::from_num(1), FXD::from_num(2), FXD::from_num(5)], 100);
    /// assert_eq!(tree.size(), 1);
    ///
    /// tree.remove(&[FXD::from_num(1), FXD::from_num(2), FXD::from_num(5)], 101);
    /// assert_eq!(tree.size(), 0);
    /// ```
    #[inline]
    pub fn remove(&mut self, query: &[A; K], item: T) -> usize {
        let mut stem_idx = self.root_index;
        let mut split_dim = 0;
        let mut removed: usize = 0;

        while KdTree::<A, T, K, B, IDX>::is_stem_index(stem_idx) {
            let Some(stem_node) = self.stems.get_mut(stem_idx.az::<usize>()) else {
                return removed;
            };

            stem_idx = if query[split_dim] < stem_node.split_val {
                stem_node.left
            } else {
                stem_node.right
            };

            split_dim = (split_dim + 1).rem(K);
        }

        let leaf_idx = stem_idx - IDX::leaf_offset();

        if let Some(leaf_node) = self.leaves.get_mut(leaf_idx.az::<usize>()) {
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
    }

    unsafe fn split(
        &mut self,
        leaf_idx: IDX,
        split_dim: usize,
        parent_idx: IDX,
        was_parents_left: bool,
    ) -> IDX {
        let orig = self.leaves.get_unchecked_mut(leaf_idx.az::<usize>());
        let mut pivot_idx: IDX = (B / 2).az::<IDX>();

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

        let mut split_val = *orig
            .content_points
            .get_unchecked(pivot_idx.az::<usize>())
            .get_unchecked(split_dim);

        // if the chosen pivot point would result in some items whose position on the split
        // dimension is the same as the split value being on the wrong side of the split:
        if *orig
            .content_points
            .get_unchecked(pivot_idx.az::<usize>() - 1)
            .get_unchecked(split_dim)
            == split_val
        {
            let orig_pivot_idx = pivot_idx;

            // Ensure that if pivot index would result in items that share the same co-ordinate
            // on the splitting dimension would end up on different sides of the split, that we
            // move the pivot to prevent this. We first try moving down, since that's the only
            // part of the bucket that was sorted by mirror_select_nth_unstable_by
            while pivot_idx > IDX::zero()
                && *orig
                    .content_points
                    .get_unchecked(pivot_idx.az::<usize>() - 1)
                    .get_unchecked(split_dim)
                    == split_val
            {
                pivot_idx = pivot_idx - IDX::one();
            }

            // If the attempt to move the pivot point above would have resulted in the pivot
            // point moving to the start of the bucket, search forwards from the original
            // pivot point instead. We need to first ensure the upper half of the bucket
            // is sorted
            if pivot_idx == IDX::zero() {
                mirror_select_nth_unstable_by(
                    &mut orig.content_points,
                    &mut orig.content_items,
                    B - 1,
                    |a, b| unsafe {
                        a.get_unchecked(split_dim)
                            .partial_cmp(b.get_unchecked(split_dim))
                            .expect("Leaf node sort failed.")
                    },
                );

                pivot_idx = orig_pivot_idx;
                while *orig
                    .content_points
                    .get_unchecked(pivot_idx.az::<usize>())
                    .get_unchecked(split_dim)
                    == split_val
                {
                    pivot_idx = pivot_idx + IDX::one();

                    if pivot_idx.az::<usize>() == B {
                        panic!("Too many items with the same position on one axis. Bucket size must be increased to at least 1 more than the number of items with the same position on one axis.");
                    }
                }
            }

            split_val = *orig
                .content_points
                .get_unchecked(pivot_idx.az::<usize>())
                .get_unchecked(split_dim);
        }

        let mut right = LeafNode::new();
        orig.size = pivot_idx;
        let dest_slice_end = B - pivot_idx.az::<usize>();

        right
            .content_points
            .get_unchecked_mut(..dest_slice_end)
            .copy_from_slice(
                orig.content_points
                    .get_unchecked((pivot_idx.az::<usize>())..),
            );
        right
            .content_items
            .get_unchecked_mut(..dest_slice_end)
            .copy_from_slice(
                orig.content_items
                    .get_unchecked((pivot_idx.az::<usize>())..),
            );

        right.size = (B.az::<IDX>()) - pivot_idx;

        self.leaves.push(right);

        self.stems.push(StemNode {
            left: leaf_idx + IDX::leaf_offset(),
            right: (self.leaves.len().az::<IDX>()) - IDX::one() + IDX::leaf_offset(),
            split_val,
        });
        let new_stem_index: IDX = (self.stems.len().az::<IDX>()) - IDX::one();

        if parent_idx != <IDX as Index>::max() {
            let parent_node = self.stems.get_unchecked_mut(parent_idx.az::<usize>());
            if was_parents_left {
                parent_node.left = new_stem_index;
            } else {
                parent_node.right = new_stem_index;
            }
        } else {
            self.root_index = new_stem_index;
        }

        new_stem_index
    }
}

#[cfg(test)]
mod tests {
    use fixed::types::extra::U16;
    use fixed::FixedU16;

    use crate::fixed::kdtree::KdTree;

    type FXD = FixedU16<U16>;

    fn n(num: f32) -> FXD {
        FXD::from_num(num)
    }

    #[test]
    fn can_add_an_item() {
        let mut tree: KdTree<FXD, u32, 4, 32, u32> = KdTree::new();

        let point: [FXD; 4] = [n(0.1f32), n(0.2f32), n(0.3f32), n(0.4f32)];
        let item = 123;

        tree.add(&point, item);

        assert_eq!(tree.size(), 1);
    }

    #[test]
    fn can_add_enough_items_to_cause_a_split() {
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
    }

    #[test]
    fn can_remove_an_item() {
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

        let removed = tree.remove(&[n(0.9f32), n(0.0f32), n(0.9f32), n(0.0f32)], 9);

        assert_eq!(removed, 1);
        assert_eq!(tree.size(), 15);
    }
}
