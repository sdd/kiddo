use az::{Az, Cast};
use crate::tuned::u16::dn::util::mirror_select_nth_unstable_by;
use std::ops::Rem;
use crate::tuned::u16::dn::kdtree::{Axis, Content, Index, KdTree, LeafNode, StemNode};

impl<A: Axis, T: Content, const K: usize, const B: usize, IDX: Index<T = IDX>> KdTree<A, T, K, B, IDX> where usize: Cast<IDX> {
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

                stem_node.extend(query);

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

            *leaf_node.content_points.get_unchecked_mut(leaf_node.size.az::<usize>()) = *query;
            *leaf_node.content_items.get_unchecked_mut(leaf_node.size.az::<usize>()) = item;

            leaf_node.size = leaf_node.size + IDX::one();
            leaf_node.extend(query);
        }
        self.size = self.size + T::one();
    }

    unsafe fn split(
        &mut self,
        leaf_idx: IDX,
        split_dim: usize,
        parent_idx: IDX,
        was_parents_left: bool,
    ) -> IDX {
        let orig = self.leaves.get_unchecked_mut(leaf_idx.az::<usize>());
        let orig_min_bound = orig.min_bound;
        let orig_max_bound = orig.max_bound;
        let pivot_idx: IDX = B.div_floor(2).az::<IDX>();

        mirror_select_nth_unstable_by(
            &mut orig.content_points,
            &mut orig.content_items,
            pivot_idx.az::<usize>(),
            |a, b| {
                unsafe { a.get_unchecked(split_dim)
                    .partial_cmp(b.get_unchecked(split_dim))
                    .expect("Leaf node sort failed.") }
            },
        );

        let split_val = *orig.content_points.get_unchecked(pivot_idx.az::<usize>()).get_unchecked(split_dim);

        let mut left = LeafNode::new();
        let mut right = LeafNode::new();
        left.min_bound = orig_min_bound;
        left.max_bound = orig_max_bound;
        right.min_bound = orig_min_bound;
        right.max_bound = orig_max_bound;

        *left.max_bound.get_unchecked_mut(split_dim) = *orig.content_points.get_unchecked((pivot_idx - IDX::one()).az::<usize>()).get_unchecked(split_dim);
        *right.min_bound.get_unchecked_mut(split_dim) = *orig.content_points.get_unchecked(pivot_idx.az::<usize>()).get_unchecked(split_dim);

        if B.rem(2) == 1 {
            left.content_points.get_unchecked_mut(..(pivot_idx.az::<usize>()))
                .copy_from_slice(&orig.content_points.get_unchecked(..(pivot_idx.az::<usize>())));
            left.content_items.get_unchecked_mut(..(pivot_idx.az::<usize>()))
                .copy_from_slice(&orig.content_items.get_unchecked(..(pivot_idx.az::<usize>())));
            left.size = pivot_idx;

            right.content_points.get_unchecked_mut(..((pivot_idx + IDX::one()).az::<usize>()))
                .copy_from_slice(&orig.content_points.get_unchecked((pivot_idx.az::<usize>())..));
            right.content_items.get_unchecked_mut(..((pivot_idx + IDX::one()).az::<usize>()))
                .copy_from_slice(&orig.content_items.get_unchecked((pivot_idx.az::<usize>())..));

            right.size = (B.az::<IDX>()) - pivot_idx;
        } else {
            left.content_points.get_unchecked_mut(..(pivot_idx.az::<usize>()))
                .copy_from_slice(&orig.content_points.get_unchecked(..(pivot_idx.az::<usize>())));
            left.content_items.get_unchecked_mut(..(pivot_idx.az::<usize>()))
                .copy_from_slice(&orig.content_items.get_unchecked(..(pivot_idx.az::<usize>())));
            left.size = pivot_idx;

            right.content_points.get_unchecked_mut(..(pivot_idx.az::<usize>()))
                .copy_from_slice(&orig.content_points.get_unchecked((pivot_idx.az::<usize>())..));
            right.content_items.get_unchecked_mut(..(pivot_idx.az::<usize>()))
                .copy_from_slice(&orig.content_items.get_unchecked((pivot_idx.az::<usize>())..));

            right.size = (B.az::<IDX>()) - pivot_idx;
        }

        *orig = left;
        self.leaves.push(right);

        self.stems.push(StemNode {
            left: leaf_idx + IDX::leaf_offset(),
            right: (self.leaves.len().az::<IDX>()) - IDX::one() + IDX::leaf_offset(),
            split_val,
            min_bound: orig_min_bound,
            max_bound: orig_max_bound,
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

    use crate::tuned::u16::dn::kdtree::KdTree;

    type FXD = FixedU16<U16>;

    fn n(num: f32) -> FXD {
        FXD::from_num(num)
    }

    #[test]
    fn can_add_an_item() {
        let mut tree: KdTree<FXD, u32, 4, 32, u32> = KdTree::new();

        let point: [FXD; 4] = [
            n(0.1f32), n(0.2f32), n(0.3f32), n(0.4f32)
        ];
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
}