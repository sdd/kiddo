use crate::tuned::u16::d4::kdtree::{
    KdTree, LeafNode, StemNode, B, IDX, K, LEAF_OFFSET, PT, T,
};
use crate::tuned::u16::d4::util::mirror_select_nth_unstable_by;
use std::ops::Rem;

impl KdTree {
    #[inline]
    pub fn add(&mut self, query: &PT, item: T) {
        unsafe {
            let mut stem_idx = self.root_index;
            let mut split_dim = 0;
            let mut stem_node;
            let mut parent_idx = IDX::MAX;
            let mut is_left_child: bool = false;

            while KdTree::is_stem_index(stem_idx) {
                parent_idx = stem_idx;
                stem_node = self.stems.get_unchecked_mut(stem_idx as usize);

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

            let mut leaf_idx = stem_idx - LEAF_OFFSET;
            let mut leaf_node = self.leaves.get_unchecked_mut(leaf_idx as usize);

            if leaf_node.size == B as IDX {
                stem_idx = self.split(leaf_idx, split_dim, parent_idx, is_left_child);
                let node = self.stems.get_unchecked_mut(stem_idx as usize);

                leaf_idx = (if *query.get_unchecked(split_dim) < node.split_val {
                    node.left
                } else {
                    node.right
                } - LEAF_OFFSET);

                leaf_node = self.leaves.get_unchecked_mut(leaf_idx as usize);
            }

            *leaf_node.content_points.get_unchecked_mut(leaf_node.size as usize) = *query;
            *leaf_node.content_items.get_unchecked_mut(leaf_node.size as usize) = item;

            leaf_node.size += 1;
            leaf_node.extend(query);
        }
        self.size += 1;
    }

    unsafe fn split(
        &mut self,
        leaf_idx: IDX,
        split_dim: usize,
        parent_idx: IDX,
        was_parents_left: bool,
    ) -> IDX {
        let orig = self.leaves.get_unchecked_mut(leaf_idx as usize);
        let orig_min_bound = orig.min_bound;
        let orig_max_bound = orig.max_bound;
        let pivot_idx: IDX = B.div_floor(2) as IDX;

        mirror_select_nth_unstable_by(
            &mut orig.content_points,
            &mut orig.content_items,
            pivot_idx as usize,
            |a, b| {
                unsafe { a.get_unchecked(split_dim)
                    .partial_cmp(b.get_unchecked(split_dim))
                    .expect("Leaf node sort failed.") }
            },
        );

        let split_val = *orig.content_points.get_unchecked(pivot_idx as usize).get_unchecked(split_dim);

        let mut left = LeafNode::new();
        let mut right = LeafNode::new();
        left.min_bound = orig_min_bound;
        left.max_bound = orig_max_bound;
        right.min_bound = orig_min_bound;
        right.max_bound = orig_max_bound;

        *left.max_bound.get_unchecked_mut(split_dim) = *orig.content_points.get_unchecked((pivot_idx - 1) as usize).get_unchecked(split_dim);
        *right.min_bound.get_unchecked_mut(split_dim) = *orig.content_points.get_unchecked(pivot_idx as usize).get_unchecked(split_dim);

        if B.rem(2) == 1 {
            left.content_points.get_unchecked_mut(..(pivot_idx as usize))
                .copy_from_slice(&orig.content_points.get_unchecked(..(pivot_idx as usize)));
            left.content_items.get_unchecked_mut(..(pivot_idx as usize))
                .copy_from_slice(&orig.content_items.get_unchecked(..(pivot_idx as usize)));
            left.size = pivot_idx;

            right.content_points.get_unchecked_mut(..((pivot_idx + 1) as usize))
                .copy_from_slice(&orig.content_points.get_unchecked((pivot_idx as usize)..));
            right.content_items.get_unchecked_mut(..((pivot_idx + 1) as usize))
                .copy_from_slice(&orig.content_items.get_unchecked((pivot_idx as usize)..));

            right.size = (B as IDX) - pivot_idx;
        } else {
            left.content_points.get_unchecked_mut(..(pivot_idx as usize))
                .copy_from_slice(&orig.content_points.get_unchecked(..(pivot_idx as usize)));
            left.content_items.get_unchecked_mut(..(pivot_idx as usize))
                .copy_from_slice(&orig.content_items.get_unchecked(..(pivot_idx as usize)));
            left.size = pivot_idx;

            right.content_points.get_unchecked_mut(..(pivot_idx as usize))
                .copy_from_slice(&orig.content_points.get_unchecked((pivot_idx as usize)..));
            right.content_items.get_unchecked_mut(..(pivot_idx as usize))
                .copy_from_slice(&orig.content_items.get_unchecked((pivot_idx as usize)..));

            right.size = (B as IDX) - pivot_idx;
        }

        *orig = left;
        self.leaves.push(right);

        self.stems.push(StemNode {
            left: leaf_idx + LEAF_OFFSET,
            right: (self.leaves.len() as IDX) - 1 + LEAF_OFFSET,
            split_val,
            min_bound: orig_min_bound,
            max_bound: orig_max_bound,
        });
        let new_stem_index: IDX = (self.stems.len() as IDX) - 1;

        if parent_idx != IDX::MAX {
            let parent_node = self.stems.get_unchecked_mut(parent_idx as usize);
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
    use crate::tuned::u16::d4::kdtree::{KdTree, PT, T};
    use fixed::types::extra::U16;
    use fixed::FixedU16;

    fn n(num: f32) -> FixedU16<U16> {
        FixedU16::<U16>::from_num(num)
    }

    #[test]
    fn can_add_an_item() {
        let mut tree: KdTree = KdTree::new();

        let point: PT = [
            n(0.1f32), n(0.2f32), n(0.3f32), n(0.4f32)
        ];
        let item = 123;

        tree.add(&point, item);

        assert_eq!(tree.size(), 1);
    }

    #[test]
    fn can_add_enough_items_to_cause_a_split() {
        let mut tree: KdTree = KdTree::new();

        let content_to_add: [(PT, T); 16] = [
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
