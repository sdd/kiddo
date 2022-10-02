use crate::tuned::f32::d4::kdtree::{
    KdTree, LeafNode, StemNode, B, IDX, K, LEAF_OFFSET, PT, T,
};
use crate::tuned::f32::d4::util::mirror_select_nth_unstable_by;
use std::ops::Rem;

impl KdTree {
    #[inline]
    pub fn add(&mut self, query: &PT, item: T) {
        unsafe {
            debug_assert!(query.as_ptr() as usize % 16 == 0);

            let mut stem_idx = self.root_index;
            let mut split_dim = 0;
            let mut stem_node;
            let mut parent_idx: IDX = IDX::MAX;
            let mut was_parents_left: bool = false;

            while KdTree::is_stem_index(stem_idx) {
                parent_idx = stem_idx;
                stem_node = self.stems.get_unchecked_mut(stem_idx);

                debug_assert!(stem_node.min_bound.as_ptr() as usize % 16 == 0);
                debug_assert!(stem_node.max_bound.as_ptr() as usize % 16 == 0);

                stem_node.extend(query);

                stem_idx = if *query.get_unchecked(split_dim) < stem_node.split_val {
                    was_parents_left = true;
                    stem_node.left
                } else {
                    was_parents_left = false;
                    stem_node.right
                };

                split_dim = (split_dim + 1).rem(K);
            }

            let mut leaf_idx = stem_idx - LEAF_OFFSET;
            let mut leaf_node = self.leaves.get_unchecked_mut(leaf_idx);

            if leaf_node.size == B {
                stem_idx = self.split(leaf_idx, split_dim, parent_idx, was_parents_left);
                let node = self.stems.get_unchecked_mut(stem_idx);

                leaf_idx = (if query[split_dim] < node.split_val {
                    node.left
                } else {
                    node.right
                } - LEAF_OFFSET);

                leaf_node = self.leaves.get_unchecked_mut(leaf_idx);
            }

            leaf_node.content_points[leaf_node.size] = *query;
            leaf_node.content_items[leaf_node.size] = item;

            leaf_node.size += 1;
            leaf_node.extend(query);
        }

        self.size += 1;
    }

    #[inline]
    pub fn add_recursive(&mut self, query: &PT, item: T) {
        self.add_recurse_stem(query, item, self.root_index, 0, usize::MAX, false);
    }

    fn add_recurse_stem(
        &mut self,
        query: &PT,
        item: T,
        stem_idx: usize,
        split_dim: usize,
        parent_idx: usize,
        was_parents_left: bool,
    ) -> (PT, PT) {
        let next_split_dim = (split_dim + 1).rem(K);

        if KdTree::is_stem_index(stem_idx) {
            unsafe {
                let mut was_parents_left: bool = false;
                let next_stem_idx = if *query.get_unchecked(split_dim)
                    < self.stems.get_unchecked(stem_idx).split_val
                {
                    was_parents_left = true;
                    self.stems.get_unchecked(stem_idx).left
                } else {
                    self.stems.get_unchecked(stem_idx).right
                };

                let extend_result = self.add_recurse_stem(
                    query,
                    item,
                    next_stem_idx,
                    next_split_dim,
                    stem_idx,
                    was_parents_left,
                );
                self.stems
                    .get_unchecked_mut(stem_idx)
                    .extend(&extend_result.0);
                self.stems
                    .get_unchecked_mut(stem_idx)
                    .extend(&extend_result.1);

                (
                    self.stems.get_unchecked(stem_idx).min_bound,
                    self.stems.get_unchecked(stem_idx).max_bound,
                )
            }
        } else {
            self.add_recurse_leaf(
                query,
                item,
                stem_idx - LEAF_OFFSET,
                split_dim,
                was_parents_left,
                parent_idx,
            )
        }
    }

    fn add_recurse_leaf(
        &mut self,
        query: &PT,
        item: T,
        mut leaf_idx: usize,
        split_dim: usize,
        was_parents_left: bool,
        parent_idx: usize,
    ) -> (PT, PT) {
        let mut leaf_node = &mut self.leaves[leaf_idx];

        if leaf_node.size == B {
            let stem_idx = self.split(leaf_idx, split_dim, parent_idx, was_parents_left);
            let node = &self.stems[stem_idx];

            leaf_idx = (if query[split_dim] < node.split_val {
                node.left
            } else {
                node.right
            } - LEAF_OFFSET);

            leaf_node = &mut self.leaves[leaf_idx];
        }

        leaf_node.content_points[leaf_node.size] = *query;
        leaf_node.content_items[leaf_node.size] = item;
        leaf_node.size += 1;
        self.size += 1;

        leaf_node.extend_with_result(query)
    }

    //#[inline(never)]
    fn split(
        &mut self,
        leaf_idx: usize,
        split_dim: usize,
        parent_idx: usize,
        was_parents_left: bool,
    ) -> usize {
        let orig = &mut self.leaves[leaf_idx];
        let orig_min_bound = orig.min_bound;
        let orig_max_bound = orig.max_bound;
        let pivot_idx = B.div_floor(2);

        mirror_select_nth_unstable_by(
            &mut orig.content_points,
            &mut orig.content_items,
            pivot_idx,
            |a, b| {
                a[split_dim]
                    .partial_cmp(&b[split_dim])
                    .expect("Leaf node sort failed. Have you put a NaN in here?")
            },
        );

        let split_val = orig.content_points[pivot_idx][split_dim];

        let mut left = LeafNode::new();
        let mut right = LeafNode::new();
        left.min_bound = orig_min_bound;
        left.max_bound = orig_max_bound;
        right.min_bound = orig_min_bound;
        right.max_bound = orig_max_bound;
        left.max_bound[split_dim] = orig.content_points[pivot_idx - 1][split_dim];
        right.min_bound[split_dim] = orig.content_points[pivot_idx][split_dim];

        if B.rem(2) == 1 {
            left.content_points[..pivot_idx].copy_from_slice(&orig.content_points[..pivot_idx]);
            left.content_items[..pivot_idx].copy_from_slice(&orig.content_items[..pivot_idx]);
            left.size = pivot_idx;

            right.content_points[..pivot_idx + 1]
                .copy_from_slice(&orig.content_points[pivot_idx..]);
            right.content_items[..pivot_idx + 1].copy_from_slice(&orig.content_items[pivot_idx..]);
            right.size = B - pivot_idx;
        } else {
            left.content_points[..pivot_idx].copy_from_slice(&orig.content_points[..pivot_idx]);
            left.content_items[..pivot_idx].copy_from_slice(&orig.content_items[..pivot_idx]);
            left.size = pivot_idx;

            right.content_points[..pivot_idx].copy_from_slice(&orig.content_points[pivot_idx..]);
            right.content_items[..pivot_idx].copy_from_slice(&orig.content_items[pivot_idx..]);
            right.size = B - pivot_idx;
        }

        *orig = left;
        self.leaves.push(right);

        self.stems.push(StemNode {
            left: leaf_idx + LEAF_OFFSET,
            right: self.leaves.len() - 1 + LEAF_OFFSET,
            split_val,
            min_bound: orig_min_bound,
            max_bound: orig_max_bound,
        });
        let new_stem_index = self.stems.len() - 1;

        if parent_idx != usize::MAX {
            let parent_node = &mut self.stems[parent_idx];
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
    use crate::tuned::f32::d4::kdtree::{KdTree, PT, T};
    use aligned_array::Aligned;
    use aligned_array::A16;

    #[test]
    fn can_add_an_item() {
        let mut tree: KdTree = KdTree::new();
        let mut tree2: KdTree = KdTree::new();

        let point: Aligned<A16, PT> = Aligned([1f32, 2f32, 3f32, 4f32]);
        debug_assert!(point.as_ptr() as usize % 16 == 0);
        let item = 123;

        tree.add(&point, item);
        tree2.add_recursive(&point, item);

        assert_eq!(tree.size(), 1);
        assert_eq!(tree2.size(), 1);
    }

    #[test]
    fn can_add_enough_items_to_cause_a_split() {
        let mut tree: KdTree = KdTree::new();
        let mut tree2: KdTree = KdTree::new();

        let content_to_add: [(PT, T); 16] = [
            ([9f32, 0f32, 9f32, 0f32], 9),
            ([4f32, 500f32, 4f32, 500f32], 4),
            ([12f32, -300f32, 12f32, -300f32], 12),
            ([7f32, 200f32, 7f32, 200f32], 7),
            ([13f32, -400f32, 13f32, -400f32], 13),
            ([6f32, 300f32, 6f32, 300f32], 6),
            ([2f32, 700f32, 2f32, 700f32], 2),
            ([14f32, -500f32, 14f32, -500f32], 14),
            ([3f32, 600f32, 3f32, 600f32], 3),
            ([10f32, -100f32, 10f32, -100f32], 10),
            ([16f32, -700f32, 16f32, -700f32], 16),
            ([1f32, 800f32, 1f32, 800f32], 1),
            ([15f32, -600f32, 15f32, -600f32], 15),
            ([5f32, 400f32, 5f32, 400f32], 5),
            ([8f32, 100f32, 8f32, 100f32], 8),
            ([11f32, -200f32, 11f32, -200f32], 11),
        ];

        // ensure we have 128bit aligned points
        debug_assert!(content_to_add[0].0.as_ptr() as usize % 16 == 0);
        //debug_assert!(content_to_add[1].0.as_ptr() as usize % 16 == 0);

        for (point, item) in content_to_add {
            tree.add(&point, item);
            tree2.add_recursive(&point, item);
        }

        assert_eq!(tree.size(), 16);
        assert_eq!(tree2.size(), 16);
    }
}
