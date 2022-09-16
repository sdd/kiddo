use crate::sok::{Axis, Content, LeafNode, LeafNodeEntry, StemNode, LEAF_OFFSET};
use crate::KdTree;
use std::ops::Rem;

impl<A: Axis, T: Content, const K: usize, const B: usize> KdTree<A, T, K, B> {
    #[inline]
    pub fn add(&mut self, query: &[A; K], item: T) {
        let mut stem_idx = self.root_index;
        let mut split_dim = 0;
        let mut stem_node;
        let mut parent_idx = usize::MAX;
        let mut was_parents_left: bool = false;

        while KdTree::<A, T, K, B>::is_stem_index(stem_idx) {
            parent_idx = stem_idx;
            stem_node = &mut self.stems[stem_idx];
            stem_node.extend(query);

            stem_idx = if query[split_dim] < stem_node.split_val {
                was_parents_left = true;
                stem_node.left
            } else {
                was_parents_left = false;
                stem_node.right
            };

            split_dim = (split_dim + 1).rem(K);
        }

        let mut leaf_idx = stem_idx - LEAF_OFFSET;
        let mut leaf_node = &mut self.leaves[leaf_idx];

        if leaf_node.size == B {
            stem_idx = self.split(leaf_idx, split_dim, parent_idx, was_parents_left);
            let node = &self.stems[stem_idx];

            leaf_idx = (if query[split_dim] < node.split_val {
                node.left
            } else {
                node.right
            } - LEAF_OFFSET);

            leaf_node = &mut self.leaves[leaf_idx];
        }

        leaf_node.content[leaf_node.size] = LeafNodeEntry::new(*query, item);
        leaf_node.size += 1;
        leaf_node.extend(query);

        self.size += 1;
    }

    #[inline]
    pub fn add_recursive(&mut self, query: &[A; K], item: T) {
        self.add_recurse_stem(query, item, self.root_index, 0, usize::MAX, false);
    }

    fn add_recurse_stem(
        &mut self,
        query: &[A; K],
        item: T,
        stem_idx: usize,
        split_dim: usize,
        parent_idx: usize,
        was_parents_left: bool,
    ) -> ([A; K], [A; K]) {
        let next_split_dim = (split_dim + 1).rem(K);

        if KdTree::<A, T, K, B>::is_stem_index(stem_idx) {
            let mut was_parents_left: bool = false;
            let next_stem_idx = if query[split_dim] < self.stems[stem_idx].split_val {
                was_parents_left = true;
                self.stems[stem_idx].left
            } else {
                self.stems[stem_idx].right
            };

            let extend_result = self.add_recurse_stem(
                query,
                item,
                next_stem_idx,
                next_split_dim,
                stem_idx,
                was_parents_left,
            );
            self.stems[stem_idx].extend(&extend_result.0);
            self.stems[stem_idx].extend(&extend_result.1);

            (
                self.stems[stem_idx].min_bound,
                self.stems[stem_idx].max_bound,
            )
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
        query: &[A; K],
        item: T,
        mut leaf_idx: usize,
        split_dim: usize,
        was_parents_left: bool,
        parent_idx: usize,
    ) -> ([A; K], [A; K]) {
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

        leaf_node.content[leaf_node.size] = LeafNodeEntry::new(*query, item);
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

        orig.content.select_nth_unstable_by(pivot_idx, |a, b| {
            a.point[split_dim]
                .partial_cmp(&b.point[split_dim])
                .expect("Leaf node sort failed. Have you put a NaN in here?")
        });

        let split_val = orig.content[pivot_idx].point[split_dim];

        let mut left = LeafNode::<A, T, K, B>::new();
        let mut right = LeafNode::<A, T, K, B>::new();
        left.min_bound = orig_min_bound;
        left.max_bound = orig_max_bound;
        right.min_bound = orig_min_bound;
        right.max_bound = orig_max_bound;
        left.max_bound[split_dim] = orig.content[pivot_idx - 1].point[split_dim];
        right.min_bound[split_dim] = orig.content[pivot_idx].point[split_dim];

        if B.rem(2) == 1 {
            left.content[..pivot_idx].copy_from_slice(&orig.content[..pivot_idx]);
            left.size = pivot_idx;

            right.content[..pivot_idx + 1].copy_from_slice(&orig.content[pivot_idx..]);
            right.size = B - pivot_idx;
        } else {
            left.content[..pivot_idx].copy_from_slice(&orig.content[..pivot_idx]);
            left.size = pivot_idx;

            right.content[..pivot_idx].copy_from_slice(&orig.content[pivot_idx..]);
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
    use crate::KdTree;

    #[test]
    fn can_add_an_item() {
        let mut tree: KdTree<f64, i32, 2, 10> = KdTree::new();
        let mut tree2: KdTree<f64, i32, 2, 10> = KdTree::new();

        let point = [1f64, 2f64];
        let item = 123;

        tree.add(&point, item);
        tree2.add_recursive(&point, item);

        assert_eq!(tree.size(), 1);
        assert_eq!(tree2.size(), 1);
    }

    #[test]
    fn can_add_enough_items_to_cause_a_split() {
        let mut tree: KdTree<f64, i32, 2, 4> = KdTree::new();
        let mut tree2: KdTree<f64, i32, 2, 4> = KdTree::new();

        let content_to_add = [
            ([9f64, 0f64], 9),
            ([4f64, 500f64], 4),
            ([12f64, -300f64], 12),
            ([7f64, 200f64], 7),
            ([13f64, -400f64], 13),
            ([6f64, 300f64], 6),
            ([2f64, 700f64], 2),
            ([14f64, -500f64], 14),
            ([3f64, 600f64], 3),
            ([10f64, -100f64], 10),
            ([16f64, -700f64], 16),
            ([1f64, 800f64], 1),
            ([15f64, -600f64], 15),
            ([5f64, 400f64], 5),
            ([8f64, 100f64], 8),
            ([11f64, -200f64], 11),
        ];

        for (point, item) in content_to_add {
            tree.add(&point, item);
            tree2.add_recursive(&point, item);
        }

        assert_eq!(tree.size(), 16);
        assert_eq!(tree2.size(), 16);
    }
}
