use std::collections::VecDeque;

use crate::fixed::kdtree as fixed;
use crate::float::kdtree as float;
use crate::immutable::float::kdtree as immut;
use crate::types::{Content, Index};

pub(crate) type LeafData<A: Copy + Default, T: Content, const K: usize> = VecDeque<(T, [A; K])>;

pub(crate) trait IterableTreeData<A: Copy + Default, T: Content, const K: usize> {
    fn get_leaf_data(&self, idx: usize) -> Option<LeafData<A, T, K>>;
}

#[derive(Debug)]
pub struct TreeIter<'a, A: Copy + Default, T: Content, const K: usize, X: IterableTreeData<A, T, K>>
{
    tree: &'a X,
    leaf_idx: usize,
    leaf_data: Option<LeafData<A, T, K>>,
}

impl<'a, A: Copy + Default, T: Content, const K: usize, X: IterableTreeData<A, T, K>>
    TreeIter<'a, A, T, K, X>
{
    pub(crate) fn new(tree: &'a X) -> Self {
        let leaf_data = tree.get_leaf_data(0);
        Self {
            tree,
            leaf_idx: 0,
            leaf_data: Some(LeafData::default()),
        }
    }
}

impl<'a, A: Copy + Default, T: Content, const K: usize, X: IterableTreeData<A, T, K>> Iterator
    for TreeIter<'a, A, T, K, X>
{
    type Item = (T, [A; K]);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let mut data = self.leaf_data?;
            if data.is_empty() {
                self.leaf_data = self.tree.get_leaf_data(self.leaf_idx);
                self.leaf_idx += 1;
            }
            return data.pop_front();
        }
    }
}
