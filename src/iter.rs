use std::collections::VecDeque;

use crate::types::Content;

pub(crate) type LeafData<A, T, const K: usize> = VecDeque<(T, [A; K])>;

pub(crate) trait IterableTreeData<A: Copy + Default, T: Content, const K: usize> {
    fn get_leaf_data(&self, idx: usize) -> Option<LeafData<A, T, K>>;
}

#[derive(Debug)]
pub(crate) struct TreeIter<
    'a,
    A: Copy + Default,
    T: Content,
    const K: usize,
    X: IterableTreeData<A, T, K>,
> {
    tree: &'a X,
    leaf_idx: usize,
    leaf_data: Option<LeafData<A, T, K>>,
}

impl<'a, A: Copy + Default, T: Content, const K: usize, X: IterableTreeData<A, T, K>>
    TreeIter<'a, A, T, K, X>
{
    pub(crate) fn new(tree: &'a X) -> Self {
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
        while self.leaf_data.as_ref()?.is_empty() {
            self.leaf_data = self.tree.get_leaf_data(self.leaf_idx);
            self.leaf_idx += 1;
        }
        self.leaf_data.as_mut()?.pop_front()
    }
}
