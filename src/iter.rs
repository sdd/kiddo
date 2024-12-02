use crate::traits::Content;

pub(crate) trait IterableTreeData<A: Copy + Default, T: Content, const K: usize> {
    fn get_leaf_data(&self, idx: usize, out: &mut Vec<(T, [A; K])>) -> Option<usize>;
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
    leaf_data: Vec<(T, [A; K])>,
}

impl<'a, A: Copy + Default, T: Content, const K: usize, X: IterableTreeData<A, T, K>>
    TreeIter<'a, A, T, K, X>
{
    pub(crate) fn new(tree: &'a X, bucket_size: usize) -> Self {
        Self {
            tree,
            leaf_idx: 0,
            leaf_data: Vec::with_capacity(bucket_size),
        }
    }
}

impl<A: Copy + Default, T: Content, const K: usize, X: IterableTreeData<A, T, K>> Iterator
    for TreeIter<'_, A, T, K, X>
{
    type Item = (T, [A; K]);

    fn next(&mut self) -> Option<Self::Item> {
        while self.leaf_data.is_empty() {
            self.tree
                .get_leaf_data(self.leaf_idx, &mut self.leaf_data)?;
            self.leaf_idx += 1;
        }
        self.leaf_data.pop()
    }
}
