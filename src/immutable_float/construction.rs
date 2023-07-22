use crate::immutable_float::kdtree::{Axis, ImmutableKdTree};
use crate::types::Content;
use std::ops::Rem;

impl<A: Axis, T: Content, const K: usize, const B: usize>
ImmutableKdTree<A, T, K, B>
{
    #[inline]
    pub(crate) fn add_to_optimized(&mut self, query: &[A; K], item: T) {

        let mut dim = 0;
        let mut idx: usize = 1;
        let mut val: A;

        while idx < self.stems.len() {
            val = *unsafe { self.stems.get_unchecked(idx) };

            let is_right_child = *unsafe { query.get_unchecked(dim) } >= val;
            idx = (idx << 1) + usize::from(is_right_child);
            dim = (dim + 1).rem(K);
        }
        idx -= self.stems.len();

        let node_size = (unsafe { self.leaves.get_unchecked_mut(idx) }).size;
        if node_size == B {
            println!("Tree Stats: {:?}", self.generate_stats())
        }

        let node = unsafe { self.leaves.get_unchecked_mut(idx) };
        debug_assert!(node.size < B);

        *unsafe {
            node.content_points.get_unchecked_mut(node.size)
        } = *query;
        *unsafe {
            node.content_items.get_unchecked_mut(node.size)
        } = item;

        node.size += 1;
        self.size += 1;
    }
}

#[cfg(test)]
mod tests {
}
