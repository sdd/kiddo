use crate::immutable::float::kdtree::ArchivedR8ImmutableKdTree;
use crate::stem_strategies::Donnelly;
use crate::traits::StemStrategy;

impl ArchivedR8ImmutableKdTree<f32, usize, Donnelly<4, 64, 4, 4>, 4, 2> {
    pub fn get_leaf_node_idx_unrolled(&self, query: &[f32; 4]) -> usize {
        let stems_ptr = std::ptr::NonNull::new(self.stems.as_ptr() as *mut u8).unwrap();
        let mut stem_ordering = Donnelly::<4, 64, 4, 4>::new(stems_ptr);

        while stem_ordering.level() <= Into::<i32>::into(self.max_stem_level) {
            // Unroll #0
            let stem_idx = stem_ordering.stem_idx();

            let val = *unsafe { self.stems.get_unchecked(stem_idx) };
            let is_right_child = *unsafe { query.get_unchecked(stem_ordering.dim()) } >= val;
            stem_ordering.traverse(is_right_child);

            // Unroll #1
            let stem_idx = stem_ordering.stem_idx();

            let val = *unsafe { self.stems.get_unchecked(stem_idx) };
            let is_right_child = *unsafe { query.get_unchecked(stem_ordering.dim()) } >= val;
            stem_ordering.traverse(is_right_child);

            // Unroll #2
            let stem_idx = stem_ordering.stem_idx();

            let val = *unsafe { self.stems.get_unchecked(stem_idx) };
            let is_right_child = *unsafe { query.get_unchecked(stem_ordering.dim()) } >= val;
            stem_ordering.traverse(is_right_child);

            // Unroll #3
            let stem_idx = stem_ordering.stem_idx();

            let val = *unsafe { self.stems.get_unchecked(stem_idx) };
            let is_right_child = *unsafe { query.get_unchecked(stem_ordering.dim()) } >= val;
            stem_ordering.traverse(is_right_child);
        }

        stem_ordering.leaf_idx()
    }
}
