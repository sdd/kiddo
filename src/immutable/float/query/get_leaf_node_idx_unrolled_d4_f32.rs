use crate::immutable::float::kdtree::ArchivedR8ImmutableKdTree;
use crate::stem_strategies::Donnelly;
use crate::traits::StemStrategy;

impl ArchivedR8ImmutableKdTree<f32, usize, Donnelly<4, 64, 4, 4>, 4, 2> {
    pub fn get_leaf_node_idx_unrolled(&self, query: &[f32; 4]) -> usize {
        let stems_ptr = std::ptr::NonNull::new(self.stems.as_ptr() as *mut u8).unwrap();
        let mut stem_ordering = Donnelly::<4, 64, 4, 4>::new(stems_ptr);

        // Since we only care what the value of level is at the loop termination check,
        // and we're specifically constructing a tree with a level height that is a multiple of 4,
        // we can
        let max_stem_minor_tri_level = Into::<i32>::into(self.max_stem_level).wrapping_shl(2);

        while stem_ordering.level() <= max_stem_minor_tri_level {
            // Unroll #0
            let val = *unsafe { self.stems.get_unchecked(stem_ordering.stem_idx()) };
            let is_right_child = *unsafe { query.get_unchecked(stem_ordering.dim()) } >= val;
            stem_ordering.traverse_head(is_right_child);

            // Unroll #1
            let val = *unsafe { self.stems.get_unchecked(stem_ordering.stem_idx()) };
            let is_right_child = *unsafe { query.get_unchecked(stem_ordering.dim()) } >= val;
            stem_ordering.traverse_head(is_right_child);

            // Unroll #2
            let val = *unsafe { self.stems.get_unchecked(stem_ordering.stem_idx()) };
            let is_right_child = *unsafe { query.get_unchecked(stem_ordering.dim()) } >= val;
            stem_ordering.traverse_head(is_right_child);

            // Unroll #3 - tail unroll
            let val = *unsafe { self.stems.get_unchecked(stem_ordering.stem_idx()) };
            let is_right_child = *unsafe { query.get_unchecked(stem_ordering.dim()) } >= val;
            stem_ordering.traverse_tail(is_right_child);
        }

        stem_ordering.leaf_idx()
    }
}
