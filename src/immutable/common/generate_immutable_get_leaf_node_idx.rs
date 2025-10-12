use std::sync::mpsc::Sender;

#[doc(hidden)]
#[macro_export]
macro_rules! generate_immutable_get_leaf_node_idx {
    ($comments:tt) => {
        doc_comment! {
            concat!$comments,
            #[cfg_attr(not(feature = "no_inline"), inline)]
            pub fn get_leaf_node_idx(&self, query: &[A; K], tracker: Option<&Sender<usize>>) -> usize
            where
                A: $crate::leaf_slice::float::LeafSliceFloat<T> + $crate::leaf_slice::float::LeafSliceFloatChunk<T, K>,
                usize: Cast<T>,
            {
                let stems_ptr = std::ptr::NonNull::new(self.stems.as_ptr() as *mut u8).unwrap();
                let mut stem_ordering = SO::new(stems_ptr);

                while stem_ordering.level() <= Into::<i32>::into(self.max_stem_level) {
                    let stem_idx = stem_ordering.stem_idx();
                    if let Some(tracker) = tracker {
                        let ptr = unsafe { stems_ptr.as_ptr().add((stem_idx as usize) * 4 as usize) as usize };
                        tracker.send(ptr);
                    }

                    let val = *unsafe { self.stems.get_unchecked(stem_idx) };
                    let is_right_child = *unsafe { query.get_unchecked(stem_ordering.dim()) } >= val;
                    stem_ordering.traverse(is_right_child);
                }

                stem_ordering.leaf_idx()
            }
        }
    };
}
