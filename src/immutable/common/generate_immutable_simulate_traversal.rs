#[doc(hidden)]
#[macro_export]
macro_rules! generate_immutable_simulate_traversal {
    ($comments:tt) => {
        doc_comment! {
            concat!$comments,
            #[cfg_attr(not(feature = "no_inline"), inline)]
            pub fn simulate_traversal(&self, query: &[A; K], event_tx: &Sender<$crate::cache_simulator::Event>) -> usize
            where
                A: $crate::leaf_slice::float::LeafSliceFloat<T> + $crate::leaf_slice::float::LeafSliceFloatChunk<T, K>,
                usize: Cast<T>,
            {
                let _ = event_tx.send($crate::cache_simulator::Event::NewQuery);

                let stems_ptr = std::ptr::NonNull::new(self.stems.as_ptr() as *mut u8).unwrap();
                let mut stem_ordering = SO::new(stems_ptr);

                while stem_ordering.level() <= Into::<i32>::into(self.max_stem_level) {
                    let stem_idx = stem_ordering.stem_idx();

                    let offset: usize = (stem_idx as usize) * 4usize;
                    let _ = event_tx.send($crate::cache_simulator::Event::Access(offset));

                    let val = *unsafe { self.stems.get_unchecked(stem_idx) };
                    let is_right_child = *unsafe { query.get_unchecked(stem_ordering.dim()) } >= val;
                    stem_ordering.simulate_traverse(is_right_child, event_tx);
                }

                stem_ordering.leaf_idx()
            }
        }
    };
}
