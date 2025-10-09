#[doc(hidden)]
#[macro_export]
macro_rules! generate_immutable_get_leaf_node_idx {
    ($comments:tt) => {
        doc_comment! {
            concat!$comments,
            #[cfg_attr(not(feature = "no_inline"), inline)]
            pub fn get_leaf_node_idx(&self, query: &[A; K]) -> usize
            where
                A: $crate::leaf_slice::float::LeafSliceFloat<T> + $crate::leaf_slice::float::LeafSliceFloatChunk<T, K>,
                usize: Cast<T>,
            {
                let stems_ptr = std::ptr::NonNull::new(self.stems.as_ptr() as *mut u8).unwrap();
                let mut stem_ordering = SO::new(stems_ptr);
                // let mut stem_addr_hist: Vec<usize> = vec![];

                // let ptr = unsafe { stems_ptr.as_ptr().add((stem_ordering.stem_idx() as usize) * 4 as usize) as usize };
                // stem_addr_hist.push(ptr);

                while stem_ordering.level() <= Into::<i32>::into(self.max_stem_level) {
                    let val = *unsafe { self.stems.get_unchecked(stem_ordering.stem_idx()) };
                    let is_right_child = *unsafe { query.get_unchecked(stem_ordering.dim()) } >= val;
                    stem_ordering.traverse(is_right_child);

                    // let ptr = unsafe { stems_ptr.as_ptr().add((stem_ordering.stem_idx() as usize) * 4 as usize) as usize };
                    // stem_addr_hist.push(ptr);
                }

                // println!("Stem path: {:?}", stem_addr_hist);
                stem_ordering.leaf_idx()
            }
        }
    };
}
