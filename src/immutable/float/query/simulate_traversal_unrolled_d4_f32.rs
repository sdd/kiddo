use std::sync::mpsc::Sender;

use crate::cache_simulator::{Event, PrefetchLevel};
use crate::immutable::float::kdtree::ImmutableKdTree;
use crate::stem_strategies::Donnelly;
use crate::traits::StemStrategy;

impl ImmutableKdTree<f32, u32, Donnelly<4, 64, 4, 4>, 4, 32> {
    pub fn simulate_traversal_unrolled(&self, query: &[f32; 4], event_tx: &Sender<Event>) -> usize {
        let _ = event_tx.send(Event::NewQuery);

        let stems_ptr = std::ptr::NonNull::new(self.stems.as_ptr() as *mut u8).unwrap();
        let mut stem_ordering = Donnelly::<4, 64, 4, 4>::new(stems_ptr);

        while stem_ordering.level() <= Into::<i32>::into(self.max_stem_level) {
            // Unrolled #0
            let stem_idx = stem_ordering.stem_idx();

            let offset: usize = stem_idx * 4usize;
            let _ = event_tx.send(Event::Access(offset));

            let val = *unsafe { self.stems.get_unchecked(stem_idx) };
            let is_right_child = *unsafe { query.get_unchecked(stem_ordering.dim()) } >= val;
            stem_ordering.simulate_traverse(is_right_child, event_tx);

            // Unrolled #1
            let stem_idx = stem_ordering.stem_idx();

            let offset: usize = stem_idx * 4usize;
            let _ = event_tx.send(Event::Access(offset));

            let val = *unsafe { self.stems.get_unchecked(stem_idx) };
            let is_right_child = *unsafe { query.get_unchecked(stem_ordering.dim()) } >= val;
            stem_ordering.simulate_traverse(is_right_child, event_tx);

            // Unrolled #2
            let stem_idx = stem_ordering.stem_idx();

            let offset: usize = stem_idx * 4usize;
            let _ = event_tx.send(Event::Access(offset));

            let val = *unsafe { self.stems.get_unchecked(stem_idx) };
            let is_right_child = *unsafe { query.get_unchecked(stem_ordering.dim()) } >= val;
            stem_ordering.simulate_traverse(is_right_child, event_tx);

            // Unrolled #3
            let stem_idx = stem_ordering.stem_idx();

            let offset: usize = stem_idx * 4usize;
            let _ = event_tx.send(Event::Access(offset));

            let val = *unsafe { self.stems.get_unchecked(stem_idx) };
            let is_right_child = *unsafe { query.get_unchecked(stem_ordering.dim()) } >= val;

            stem_ordering.traverse(is_right_child);

            // stem_ordering.simulate_traverse(is_right_child, event_tx);

            let _ = event_tx.send(Event::Working(2));
            let _ = event_tx.send(Event::PrefetchTo(
                stem_ordering.stem_idx() * 4usize,
                PrefetchLevel::L2,
            ));
            let _ = event_tx.send(Event::Working(3));
        }

        stem_ordering.leaf_idx()
    }
}
