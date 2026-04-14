use std::cell::{Cell, RefCell};

#[derive(Clone, Debug)]
pub enum ExactQueryTraceEvent {
    ScalarStep {
        stem_idx: usize,
        level: i32,
        dim: usize,
        pivot: f64,
        query_elem: f64,
        is_right_child: bool,
        old_off: f64,
        new_off: f64,
        rd: f64,
        rd_far: f64,
        near_stem_idx: usize,
        far_stem_idx: usize,
    },
    Block3FullStep {
        stem_idx: usize,
        level: i32,
        dim: usize,
        query_val: f64,
        old_off: f64,
        parent_lower_bound: f64,
        parent_upper_bound: f64,
        rd: f64,
        best_dist: f64,
        child_idx: u8,
        candidate_mask: u8,
        new_off_values: [f64; 8],
        rd_values: [f64; 8],
        lower_bounds: [f64; 8],
        upper_bounds: [f64; 8],
    },
    Block3PendingSelection {
        stem_idx: usize,
        level: i32,
        dim: usize,
        pending_mask: u8,
        candidate_mask: u8,
        selected_child_idx: u8,
        child_off: f64,
        child_rd: f64,
        parent_lower_bound: f64,
        parent_upper_bound: f64,
        old_off: f64,
        rd: f64,
        new_off_values: [f64; 8],
        rd_values: [f64; 8],
        lower_bounds: [f64; 8],
        upper_bounds: [f64; 8],
    },
    LeafVisit {
        leaf_idx: usize,
    },
}

thread_local! {
    static ENABLED: Cell<bool> = const { Cell::new(false) };
    static EVENTS: RefCell<Vec<ExactQueryTraceEvent>> = const { RefCell::new(Vec::new()) };
}

#[inline]
pub fn set_enabled(enabled: bool) {
    ENABLED.with(|flag| flag.set(enabled));
    if enabled {
        clear();
    }
}

#[inline]
pub fn enabled() -> bool {
    ENABLED.with(Cell::get)
}

#[inline]
pub fn clear() {
    EVENTS.with(|events| events.borrow_mut().clear());
}

#[inline]
pub fn snapshot() -> Vec<ExactQueryTraceEvent> {
    EVENTS.with(|events| events.borrow().clone())
}

#[inline]
pub fn push(event: ExactQueryTraceEvent) {
    if !enabled() {
        return;
    }
    EVENTS.with(|events| events.borrow_mut().push(event));
}
