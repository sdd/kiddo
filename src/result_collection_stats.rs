#![allow(missing_docs)]

use std::cell::Cell;

#[derive(Clone, Copy, Debug, Default)]
enum LeafPhase {
    #[default]
    Unknown,
    BeforeFull,
    AfterFull,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ResultCollectionStats {
    pub leaf_visits: u64,
    pub leaf_visits_before_full: u64,
    pub leaf_visits_after_full: u64,
    pub collection_full_transitions: u64,
    pub candidates_emitted: u64,
    pub candidates_emitted_before_full: u64,
    pub candidates_emitted_after_full: u64,
    pub best_item_threshold_rejects: u64,
    pub buffer_flushes: u64,
    pub buffer_flush_size_sum: u64,
    pub buffer_flush_size_max: u64,
    pub buffer_flush_size_0: u64,
    pub buffer_flush_size_1: u64,
    pub buffer_flush_size_2_4: u64,
    pub buffer_flush_size_5_8: u64,
    pub buffer_flush_size_9_plus: u64,
    pub collector_add_calls: u64,
    pub collector_add_all_calls: u64,
    pub collector_add_all_entry_count: u64,
    pub threshold_distance_calls: u64,
    pub threshold_distance_full: u64,
    pub threshold_distance_some: u64,
    pub sorted_insert_calls: u64,
    pub sorted_insert_position_sum: u64,
    pub sorted_shifted_items_sum: u64,
    pub heap_add_pushes: u64,
    pub heap_add_replacements: u64,
    pub heap_add_discards: u64,
    pub query_stack_pushes: u64,
    pub query_stack_pops: u64,
    pub query_prunes: u64,
    pub query_scalar_traverse_steps: u64,
    pub query_scalar_far_child_pushes: u64,
    pub query_scalar_far_child_rejects: u64,
    pub query_scalar_rd_off_checks: u64,
    pub query_scalar_rd_off_mismatch_lt: u64,
    pub query_scalar_rd_off_mismatch_gt: u64,
    pub query_scalar_continuation_frame_pushes: u64,
    pub query_scalar_continuation_frame_pops: u64,
    pub query_scalar_continuation_far_rechecks: u64,
    pub query_scalar_continuation_far_enters: u64,
    pub query_scalar_continuation_far_rejects_after_near: u64,
}

thread_local! {
    static STATS: Cell<ResultCollectionStats> = Cell::new(ResultCollectionStats::default());
    static LEAF_PHASE: Cell<LeafPhase> = Cell::new(LeafPhase::Unknown);
}

#[inline]
pub fn reset() {
    STATS.with(|stats| stats.set(ResultCollectionStats::default()));
    LEAF_PHASE.with(|phase| phase.set(LeafPhase::Unknown));
}

#[inline]
pub fn snapshot() -> ResultCollectionStats {
    STATS.with(Cell::get)
}

#[inline]
fn update(f: impl FnOnce(&mut ResultCollectionStats)) {
    STATS.with(|stats| {
        let mut value = stats.get();
        f(&mut value);
        stats.set(value);
    });
}

#[inline]
pub fn record_leaf_visit() {
    update(|stats| stats.leaf_visits += 1);
}

#[inline]
pub fn record_candidate_emitted() {
    let phase = LEAF_PHASE.with(Cell::get);
    update(|stats| {
        stats.candidates_emitted += 1;
        match phase {
            LeafPhase::BeforeFull => stats.candidates_emitted_before_full += 1,
            LeafPhase::AfterFull => stats.candidates_emitted_after_full += 1,
            LeafPhase::Unknown => {}
        }
    });
}

#[inline]
pub fn record_best_item_threshold_reject() {
    update(|stats| stats.best_item_threshold_rejects += 1);
}

#[inline]
pub fn record_leaf_visit_before_full() {
    LEAF_PHASE.with(|phase| phase.set(LeafPhase::BeforeFull));
    update(|stats| {
        stats.leaf_visits += 1;
        stats.leaf_visits_before_full += 1;
    });
}

#[inline]
pub fn record_leaf_visit_after_full() {
    LEAF_PHASE.with(|phase| phase.set(LeafPhase::AfterFull));
    update(|stats| {
        stats.leaf_visits += 1;
        stats.leaf_visits_after_full += 1;
    });
}

#[inline]
pub fn clear_leaf_phase() {
    LEAF_PHASE.with(|phase| phase.set(LeafPhase::Unknown));
}

#[inline]
pub fn record_collection_full_transition() {
    update(|stats| stats.collection_full_transitions += 1);
}

#[inline]
pub fn record_buffer_flush(size: usize) {
    update(|stats| {
        stats.buffer_flushes += 1;
        stats.buffer_flush_size_sum += size as u64;
        stats.buffer_flush_size_max = stats.buffer_flush_size_max.max(size as u64);
        match size {
            0 => stats.buffer_flush_size_0 += 1,
            1 => stats.buffer_flush_size_1 += 1,
            2..=4 => stats.buffer_flush_size_2_4 += 1,
            5..=8 => stats.buffer_flush_size_5_8 += 1,
            _ => stats.buffer_flush_size_9_plus += 1,
        }
    });
}

#[inline]
pub fn record_collector_add_call() {
    update(|stats| stats.collector_add_calls += 1);
}

#[inline]
pub fn record_collector_add_all_call(entry_count: usize) {
    update(|stats| {
        stats.collector_add_all_calls += 1;
        stats.collector_add_all_entry_count += entry_count as u64;
    });
}

#[inline]
pub fn record_threshold_distance_call(is_full: bool, has_threshold: bool) {
    update(|stats| {
        stats.threshold_distance_calls += 1;
        if is_full {
            stats.threshold_distance_full += 1;
        }
        if has_threshold {
            stats.threshold_distance_some += 1;
        }
    });
}

#[inline]
pub fn record_sorted_insert(position: usize, shifted_items: usize) {
    update(|stats| {
        stats.sorted_insert_calls += 1;
        stats.sorted_insert_position_sum += position as u64;
        stats.sorted_shifted_items_sum += shifted_items as u64;
    });
}

#[inline]
pub fn record_heap_add_push() {
    update(|stats| stats.heap_add_pushes += 1);
}

#[inline]
pub fn record_heap_add_replacement() {
    update(|stats| stats.heap_add_replacements += 1);
}

#[inline]
pub fn record_heap_add_discard() {
    update(|stats| stats.heap_add_discards += 1);
}

#[inline]
pub fn record_query_stack_push() {
    update(|stats| stats.query_stack_pushes += 1);
}

#[inline]
pub fn record_query_stack_pop() {
    update(|stats| stats.query_stack_pops += 1);
}

#[inline]
pub fn record_query_prune() {
    update(|stats| stats.query_prunes += 1);
}

#[inline]
pub fn record_query_scalar_traverse_step() {
    update(|stats| stats.query_scalar_traverse_steps += 1);
}

#[inline]
pub fn record_query_scalar_far_child_push() {
    update(|stats| {
        stats.query_scalar_far_child_pushes += 1;
        stats.query_stack_pushes += 1;
    });
}

#[inline]
pub fn record_query_scalar_far_child_candidate() {
    update(|stats| stats.query_scalar_far_child_pushes += 1);
}

#[inline]
pub fn record_query_scalar_far_child_reject() {
    update(|stats| stats.query_scalar_far_child_rejects += 1);
}

#[inline]
pub fn record_query_scalar_rd_off_check(ordering: std::cmp::Ordering) {
    update(|stats| {
        stats.query_scalar_rd_off_checks += 1;
        match ordering {
            std::cmp::Ordering::Less => stats.query_scalar_rd_off_mismatch_lt += 1,
            std::cmp::Ordering::Greater => stats.query_scalar_rd_off_mismatch_gt += 1,
            std::cmp::Ordering::Equal => {}
        }
    });
}

#[inline]
pub fn record_query_scalar_continuation_frame_push() {
    update(|stats| {
        stats.query_scalar_continuation_frame_pushes += 1;
        stats.query_stack_pushes += 1;
    });
}

#[inline]
pub fn record_query_scalar_continuation_frame_pop() {
    update(|stats| {
        stats.query_scalar_continuation_frame_pops += 1;
        stats.query_stack_pops += 1;
    });
}

#[inline]
pub fn record_query_scalar_continuation_far_recheck() {
    update(|stats| stats.query_scalar_continuation_far_rechecks += 1);
}

#[inline]
pub fn record_query_scalar_continuation_far_enter() {
    update(|stats| stats.query_scalar_continuation_far_enters += 1);
}

#[inline]
pub fn record_query_scalar_continuation_far_reject_after_near() {
    update(|stats| stats.query_scalar_continuation_far_rejects_after_near += 1);
}
