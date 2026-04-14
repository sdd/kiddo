use std::cell::Cell;

#[derive(Clone, Copy, Debug, Default)]
pub struct ExactQueryStats {
    pub leaf_visits: u64,
    pub scalar_stack_pops: u64,
    pub simd_single_pops: u64,
    pub simd_stack_max_len: u64,
    pub block3_pending_pops: u64,
    pub block3_pending_mask_bits: u64,
    pub block3_candidate_mask_bits: u64,
    pub block3_candidate_mask_nonzero: u64,
    pub block3_step_entries: u64,
    pub block3_full_steps: u64,
    pub block3_scalar_fallback_steps: u64,
}

thread_local! {
    static STATS: Cell<ExactQueryStats> = Cell::new(ExactQueryStats::default());
}

#[inline]
pub fn reset() {
    STATS.with(|stats| stats.set(ExactQueryStats::default()));
}

#[inline]
pub fn snapshot() -> ExactQueryStats {
    STATS.with(Cell::get)
}

#[inline]
pub fn record_leaf_visit() {
    STATS.with(|stats| {
        let mut value = stats.get();
        value.leaf_visits += 1;
        stats.set(value);
    });
}

#[inline]
pub fn record_scalar_stack_pop() {
    STATS.with(|stats| {
        let mut value = stats.get();
        value.scalar_stack_pops += 1;
        stats.set(value);
    });
}

#[inline]
pub fn record_simd_single_pop() {
    STATS.with(|stats| {
        let mut value = stats.get();
        value.simd_single_pops += 1;
        stats.set(value);
    });
}

#[inline]
pub fn record_simd_stack_len(len: usize) {
    STATS.with(|stats| {
        let mut value = stats.get();
        value.simd_stack_max_len = value.simd_stack_max_len.max(len as u64);
        stats.set(value);
    });
}

#[inline]
pub fn record_block3_pending_pop(mask: u8) {
    STATS.with(|stats| {
        let mut value = stats.get();
        value.block3_pending_pops += 1;
        value.block3_pending_mask_bits += mask.count_ones() as u64;
        stats.set(value);
    });
}

#[inline]
pub fn record_block3_candidate_mask(mask: u8) {
    STATS.with(|stats| {
        let mut value = stats.get();
        value.block3_candidate_mask_bits += mask.count_ones() as u64;
        if mask != 0 {
            value.block3_candidate_mask_nonzero += 1;
        }
        stats.set(value);
    });
}

#[inline]
pub fn record_block3_full_step() {
    STATS.with(|stats| {
        let mut value = stats.get();
        value.block3_full_steps += 1;
        stats.set(value);
    });
}

#[inline]
pub fn record_block3_scalar_fallback_step() {
    STATS.with(|stats| {
        let mut value = stats.get();
        value.block3_scalar_fallback_steps += 1;
        stats.set(value);
    });
}

#[inline]
pub fn record_block3_step_entry() {
    STATS.with(|stats| {
        let mut value = stats.get();
        value.block3_step_entries += 1;
        stats.set(value);
    });
}
