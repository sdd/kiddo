// src/cache_sim.rs
use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Clone, Copy, Debug)]
pub enum ReplacementPolicy {
    Lru,
}

#[derive(Clone, Debug)]
pub struct CacheConfig {
    pub level_name: &'static str,
    pub size_bytes: usize,
    pub line_bytes: usize,
    pub associativity: usize,
    pub policy: ReplacementPolicy,
}

#[derive(Clone, Copy, Debug)]
pub struct LevelTiming {
    /// Hit latency (cycles) for this level when data is resident.
    pub hit_latency: u32,
    /// Max outstanding misses this level allows (MSHRs). Prefetch uses only slack beyond 'reserve_for_demand'.
    pub mshrs: usize,
    /// Keep this many MSHRs always available for demand.
    pub reserve_for_demand: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct TimingConfig {
    pub l1: LevelTiming,
    pub l2: LevelTiming,
    pub l3: LevelTiming,
    /// DRAM service latency (cycles) for a single cache line (end-to-end to the core).
    pub mem_latency: u32,
    /// Spacing between line returns from DRAM (bandwidth token interval, in cycles/line).
    /// Example: at 3.5GHz and ~50 GB/s, that's ~(64B / 50e9B/s)*3.5e9 ≈ 4.48 cycles/line -> use 5.
    pub mem_token_interval: u32,
}

#[derive(Default, Debug, Clone)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
}

#[derive(Default, Debug, Clone)]
pub struct PrefetchStats {
    pub issued: u64,
    pub filled: u64,
    pub useful: u64,
    pub useless: u64,
    pub redundant: u64,
    pub late: u64,
    /// Sum of (demand_complete_cycle - prefetch_issue_cycle) for useful prefetches.
    pub useful_lead_cycles_sum: u64,
}

#[derive(Debug)]
struct Set {
    ways: VecDeque<u64>, // MRU..LRU (front = MRU)
    capacity: usize,
}

impl Set {
    fn new(capacity: usize) -> Self {
        Self {
            ways: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    fn access_lru(&mut self, tag: u64) -> bool {
        if let Some(pos) = self.ways.iter().position(|&t| t == tag) {
            let val = self.ways.remove(pos).unwrap();
            self.ways.push_front(val);
            true
        } else {
            false
        }
    }

    /// Insert as MRU (front). Returns evicted tag if eviction occurred.
    fn insert_mru_with_evict(&mut self, tag: u64) -> Option<u64> {
        let evicted = if self.ways.len() == self.capacity {
            self.ways.pop_back()
        } else {
            None
        };
        self.ways.push_front(tag);
        evicted
    }

    /// Insert as LRU (back). Low-priority install (used for prefetch).
    fn insert_lru_tail_with_evict(&mut self, tag: u64) -> Option<u64> {
        let evicted = if self.ways.len() == self.capacity {
            self.ways.pop_back()
        } else {
            None
        };
        self.ways.push_back(tag);
        evicted
    }

    fn contains(&self, tag: u64) -> bool {
        self.ways.iter().any(|&t| t == tag)
    }

    fn occupancy(&self) -> usize {
        self.ways.len()
    }

    fn evict_lru_tag(&self) -> Option<u64> {
        self.ways.back().copied()
    }
}

#[derive(Debug)]
pub struct CacheLevel {
    cfg: CacheConfig,
    num_sets: usize,
    set_index_mask: u64,
    line_offset_bits: u32,
    set_bits: u32,
    sets: Vec<Set>,
    pub stats: CacheStats,

    // per-set stats
    per_set_hits: Vec<u64>,
    per_set_misses: Vec<u64>,
    per_set_evictions: Vec<u64>,
    per_set_peak_occupancy: Vec<usize>,

    // timing & MSHR/lightweight queues
    timing: LevelTiming,
    /// key: line_addr (addr >> line_bits), val: cycle when line becomes usable at this level
    ready_at: HashMap<u64, u64>,
    /// prefetch marks: for each set, which tags were installed by prefetch and not yet used
    prefetched_tags: Vec<HashSet<u64>>,
}

impl CacheLevel {
    pub fn new(cfg: CacheConfig) -> Self {
        assert!(cfg.size_bytes > 0);
        assert!(cfg.line_bytes.is_power_of_two());
        let num_sets = cfg.size_bytes / (cfg.line_bytes * cfg.associativity);
        assert!(num_sets.is_power_of_two());

        let line_offset_bits = cfg.line_bytes.trailing_zeros();
        let set_bits = num_sets.trailing_zeros();
        let set_index_mask = (1u64 << set_bits) - 1;

        let sets = (0..num_sets).map(|_| Set::new(cfg.associativity)).collect();
        let mut prefetched_tags = Vec::with_capacity(num_sets);
        for _ in 0..num_sets {
            prefetched_tags.push(HashSet::new());
        }

        Self {
            cfg,
            num_sets,
            set_index_mask,
            line_offset_bits,
            set_bits,
            sets,
            stats: CacheStats::default(),
            per_set_hits: vec![0; num_sets],
            per_set_misses: vec![0; num_sets],
            per_set_evictions: vec![0; num_sets],
            per_set_peak_occupancy: vec![0; num_sets],
            timing: LevelTiming {
                hit_latency: 4,
                mshrs: 8,
                reserve_for_demand: 2,
            },
            ready_at: HashMap::new(),
            prefetched_tags,
        }
    }

    pub fn set_timing(&mut self, t: LevelTiming) {
        self.timing = t;
    }

    #[inline]
    fn addr_to_line_set_tag(&self, addr: usize) -> (u64 /*line*/, usize /*set*/, u64 /*tag*/) {
        let a = addr as u64;
        let line_addr = a >> self.line_offset_bits;
        let set_index = (line_addr & self.set_index_mask) as usize;
        let tag = line_addr >> self.set_bits;
        (line_addr, set_index, tag)
    }

    #[inline]
    fn addr_to_tag_index(&self, addr: usize) -> (u64, usize) {
        let (_, set_index, tag) = self.addr_to_line_set_tag(addr);
        (tag, set_index)
    }

    fn mark_ready(&mut self, addr: usize, ready_cycle: u64) {
        let (line, set_idx, tag) = self.addr_to_line_set_tag(addr);
        self.ready_at.insert(line, ready_cycle);
        // Inclusive fill policy at this level
        let set = &mut self.sets[set_idx];
        if !set.contains(tag) {
            // install as MRU (demand path usually), but caller chooses which insert
            let ev = set.insert_mru_with_evict(tag);
            if ev.is_some() {
                self.per_set_evictions[set_idx] += 1;
            }
        } else {
            set.access_lru(tag);
        }
        let occ = set.occupancy();
        if occ > self.per_set_peak_occupancy[set_idx] {
            self.per_set_peak_occupancy[set_idx] = occ;
        }
    }

    /// Low-priority install for prefetch (to reduce pollution): insert at LRU end.
    fn prefetch_install_lowprio(&mut self, addr: usize, ready_cycle: u64) {
        let (line, set_idx, tag) = self.addr_to_line_set_tag(addr);
        self.ready_at.insert(line, ready_cycle);
        let set = &mut self.sets[set_idx];
        if !set.contains(tag) {
            let ev_tag = set.insert_lru_tail_with_evict(tag);
            if ev_tag.is_some() {
                self.per_set_evictions[set_idx] += 1;
            }
            self.prefetched_tags[set_idx].insert(tag);
        } else {
            // already present; mark as prefetched if not already
            self.prefetched_tags[set_idx].insert(tag);
        }
        let occ = set.occupancy();
        if occ > self.per_set_peak_occupancy[set_idx] {
            self.per_set_peak_occupancy[set_idx] = occ;
        }
    }

    pub fn has_line(&self, addr: usize) -> bool {
        let (tag, set_idx) = self.addr_to_tag_index(addr);
        self.sets[set_idx].contains(tag)
    }

    pub fn demand_probe_and_time(
        &mut self,
        addr: usize,
        now: u64,
        pf_stats: &mut PrefetchStats,
    ) -> (bool /*hit*/, u64 /*complete_cycle*/) {
        let (line, set_idx, tag) = self.addr_to_line_set_tag(addr);
        let set = &mut self.sets[set_idx];

        if set.access_lru(tag) {
            // resident hit; honor readiness timing
            let ready = *self.ready_at.get(&line).unwrap_or(&now);
            self.stats.hits += 1;
            self.per_set_hits[set_idx] += 1;

            // prefetch usefulness check
            if self.prefetched_tags[set_idx].remove(&tag) {
                pf_stats.useful += 1;
                // lead distance = now - issue; we don't track issue time per-line at this level,
                // but the hierarchy will track it and add to stats when demand completes.
            }

            let complete = now.max(ready) + self.timing.hit_latency as u64;
            return (true, complete);
        }

        // miss
        self.stats.misses += 1;
        self.per_set_misses[set_idx] += 1;

        (false, now + self.timing.hit_latency as u64) // caller will schedule deeper service
    }

    pub fn per_set_stats(&self) -> (&[u64], &[u64], &[u64], &[usize]) {
        (
            &self.per_set_hits,
            &self.per_set_misses,
            &self.per_set_evictions,
            &self.per_set_peak_occupancy,
        )
    }

    pub fn set_count(&self) -> usize {
        self.num_sets
    }
}

#[derive(Default, Debug, Clone)]
pub struct HierarchyStats {
    pub l1: CacheStats,
    pub l2: CacheStats,
    pub l3: CacheStats,
    pub memory_accesses: u64,
    /// Sum of stall cycles experienced by demands (timing-lite).
    pub demand_stall_cycles: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct NextLinePrefetch {
    pub degree: u32,
    pub stride_lines: u32,
}

pub enum AccessKind {
    Read,
    Write,
}

pub trait AddressTrace: Iterator<Item = (usize, AccessKind)> {}
impl<I: Iterator<Item = (usize, AccessKind)>> AddressTrace for I {}

#[derive(Clone, Copy, Debug)]
pub enum Event {
    /// Demand access of 'addr'.
    Access(usize),
    /// Software prefetch request for 'addr'.
    Prefetch(usize),
    /// Advance simulated time by these cycles.
    Working(u32),
}

struct MemPort {
    next_ready: u64,
    token_interval: u64,
    mem_latency: u32,
}

impl MemPort {
    fn new(mem_latency: u32, token_interval: u32) -> Self {
        Self {
            next_ready: 0,
            token_interval: token_interval as u64,
            mem_latency,
        }
    }

    /// Schedule one DRAM line; returns cycle when data is usable by core.
    fn schedule(&mut self, now: u64) -> u64 {
        // Serialize by tokens
        let start = now.max(self.next_ready);
        let ret = start + self.mem_latency as u64;
        self.next_ready = start + self.token_interval;
        ret
    }
}

pub struct MemoryHierarchy {
    pub l1: CacheLevel,
    pub l2: Option<CacheLevel>,
    pub l3: Option<CacheLevel>,
    pub stats: HierarchyStats,

    // timing config
    timing: TimingConfig,
    pub cycle: u64,
    mem: MemPort,

    // prefetch accounting
    pub pf_stats_l1: PrefetchStats,
    /// For timeliness: record prefetch issue time per line (L1 line address)
    prefetch_issue_time: HashMap<u64 /*line*/, u64 /*issue_cycle*/>,
}

impl MemoryHierarchy {
    pub fn new(l1: CacheLevel, l2: Option<CacheLevel>, l3: Option<CacheLevel>) -> Self {
        // sensible defaults; override with set_timing()
        let tc = TimingConfig {
            l1: LevelTiming {
                hit_latency: 4,
                mshrs: 8,
                reserve_for_demand: 2,
            },
            l2: LevelTiming {
                hit_latency: 12,
                mshrs: 12,
                reserve_for_demand: 2,
            },
            l3: LevelTiming {
                hit_latency: 40,
                mshrs: 32,
                reserve_for_demand: 4,
            },
            mem_latency: 220,
            mem_token_interval: 5,
        };

        let mut mh = Self {
            l1,
            l2,
            l3,
            stats: HierarchyStats::default(),
            timing: tc,
            cycle: 0,
            mem: MemPort::new(220, 5),
            pf_stats_l1: PrefetchStats::default(),
            prefetch_issue_time: HashMap::new(),
        };
        // propagate timing into levels
        mh.apply_level_timing();
        mh
    }

    fn apply_level_timing(&mut self) {
        self.l1.set_timing(self.timing.l1);
        if let Some(l2) = &mut self.l2 {
            l2.set_timing(self.timing.l2);
        }
        if let Some(l3) = &mut self.l3 {
            l3.set_timing(self.timing.l3);
        }
        self.mem = MemPort::new(self.timing.mem_latency, self.timing.mem_token_interval);
    }

    pub fn set_timing(&mut self, t: TimingConfig) {
        self.timing = t;
        self.apply_level_timing();
    }

    #[inline]
    fn line_of_l1(&self, addr: usize) -> u64 {
        (addr as u64) >> self.l1.line_offset_bits
    }

    /// Direct demand access (no timing realism besides per-level latencies + DRAM port).
    /// Returns which level hit or None (memory), and advances self.cycle by the stall observed.
    pub fn step_demand(&mut self, addr: usize) -> Option<&'static str> {
        let start = self.cycle;

        // L1
        let (hit_l1, t1) = self
            .l1
            .demand_probe_and_time(addr, self.cycle, &mut self.pf_stats_l1);
        if hit_l1 {
            self.cycle = t1;
            self.stats.demand_stall_cycles += self.cycle - start;
            self.stats.l1 = self.l1.stats.clone();
            if let Some(l2) = &self.l2 {
                self.stats.l2 = l2.stats.clone();
            }
            if let Some(l3) = &self.l3 {
                self.stats.l3 = l3.stats.clone();
            }
            return Some("L1");
        }

        // L2
        if let Some(l2) = &mut self.l2 {
            let (hit_l2, t2) = l2.demand_probe_and_time(addr, self.cycle, &mut self.pf_stats_l1);
            if hit_l2 {
                // Promote to L1 (inclusive) at t2
                self.l1.mark_ready(addr, t2);
                self.cycle = t2 + self.l1.timing.hit_latency as u64;
                self.stats.demand_stall_cycles += self.cycle - start;
                self.stats.l1 = self.l1.stats.clone();
                self.stats.l2 = l2.stats.clone();
                if let Some(l3) = &self.l3 {
                    self.stats.l3 = l3.stats.clone();
                }
                return Some("L2");
            }
        }

        // L3
        if let Some(l3) = &mut self.l3 {
            let (hit_l3, t3) = l3.demand_probe_and_time(addr, self.cycle, &mut self.pf_stats_l1);
            if hit_l3 {
                // Promote up at t3
                if let Some(l2) = &mut self.l2 {
                    l2.mark_ready(addr, t3);
                }
                self.l1.mark_ready(addr, t3);
                self.cycle = t3 + self.l1.timing.hit_latency as u64;
                self.stats.demand_stall_cycles += self.cycle - start;
                self.stats.l1 = self.l1.stats.clone();
                if let Some(l2) = &self.l2 {
                    self.stats.l2 = l2.stats.clone();
                }
                self.stats.l3 = l3.stats.clone();
                return Some("L3");
            }
        }

        // Memory miss
        self.stats.memory_accesses += 1;
        let ret = self.mem.schedule(self.cycle);
        // Install everywhere inclusively at 'ret'
        if let Some(l3) = &mut self.l3 {
            l3.mark_ready(addr, ret);
        }
        if let Some(l2) = &mut self.l2 {
            l2.mark_ready(addr, ret);
        }
        self.l1.mark_ready(addr, ret);

        // If a prefetch was issued for this line earlier, count 'late'
        let line = self.line_of_l1(addr);
        if self.prefetch_issue_time.remove(&line).is_some() {
            self.pf_stats_l1.late += 1;
        }

        // Demand completes after L1 latency
        self.cycle = ret + self.l1.timing.hit_latency as u64;
        self.stats.demand_stall_cycles += self.cycle - start;
        self.stats.l1 = self.l1.stats.clone();
        if let Some(l2) = &self.l2 {
            self.stats.l2 = l2.stats.clone();
        }
        if let Some(l3) = &self.l3 {
            self.stats.l3 = l3.stats.clone();
        }
        None
    }

    /// Software prefetch: schedules memory service if line not already present; installs at L2/L1 with low priority at the ready time.
    pub fn step_prefetch(&mut self, addr: usize) {
        self.pf_stats_l1.issued += 1;

        // Redundant if already in L1
        if self.l1.has_line(addr) {
            self.pf_stats_l1.redundant += 1;
            return;
        }
        // If present in L2/L3, we can just mark low-priority install to L1 at 'now + hit_latency'.
        if let Some(l2) = &mut self.l2 {
            if l2.has_line(addr) {
                let t = self.cycle + l2.timing.hit_latency as u64;
                self.l1.prefetch_install_lowprio(addr, t);
                self.pf_stats_l1.filled += 1;
                self.prefetch_issue_time
                    .entry(self.line_of_l1(addr))
                    .or_insert(self.cycle);
                return;
            }
        }
        if let Some(l3) = &mut self.l3 {
            if l3.has_line(addr) {
                let t = self.cycle + l3.timing.hit_latency as u64;
                self.l1.prefetch_install_lowprio(addr, t);
                self.pf_stats_l1.filled += 1;
                self.prefetch_issue_time
                    .entry(self.line_of_l1(addr))
                    .or_insert(self.cycle);
                return;
            }
        }

        // Need memory: schedule via DRAM port, install low-priority at return
        let ret = self.mem.schedule(self.cycle);
        if let Some(l3) = &mut self.l3 {
            l3.prefetch_install_lowprio(addr, ret);
        }
        if let Some(l2) = &mut self.l2 {
            l2.prefetch_install_lowprio(addr, ret);
        }
        self.l1.prefetch_install_lowprio(addr, ret);
        self.pf_stats_l1.filled += 1;
        self.prefetch_issue_time
            .entry(self.line_of_l1(addr))
            .or_insert(self.cycle);
    }

    /// Advance simulated time by N cycles (e.g., work between accesses).
    pub fn step_work(&mut self, cycles: u32) {
        self.cycle += cycles as u64;
    }

    /// Unified event API
    pub fn step_event(&mut self, ev: Event) -> Option<&'static str> {
        match ev {
            Event::Access(addr) => {
                let before = self.cycle;
                let hitlvl = self.step_demand(addr);

                // If this access hit a prefetched line, add lead distance (now - issue) to stats
                let line = self.line_of_l1(addr);
                if let Some(issue) = self.prefetch_issue_time.remove(&line) {
                    // If it was late, 'late' counter already incremented in step_demand; only add lead for useful
                    if self.l1.has_line(addr) {
                        self.pf_stats_l1.useful_lead_cycles_sum += (before.max(self.cycle)) - issue;
                    }
                }
                hitlvl
            }
            Event::Prefetch(addr) => {
                self.step_prefetch(addr);
                None
            }
            Event::Working(c) => {
                self.step_work(c);
                None
            }
        }
    }

    pub fn run_trace<T: AddressTrace>(&mut self, trace: &mut T) {
        for (addr, _kind) in trace {
            self.step_demand(addr);
        }
    }

    pub fn snapshot_stats(&self) -> HierarchyStats {
        HierarchyStats {
            l1: self.l1.stats.clone(),
            l2: self
                .l2
                .as_ref()
                .map(|c| c.stats.clone())
                .unwrap_or_default(),
            l3: self
                .l3
                .as_ref()
                .map(|c| c.stats.clone())
                .unwrap_or_default(),
            memory_accesses: self.stats.memory_accesses,
            demand_stall_cycles: self.stats.demand_stall_cycles,
        }
    }

    pub fn reset(&mut self) {
        self.l1.stats = CacheStats::default();
        if let Some(l2) = &mut self.l2 {
            l2.stats = CacheStats::default();
        }
        if let Some(l3) = &mut self.l3 {
            l3.stats = CacheStats::default();
        }
        self.stats = HierarchyStats::default();
        self.cycle = 0;
        self.mem.next_ready = 0;
        self.pf_stats_l1 = PrefetchStats::default();
        self.prefetch_issue_time.clear();
    }
}

// Ready-made CPU profiles + timing defaults
pub mod profiles {
    use super::*;

    pub fn zen3() -> MemoryHierarchy {
        let mut l1 = CacheLevel::new(CacheConfig {
            level_name: "L1D",
            size_bytes: 32 * 1024,
            line_bytes: 64,
            associativity: 8,
            policy: ReplacementPolicy::Lru,
        });
        let mut l2 = CacheLevel::new(CacheConfig {
            level_name: "L2",
            size_bytes: 512 * 1024,
            line_bytes: 64,
            associativity: 8,
            policy: ReplacementPolicy::Lru,
        });
        let mut l3 = CacheLevel::new(CacheConfig {
            level_name: "L3",
            size_bytes: 16 * 1024 * 1024,
            line_bytes: 64,
            associativity: 16,
            policy: ReplacementPolicy::Lru,
        });

        // Timing-ish defaults
        l1.set_timing(LevelTiming {
            hit_latency: 4,
            mshrs: 8,
            reserve_for_demand: 2,
        });
        l2.set_timing(LevelTiming {
            hit_latency: 12,
            mshrs: 12,
            reserve_for_demand: 2,
        });
        l3.set_timing(LevelTiming {
            hit_latency: 40,
            mshrs: 32,
            reserve_for_demand: 4,
        });

        let mut mh = MemoryHierarchy::new(l1, Some(l2), Some(l3));
        mh.set_timing(TimingConfig {
            l1: LevelTiming {
                hit_latency: 4,
                mshrs: 8,
                reserve_for_demand: 2,
            },
            l2: LevelTiming {
                hit_latency: 12,
                mshrs: 12,
                reserve_for_demand: 2,
            },
            l3: LevelTiming {
                hit_latency: 40,
                mshrs: 32,
                reserve_for_demand: 4,
            },
            mem_latency: 160,
            mem_token_interval: 3,
        });
        mh
    }

    pub fn m1() -> MemoryHierarchy {
        let mut l1 = CacheLevel::new(CacheConfig {
            level_name: "L1D",
            size_bytes: 64 * 1024,
            line_bytes: 64,
            associativity: 8,
            policy: ReplacementPolicy::Lru,
        });
        let mut l2 = CacheLevel::new(CacheConfig {
            level_name: "L2",
            size_bytes: 12 * 1024 * 1024,
            line_bytes: 64,
            associativity: 12,
            policy: ReplacementPolicy::Lru,
        });

        l1.set_timing(LevelTiming {
            hit_latency: 4,
            mshrs: 8,
            reserve_for_demand: 2,
        });
        l2.set_timing(LevelTiming {
            hit_latency: 16,
            mshrs: 16,
            reserve_for_demand: 2,
        });

        let mut mh = MemoryHierarchy::new(l1, Some(l2), None);
        mh.set_timing(TimingConfig {
            l1: LevelTiming {
                hit_latency: 4,
                mshrs: 8,
                reserve_for_demand: 2,
            },
            l2: LevelTiming {
                hit_latency: 16,
                mshrs: 16,
                reserve_for_demand: 2,
            },
            l3: LevelTiming {
                hit_latency: 40,
                mshrs: 32,
                reserve_for_demand: 4,
            },
            mem_latency: 260,
            mem_token_interval: 6,
        });
        mh
    }
}
