// src/cache_sim.rs
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt::Write;

/// Cache replacement policy to use when evicting lines.
#[derive(Clone, Copy, Debug)]
pub enum ReplacementPolicy {
    /// Least Recently Used replacement policy.
    Lru,
}

/// Configuration parameters for a cache level.
#[derive(Clone, Debug)]
pub struct CacheConfig {
    /// Human-readable name for this cache level (e.g., "L1D", "L2").
    pub level_name: &'static str,
    /// Total size of the cache in bytes.
    pub size_bytes: usize,
    /// Size of each cache line in bytes (must be power of two).
    pub line_bytes: usize,
    /// Number of ways per set (associativity).
    pub associativity: usize,
    /// Replacement policy to use when evicting lines.
    pub policy: ReplacementPolicy,
}

/// Timing configuration for a single cache level.
#[derive(Clone, Copy, Debug)]
pub struct LevelTiming {
    /// Hit latency (cycles) for this level when data is resident.
    pub hit_latency: u32,
    /// Max outstanding misses this level allows (MSHRs). Prefetch uses only slack beyond 'reserve_for_demand'.
    pub mshrs: usize,
    /// Keep this many MSHRs always available for demand.
    pub reserve_for_demand: usize,
}

/// Timing configuration for the entire memory hierarchy.
#[derive(Clone, Copy, Debug)]
pub struct TimingConfig {
    /// L1 cache timing parameters.
    pub l1: LevelTiming,
    /// L2 cache timing parameters.
    pub l2: LevelTiming,
    /// L3 cache timing parameters.
    pub l3: LevelTiming,
    /// DRAM service latency (cycles) for a single cache line (end-to-end to the core).
    pub mem_latency: u32,
    /// Spacing between line returns from DRAM (bandwidth token interval, in cycles/line).
    /// Example: at 3.5GHz and ~50 GB/s, that's ~(64B / 50e9B/s)*3.5e9 ≈ 4.48 cycles/line -> use 5.
    pub mem_token_interval: u32,
}

/// Statistics for cache hit/miss behavior.
#[derive(Default, Debug, Clone)]
pub struct CacheStats {
    /// Number of cache hits.
    pub hits: u64,
    /// Number of cache misses.
    pub misses: u64,
}

#[derive(Clone, Copy, Debug)]
pub enum PrefetchLevel {
    L1,
    L2,
    L3,
}

/// Extend prefetch stats to hierarchy
#[derive(Default, Debug, Clone)]
pub struct HierarchyPrefetchStats {
    pub l1: PrefetchStats,
    pub l2: PrefetchStats,
    pub l3: PrefetchStats,
}

/// Statistics for prefetch behavior and effectiveness.
#[derive(Default, Debug, Clone)]
pub struct PrefetchStats {
    /// Number of prefetch requests issued.
    pub issued: u64,
    /// Number of prefetches that completed (filled into cache).
    pub filled: u64,
    /// Number of prefetches that were later used by demand accesses.
    pub useful: u64,
    /// Number of prefetches that were evicted before being used.
    pub useless: u64,
    /// Number of prefetch requests for lines already in cache.
    pub redundant: u64,
    /// Number of prefetches that arrived after the demand access.
    pub late: u64,
    /// Sum of (demand_complete_cycle - prefetch_issue_cycle) for useful prefetches.
    pub useful_lead_cycles_sum: u64,
}

/// Configuration for stride analysis.
pub struct StrideCfg {
    /// Cache line size used to quantize addresses into "lines".
    pub line_size: u64,
    /// Keep Markov states only in [-clamp, +clamp]; larger magnitudes are clamped.
    pub clamp: i16,
    /// Whether to ignore stride==0 in the Markov chain (recommended).
    pub ignore_zero_in_markov: bool,
}

impl Default for StrideCfg {
    fn default() -> Self {
        Self {
            line_size: 64,
            clamp: 1024,
            ignore_zero_in_markov: true,
        }
    }
}

/// Analyzer for memory access stride patterns and Markov chain transitions.
#[derive(Default)]
pub struct StrideAnalyzer {
    cfg: StrideCfg,
    // histogram: stride -> count
    hist: HashMap<i16, u64>,
    // markov: prev_stride -> (next_stride -> count)
    trans: HashMap<i16, HashMap<i16, u64>>,
    // rolling state
    last_addr: Option<u64>,
    last_stride: Option<i16>, // only used for Markov
}

impl StrideAnalyzer {
    /// Create a new stride analyzer with the given configuration.
    pub fn new(cfg: StrideCfg) -> Self {
        Self {
            cfg,
            ..Default::default()
        }
    }

    /// Call this at the start of each query (or any place you want to cut the chain).
    pub fn reset_chain(&mut self) {
        self.last_addr = None;
        self.last_stride = None;
    }

    /// Record one memory access. Pass `true` when this access begins a new query.
    pub fn record(&mut self, addr: u64, print: bool) {
        // If this isn’t the very first address, compute a stride
        if let Some(prev) = self.last_addr {
            // --- 1. Compute a signed stride in cache-line units ---
            let raw_delta = (addr as i128 - prev as i128) / self.cfg.line_size as i128;
            let stride = raw_delta.clamp(-self.cfg.clamp as i128, self.cfg.clamp as i128) as i16;

            if print {
                println!("Access Delta: {raw_delta}");
            }

            // --- 2. Guardrails for debugging ---
            if addr < prev && stride >= 0 {
                eprintln!(
                    "BUG: addr < prev but stride >= 0 (addr={:#x}, prev={:#x}, stride={})",
                    addr, prev, stride
                );
            }
            if self.cfg.ignore_zero_in_markov {
                debug_assert!(self.last_stride != Some(0));
            }

            // --- 3. Always update the histogram ---
            *self.hist.entry(stride).or_insert(0) += 1;

            // --- 5. Handle stride == 0 if we’re ignoring zero in Markov ---
            if self.cfg.ignore_zero_in_markov && stride == 0 {
                self.last_stride = None; // break the chain on intra-line reuse
                self.last_addr = Some(addr);
                return;
            }

            // --- 6. Normal Markov update (only non-zero, non-reset strides reach here) ---
            if let Some(prev_s) = self.last_stride {
                let row = self.trans.entry(prev_s).or_default();
                *row.entry(stride).or_insert(0) += 1;
            }
            self.last_stride = Some(stride);
        }

        // --- 7. Update last_addr for the next call ---
        self.last_addr = Some(addr);
    }

    #[inline]
    fn quantize_stride(addr: u64, prev: u64, line_size: u64, clamp: i16) -> i16 {
        // SIGNED delta first. Never do (addr - prev) as u64.
        let delta_lines = ((addr as i128 - prev as i128) / line_size as i128) as i64;
        let c = clamp as i64;
        delta_lines.clamp(-c, c) as i16
    }

    /// Render a simple bar chart for the histogram.
    /// Render a simple bar chart for the histogram.
    pub fn render_histogram(&self, width: usize) -> String {
        let mut s = String::new();
        let mut items: Vec<(i16, u64)> = self.hist.iter().map(|(k, v)| (*k, *v)).collect();
        items.sort_by_key(|(d, _)| *d);
        let maxc = items.iter().map(|(_, c)| *c).max().unwrap_or(1);

        writeln!(
            s,
            "Stride histogram (cache-line deltas; clamped to ±{}):",
            self.cfg.clamp
        )
        .unwrap();
        for (d, c) in items {
            let bar = ((c as f64 / maxc as f64) * width as f64).round() as usize;
            let _ = writeln!(s, "{:>6}: {:<w$} {}", d, "█".repeat(bar), c, w = width);
        }
        s
    }

    /// Render a compact Markov matrix for the top-K most frequent strides.
    /// Render a compact Markov matrix for the top-K most frequent strides.
    pub fn render_markov(&self, top_k: usize) -> String {
        let mut s = String::new();

        // Rank strides by their total outgoing counts.
        let mut ranks: Vec<(i16, u64)> = self
            .trans
            .iter()
            .map(|(k, row)| (*k, row.values().copied().sum()))
            .collect();
        ranks.sort_by(|a, b| b.1.cmp(&a.1)); // desc
        if ranks.is_empty() {
            return "\nStride transition matrix: (empty after filtering; try allow zeros)"
                .to_string();
        }
        let keys: Vec<i16> = ranks.into_iter().take(top_k).map(|(k, _)| k).collect();

        if keys.len() <= 64 {
            writeln!(
                s,
                "\nStride transition matrix (counts, ignoring zeros: {}):",
                self.cfg.ignore_zero_in_markov
            )
            .unwrap();

            // Header
            write!(s, "{:>7}", "").unwrap();
            for &k in &keys {
                write!(s, "{:>7}", k).unwrap();
            }
            writeln!(s).unwrap();

            // Rows
            for &i in &keys {
                write!(s, "{:>7}", i).unwrap();
                for &j in &keys {
                    let v = self
                        .trans
                        .get(&i)
                        .and_then(|row| row.get(&j))
                        .copied()
                        .unwrap_or(0);
                    if v == 0 {
                        write!(s, "{:>7}", ".").unwrap();
                    } else {
                        write!(s, "{:>7}", v).unwrap();
                    }
                }
                writeln!(s).unwrap();
            }
        }

        // Optional: show top transitions per state (easier to read than a big grid)
        writeln!(s, "\nTop transitions per state:").unwrap();
        for &i in &keys {
            if let Some(row) = self.trans.get(&i) {
                let mut v: Vec<(i16, u64)> = row.iter().map(|(k, c)| (*k, *c)).collect();
                v.sort_by(|a, b| b.1.cmp(&a.1));
                let head = v
                    .into_iter()
                    .take(5)
                    .map(|(k, c)| format!("{}:{}", k, c))
                    .collect::<Vec<_>>()
                    .join(", ");
                writeln!(s, "{:>5} → {}", i, head).unwrap();
            }
        }

        s
    }
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

/// A single level of cache in the memory hierarchy.
#[derive(Debug)]
pub struct CacheLevel {
    cfg: CacheConfig,
    num_sets: usize,
    set_index_mask: u64,
    line_offset_bits: u32,
    set_bits: u32,
    sets: Vec<Set>,
    /// Cache hit/miss statistics for this level.
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
    /// Create a new cache level with the given configuration.
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

    /// Set the timing parameters for this cache level.
    pub fn set_timing(&mut self, t: LevelTiming) {
        self.timing = t;
    }

    /// Get the line size in bytes for this cache level.
    pub fn line_bytes(&self) -> usize {
        self.cfg.line_bytes
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

    /// Check if a cache line containing the given address is present in this level.
    pub fn has_line(&self, addr: usize) -> bool {
        let (tag, set_idx) = self.addr_to_tag_index(addr);
        self.sets[set_idx].contains(tag)
    }

    /// Probe the cache for a demand access and return hit/miss status and completion time.
    /// Updates prefetch statistics if the line was prefetched.
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

    /// Get per-set statistics: (hits, misses, evictions, peak_occupancy).
    pub fn per_set_stats(&self) -> (&[u64], &[u64], &[u64], &[usize]) {
        (
            &self.per_set_hits,
            &self.per_set_misses,
            &self.per_set_evictions,
            &self.per_set_peak_occupancy,
        )
    }

    /// Get the total number of sets in this cache level.
    pub fn set_count(&self) -> usize {
        self.num_sets
    }
}

/// Statistics for the entire memory hierarchy.
#[derive(Default, Debug, Clone)]
pub struct HierarchyStats {
    /// L1 cache statistics.
    pub l1: CacheStats,
    /// L2 cache statistics.
    pub l2: CacheStats,
    /// L3 cache statistics.
    pub l3: CacheStats,
    /// Number of accesses that went to main memory.
    pub memory_accesses: u64,
    /// Sum of stall cycles experienced by demands (timing-lite).
    pub demand_stall_cycles: u64,
}

/// Configuration for next-line prefetching.
#[derive(Clone, Copy, Debug)]
pub struct NextLinePrefetch {
    /// Number of cache lines to prefetch ahead.
    pub degree: u32,
    /// Stride between prefetched lines (in cache line units).
    pub stride_lines: u32,
}

/// Type of memory access operation.
pub enum AccessKind {
    /// Read operation.
    Read,
    /// Write operation.
    Write,
}

/// Trait for iterating over memory address traces.
pub trait AddressTrace: Iterator<Item = (usize, AccessKind)> {}
impl<I: Iterator<Item = (usize, AccessKind)>> AddressTrace for I {}

/// Events that can be processed by the memory hierarchy simulator.
#[derive(Clone, Copy, Debug)]
pub enum Event {
    /// Demand access of 'addr'.
    Access(usize),
    /// Software prefetch request for 'addr'.
    PrefetchTo(usize, PrefetchLevel),
    /// Advance simulated time by these cycles.
    Working(u32),
    /// Inform the simulator that a new query has started.
    /// For bookkeeping purposes rather than affecting the sim itself
    NewQuery,
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

/// Complete memory hierarchy simulator with L1/L2/L3 caches and main memory.
pub struct MemoryHierarchy {
    /// L1 cache level (always present).
    pub l1: CacheLevel,
    /// L2 cache level (optional).
    pub l2: Option<CacheLevel>,
    /// L3 cache level (optional).
    pub l3: Option<CacheLevel>,
    /// Aggregate statistics across all levels.
    pub stats: HierarchyStats,

    // timing config
    timing: TimingConfig,
    /// Current simulation time in cycles.
    pub cycle: u64,
    mem: MemPort,

    // prefetch accounting
    pub pf_stats: HierarchyPrefetchStats,

    /// For timeliness: record prefetch issue time per line (L1 line address)
    prefetch_issue_time: HashMap<u64 /*line*/, (PrefetchLevel, u64 /*issue_cycle*/)>,

    // Stride Tracking
    /// Stride pattern analyzer for memory accesses.
    pub stride_analyzer: StrideAnalyzer,
    /// Cache line size in bytes.
    pub line_size: usize,

    // Query counter
    /// Number of queries processed so far.
    pub query_counter: u64,

    // Global per-address access counts (by byte address as seen in Access events)
    addr_access_counts: HashMap<u64, u64>,
}

impl MemoryHierarchy {
    /// Create a new memory hierarchy with the given cache levels.
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

        let line_size = l1.line_bytes();

        let cfg = StrideCfg {
            line_size: line_size as u64,
            clamp: 1024,
            ignore_zero_in_markov: false,
        };
        let stride_analyzer = StrideAnalyzer::new(cfg);

        let mut mh = Self {
            l1,
            l2,
            l3,
            stats: HierarchyStats::default(),
            timing: tc,
            cycle: 0,
            mem: MemPort::new(220, 5),
            pf_stats: HierarchyPrefetchStats::default(),
            prefetch_issue_time: HashMap::new(),

            stride_analyzer,
            line_size,

            query_counter: 0,
            addr_access_counts: HashMap::new(),
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

    /// Set timing configuration for the entire hierarchy.
    pub fn set_timing(&mut self, t: TimingConfig) {
        self.timing = t;
        self.apply_level_timing();
    }

    #[inline]
    fn line_of_l1(&self, addr: usize) -> u64 {
        (addr as u64) >> self.l1.line_offset_bits
    }

    #[inline]
    fn pf_stats_for_level_mut(&mut self, level: PrefetchLevel) -> &mut PrefetchStats {
        match level {
            PrefetchLevel::L1 => &mut self.pf_stats.l1,
            PrefetchLevel::L2 => &mut self.pf_stats.l2,
            PrefetchLevel::L3 => &mut self.pf_stats.l3,
        }
    }

    /// Direct demand access (no timing realism besides per-level latencies + DRAM port).
    /// Returns which level hit or None (memory), and advances self.cycle by the stall observed.
    /// Process a demand memory access and return which cache level hit (or None for memory).
    /// Advances simulation time based on access latency.
    pub fn step_demand(&mut self, addr: usize) -> Option<&'static str> {
        // track stride
        self.stride_analyzer
            .record(addr as u64, self.query_counter < 3);

        let start = self.cycle;

        // --- L1 ---
        let (hit_l1, t1) = self
            .l1
            .demand_probe_and_time(addr, self.cycle, &mut self.pf_stats.l1);
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

            // mark lead distance if prefetched
            let line = self.line_of_l1(addr);
            if let Some((lvl, issue_cycle)) = self.prefetch_issue_time.remove(&line) {
                let lead = self.cycle.saturating_sub(issue_cycle);
                self.pf_stats_for_level_mut(lvl).useful_lead_cycles_sum += lead;
            }
            return Some("L1");
        }

        // --- L2 ---
        if let Some(l2) = &mut self.l2 {
            let (hit_l2, t2) = l2.demand_probe_and_time(addr, self.cycle, &mut self.pf_stats.l2);
            if hit_l2 {
                // promote to L1
                self.l1.mark_ready(addr, t2);
                self.cycle = t2 + self.l1.timing.hit_latency as u64;
                self.stats.demand_stall_cycles += self.cycle - start;
                self.stats.l1 = self.l1.stats.clone();
                self.stats.l2 = l2.stats.clone();
                if let Some(l3) = &self.l3 {
                    self.stats.l3 = l3.stats.clone();
                }

                // if it had been prefetched earlier
                let line = self.line_of_l1(addr);
                if let Some((lvl, issue_cycle)) = self.prefetch_issue_time.remove(&line) {
                    let lead = self.cycle.saturating_sub(issue_cycle);
                    self.pf_stats_for_level_mut(lvl).useful_lead_cycles_sum += lead;
                }
                return Some("L2");
            }
        }

        // --- L3 ---
        if let Some(l3) = &mut self.l3 {
            let (hit_l3, t3) = l3.demand_probe_and_time(addr, self.cycle, &mut self.pf_stats.l3);
            if hit_l3 {
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

                let line = self.line_of_l1(addr);
                if let Some((lvl, issue_cycle)) = self.prefetch_issue_time.remove(&line) {
                    let lead = self.cycle.saturating_sub(issue_cycle);
                    self.pf_stats_for_level_mut(lvl).useful_lead_cycles_sum += lead;
                }
                return Some("L3");
            }
        }

        // --- Memory miss ---
        self.stats.memory_accesses += 1;
        let ret = self.mem.schedule(self.cycle);

        // install inclusively
        if let Some(l3) = &mut self.l3 {
            l3.mark_ready(addr, ret);
        }
        if let Some(l2) = &mut self.l2 {
            l2.mark_ready(addr, ret);
        }
        self.l1.mark_ready(addr, ret);

        // Late prefetch: whichever level’s issue time exists
        let line = self.line_of_l1(addr);
        if self.prefetch_issue_time.remove(&line).is_some() {
            // We don't know which level it targeted, but usually L1
            self.pf_stats.l1.late += 1;
        }

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
    pub fn step_prefetch_to(&mut self, addr: usize, level: PrefetchLevel) {
        let stats = match level {
            PrefetchLevel::L1 => &mut self.pf_stats.l1,
            PrefetchLevel::L2 => &mut self.pf_stats.l2,
            PrefetchLevel::L3 => &mut self.pf_stats.l3,
        };
        stats.issued += 1;

        // Helper closures for convenience
        let schedule_mem = |mem: &mut MemPort, now: u64| mem.schedule(now);

        match level {
            PrefetchLevel::L1 => {
                // same behaviour as before
                if self.l1.has_line(addr) {
                    stats.redundant += 1;
                    return;
                }
                if let Some(l2) = &mut self.l2 {
                    if l2.has_line(addr) {
                        let t = self.cycle + l2.timing.hit_latency as u64;
                        self.l1.prefetch_install_lowprio(addr, t);
                        stats.filled += 1;
                        self.prefetch_issue_time
                            .entry(self.line_of_l1(addr))
                            .or_insert((PrefetchLevel::L1, self.cycle));
                        return;
                    }
                }
                if let Some(l3) = &mut self.l3 {
                    if l3.has_line(addr) {
                        let t = self.cycle + l3.timing.hit_latency as u64;
                        self.l1.prefetch_install_lowprio(addr, t);
                        stats.filled += 1;
                        self.prefetch_issue_time
                            .entry(self.line_of_l1(addr))
                            .or_insert((PrefetchLevel::L1, self.cycle));
                        return;
                    }
                }
                // memory
                let ret = schedule_mem(&mut self.mem, self.cycle);
                if let Some(l3) = &mut self.l3 {
                    l3.prefetch_install_lowprio(addr, ret);
                }
                if let Some(l2) = &mut self.l2 {
                    l2.prefetch_install_lowprio(addr, ret);
                }
                self.l1.prefetch_install_lowprio(addr, ret);
                stats.filled += 1;
                self.prefetch_issue_time
                    .entry(self.line_of_l1(addr))
                    .or_insert((PrefetchLevel::L1, self.cycle));
            }

            PrefetchLevel::L2 => {
                // Only up to L2 (doesn't touch L1 until demand)
                if let Some(l2) = &mut self.l2 {
                    if l2.has_line(addr) {
                        stats.redundant += 1;
                        return;
                    }
                    if let Some(l3) = &mut self.l3 {
                        if l3.has_line(addr) {
                            let t = self.cycle + l3.timing.hit_latency as u64;
                            l2.prefetch_install_lowprio(addr, t);
                            stats.filled += 1;

                            self.prefetch_issue_time
                                .entry(self.line_of_l1(addr))
                                .or_insert((PrefetchLevel::L2, self.cycle));
                            return;
                        }
                    }
                    let ret = schedule_mem(&mut self.mem, self.cycle);
                    if let Some(l3) = &mut self.l3 {
                        l3.prefetch_install_lowprio(addr, ret);
                    }
                    l2.prefetch_install_lowprio(addr, ret);
                    stats.filled += 1;

                    self.prefetch_issue_time
                        .entry(self.line_of_l1(addr))
                        .or_insert((PrefetchLevel::L2, self.cycle));
                }
            }

            PrefetchLevel::L3 => {
                if let Some(l3) = &mut self.l3 {
                    if l3.has_line(addr) {
                        stats.redundant += 1;
                        return;
                    }
                    let ret = schedule_mem(&mut self.mem, self.cycle);
                    l3.prefetch_install_lowprio(addr, ret);
                    stats.filled += 1;

                    self.prefetch_issue_time
                        .entry(self.line_of_l1(addr))
                        .or_insert((PrefetchLevel::L3, self.cycle));
                }
            }
        }
    }

    /// Advance simulated time by N cycles (e.g., work between accesses).
    /// Advance simulated time by N cycles (e.g., work between accesses).
    pub fn step_work(&mut self, cycles: u32) {
        self.cycle += cycles as u64;
    }

    /// Unified event API
    /// Process a simulation event and return which cache level hit (if applicable).
    pub fn step_event(&mut self, ev: Event) -> Option<&'static str> {
        match ev {
            Event::NewQuery => {
                self.query_counter += 1;
                self.stride_analyzer.reset_chain();

                if self.query_counter <= 3 {
                    println!("New Query");
                }

                None
            }

            Event::Access(addr) => {
                *self.addr_access_counts.entry(addr as u64).or_insert(0) += 1;

                let before = self.cycle;
                let hitlvl = self.step_demand(addr);

                // If this access hit a prefetched line, record lead distance
                let line = self.line_of_l1(addr);
                if let Some((lvl, issue_cycle)) = self.prefetch_issue_time.remove(&line) {
                    // Add lead cycles only if the prefetch arrived in time
                    if self.l1.has_line(addr)
                        || self.l2.as_ref().map_or(false, |l2| l2.has_line(addr))
                        || self.l3.as_ref().map_or(false, |l3| l3.has_line(addr))
                    {
                        let lead = (before.max(self.cycle)) - issue_cycle;
                        self.pf_stats_for_level_mut(lvl).useful_lead_cycles_sum += lead;
                    }
                }

                hitlvl
            }
            Event::PrefetchTo(addr, lvl) => {
                self.step_prefetch_to(addr, lvl);
                None
            }
            Event::Working(c) => {
                self.step_work(c);
                None
            }
        }
    }

    /// Run a complete address trace through the memory hierarchy.
    pub fn run_trace<T: AddressTrace>(&mut self, trace: &mut T) {
        for (addr, _kind) in trace {
            self.step_demand(addr);
        }
    }

    /// Take a snapshot of current hierarchy statistics.
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

    /// Reset all statistics and simulation state to initial values.
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
        self.pf_stats = HierarchyPrefetchStats::default();
        self.prefetch_issue_time.clear();
        self.addr_access_counts.clear();
    }

    /// Print top-K most frequently accessed byte addresses across the whole run.
    /// If quantization by line is desired, call with addresses already masked/shifted by the caller.
    /// Print the top-K most frequently accessed addresses with access counts.
    pub fn print_top_addresses(&self, top_k: usize) {
        let mut items: Vec<(u64, u64)> = self
            .addr_access_counts
            .iter()
            .map(|(&a, &c)| (a, c))
            .collect();
        if items.is_empty() {
            println!("Top {} most frequently accessed addresses: (none)", top_k);
            return;
        }
        items.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        let take_n = top_k.min(items.len());
        let maxc = items
            .iter()
            .take(take_n)
            .map(|&(_, c)| c)
            .max()
            .unwrap_or(1);

        println!("Top {} most frequently accessed addresses:", take_n);
        for &(addr, cnt) in items.iter().take(take_n) {
            let bar_len = ((cnt as f64 / maxc as f64) * 50.0).round() as usize;
            println!("{:>5}: {:<50} {}", addr, "█".repeat(bar_len), cnt);
        }
    }
}

/// Ready-made CPU profiles with realistic cache configurations and timing defaults.
pub mod profiles {
    use super::*;

    /// Create a memory hierarchy approximating AMD Zen 3 architecture.
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

    /// Create a memory hierarchy approximating Apple M1 architecture.
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
