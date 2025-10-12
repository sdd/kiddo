// src/cache_sim.rs
use std::collections::VecDeque;

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

#[derive(Default, Debug, Clone)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
}

#[derive(Debug)]
struct Set {
    ways: VecDeque<u64>, // MRU..LRU
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

    fn insert_lru_with_evict(&mut self, tag: u64) -> bool {
        let mut evicted = false;
        if self.ways.len() == self.capacity {
            self.ways.pop_back();
            evicted = true;
        }
        self.ways.push_front(tag);
        evicted
    }

    fn contains(&self, tag: u64) -> bool {
        self.ways.iter().any(|&t| t == tag)
    }

    fn occupancy(&self) -> usize {
        self.ways.len()
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

    per_set_hits: Vec<u64>,
    per_set_misses: Vec<u64>,
    per_set_evictions: Vec<u64>,
    per_set_peak_occupancy: Vec<usize>,
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
        }
    }

    #[inline]
    fn addr_to_tag_index(&self, addr: usize) -> (u64, usize) {
        let a = addr as u64;
        let line_addr = a >> self.line_offset_bits;
        let set_index = (line_addr & self.set_index_mask) as usize;
        let tag = line_addr >> self.set_bits;
        (tag, set_index)
    }

    pub fn probe(&mut self, addr: usize) -> bool {
        let (tag, set_idx) = self.addr_to_tag_index(addr);
        let set = &mut self.sets[set_idx];
        let hit = set.access_lru(tag);

        if hit {
            self.stats.hits += 1;
            self.per_set_hits[set_idx] += 1;
        } else {
            self.stats.misses += 1;
            self.per_set_misses[set_idx] += 1;
        }
        hit
    }

    pub fn fill(&mut self, addr: usize) {
        let (tag, set_idx) = self.addr_to_tag_index(addr);
        let set = &mut self.sets[set_idx];
        if !set.contains(tag) {
            let ev = set.insert_lru_with_evict(tag);
            if ev {
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

    pub fn reset_stats(&mut self) {
        self.stats = CacheStats::default();
        self.per_set_hits.fill(0);
        self.per_set_misses.fill(0);
        self.per_set_evictions.fill(0);
        self.per_set_peak_occupancy.fill(0);
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
}

#[derive(Clone, Copy, Debug)]
pub struct NextLinePrefetch {
    pub degree: u32,
    pub stride_lines: u32,
}

#[derive(Default)]
pub struct PrefetchStats {
    pub issued: u64,
    pub useful: u64,
    pub useless: u64,
}

pub enum AccessKind {
    Read,
    Write,
}

pub trait AddressTrace: Iterator<Item = (usize, AccessKind)> {}
impl<I: Iterator<Item = (usize, AccessKind)>> AddressTrace for I {}

pub struct MemoryHierarchy {
    pub l1: CacheLevel,
    pub l2: Option<CacheLevel>,
    pub l3: Option<CacheLevel>,
    pub stats: HierarchyStats,

    pub l1_nextline: Option<NextLinePrefetch>,
    pub pf_stats_l1: PrefetchStats,
}

impl MemoryHierarchy {
    pub fn new(l1: CacheLevel, l2: Option<CacheLevel>, l3: Option<CacheLevel>) -> Self {
        Self {
            l1,
            l2,
            l3,
            stats: HierarchyStats::default(),
            l1_nextline: None,
            pf_stats_l1: PrefetchStats::default(),
        }
    }

    pub fn step(&mut self, addr: usize) -> Option<&'static str> {
        if self.l1.probe(addr) {
            return Some("L1");
        }

        if let Some(l2) = &mut self.l2 {
            if l2.probe(addr) {
                self.l1.fill(addr);
                return Some("L2");
            }
        }

        if let Some(l3) = &mut self.l3 {
            if l3.probe(addr) {
                if let Some(l2) = &mut self.l2 {
                    l2.fill(addr);
                }
                self.l1.fill(addr);
                return Some("L3");
            }
        }

        // Memory miss
        self.stats.memory_accesses += 1;
        if let Some(l3) = &mut self.l3 {
            l3.fill(addr);
        }
        if let Some(l2) = &mut self.l2 {
            l2.fill(addr);
        }
        self.l1.fill(addr);

        // optional next-line prefetch
        if let Some(p) = self.l1_nextline {
            let line_addr = (addr as u64) >> self.l1.line_offset_bits;
            for i in 1..=p.degree {
                let pf_line = line_addr.wrapping_add(i as u64 * p.stride_lines as u64);
                let pf_addr = (pf_line << self.l1.line_offset_bits) as usize;
                if !self.l1.probe(pf_addr) {
                    self.l1.fill(pf_addr);
                    self.pf_stats_l1.issued += 1;
                }
            }
        }

        None
    }

    pub fn run_trace<T: AddressTrace>(&mut self, trace: &mut T) {
        for (addr, _kind) in trace {
            self.step(addr);
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
        }
    }

    pub fn reset(&mut self) {
        self.l1.reset_stats();
        if let Some(l2) = &mut self.l2 {
            l2.reset_stats();
        }
        if let Some(l3) = &mut self.l3 {
            l3.reset_stats();
        }
        self.stats = HierarchyStats::default();
    }
}

// Ready-made CPU profiles
pub mod profiles {
    use super::*;

    pub fn zen3() -> MemoryHierarchy {
        let l1 = CacheLevel::new(CacheConfig {
            level_name: "L1D",
            size_bytes: 32 * 1024,
            line_bytes: 64,
            associativity: 8,
            policy: ReplacementPolicy::Lru,
        });
        let l2 = CacheLevel::new(CacheConfig {
            level_name: "L2",
            size_bytes: 512 * 1024,
            line_bytes: 64,
            associativity: 8,
            policy: ReplacementPolicy::Lru,
        });
        let l3 = CacheLevel::new(CacheConfig {
            level_name: "L3",
            size_bytes: 16 * 1024 * 1024,
            line_bytes: 64,
            associativity: 16,
            policy: ReplacementPolicy::Lru,
        });
        MemoryHierarchy::new(l1, Some(l2), Some(l3))
    }

    pub fn m1() -> MemoryHierarchy {
        let l1 = CacheLevel::new(CacheConfig {
            level_name: "L1D",
            size_bytes: 64 * 1024,
            line_bytes: 64,
            associativity: 8,
            policy: ReplacementPolicy::Lru,
        });
        let l2 = CacheLevel::new(CacheConfig {
            level_name: "L2",
            size_bytes: 12 * 1024 * 1024,
            line_bytes: 64,
            associativity: 12,
            policy: ReplacementPolicy::Lru,
        });
        MemoryHierarchy::new(l1, Some(l2), None)
    }
}
