#![cfg_attr(coverage_nightly, feature(coverage_attribute))]
#![cfg_attr(coverage_nightly, coverage(off))]

#[cfg(target_os = "linux")]
mod linux {
    use kiddo::dist::SquaredEuclidean;
    use kiddo::huge_pages::{
        mapping_report_for_slice, prepare_archived_bytes, HugePageMappingReport, HugePageMode,
    };
    use kiddo::leaf_strategy::VecOfArenas;
    use kiddo::stem_strategy::donnelly_2_pf::DonnellyPf;
    use rkyv_08::util::AlignedVec;
    use std::error::Error;
    use std::fs::File;
    use std::hint::black_box;
    use std::os::fd::AsRawFd;
    use std::path::{Path, PathBuf};
    use std::time::{Duration, Instant};

    const K: usize = 3;
    const B: usize = 32;
    const DEFAULT_REPEATS: usize = 100;
    const DEFAULT_START_DELAY_MS: u64 = 0;

    type ArenaLeaves = VecOfArenas<f64, u32, K, B>;
    type ArchivedDonnellyPfTree =
        kiddo::kd_tree::ArchivedKdTree<f64, u32, DonnellyPf<3, 64, 8, K>, ArenaLeaves, K, B>;

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum LoadMode {
        Mmap,
        Owned,
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum HugeMode {
        Off,
        NoHuge,
        Advise,
        Collapse,
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum QueryKind {
        NearestOne,
        ApproxNearestOne,
    }

    struct MmapArchive {
        ptr: *mut libc::c_void,
        len: usize,
    }

    impl MmapArchive {
        fn new(path: &Path) -> Result<Self, Box<dyn Error>> {
            let file = File::open(path)?;
            let len = usize::try_from(file.metadata()?.len())?;
            if len == 0 {
                return Err(format!("cannot mmap empty archive: {}", path.display()).into());
            }

            let ptr = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    len,
                    libc::PROT_READ,
                    libc::MAP_PRIVATE,
                    file.as_raw_fd(),
                    0,
                )
            };
            if ptr == libc::MAP_FAILED {
                return Err(std::io::Error::last_os_error().into());
            }

            Ok(Self { ptr, len })
        }

        fn as_slice(&self) -> &[u8] {
            unsafe { std::slice::from_raw_parts(self.ptr.cast::<u8>(), self.len) }
        }
    }

    impl Drop for MmapArchive {
        fn drop(&mut self) {
            unsafe {
                let _ = libc::munmap(self.ptr, self.len);
            }
        }
    }

    enum ArchiveBytes {
        Mmap(MmapArchive),
        Owned(AlignedVec<128>),
    }

    impl ArchiveBytes {
        fn as_slice(&self) -> &[u8] {
            match self {
                Self::Mmap(bytes) => bytes.as_slice(),
                Self::Owned(bytes) => &bytes[..],
            }
        }
    }

    #[derive(Clone, Copy, Debug)]
    struct RunResult {
        elapsed_ns: f64,
        checksum_dist: f64,
        checksum_item: u64,
    }

    fn read_usize_env(var: &str, default: usize) -> usize {
        std::env::var(var)
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(default)
    }

    fn read_u64_env(var: &str, default: u64) -> u64 {
        std::env::var(var)
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(default)
    }

    fn read_bool_env(var: &str, default: bool) -> bool {
        std::env::var(var)
            .ok()
            .and_then(|value| match value.as_str() {
                "1" | "true" | "yes" | "on" => Some(true),
                "0" | "false" | "no" | "off" => Some(false),
                _ => None,
            })
            .unwrap_or(default)
    }

    fn archive_path(prefix: &Path, suffix: &str) -> PathBuf {
        PathBuf::from(format!("{}-{suffix}.rkyv", prefix.display()))
    }

    fn parse_load_mode() -> Result<LoadMode, Box<dyn Error>> {
        match std::env::var("KIDDO_PROFILE_LOAD_MODE")
            .unwrap_or_else(|_| "mmap".to_owned())
            .as_str()
        {
            "mmap" => Ok(LoadMode::Mmap),
            "owned" => Ok(LoadMode::Owned),
            value => Err(format!("unsupported KIDDO_PROFILE_LOAD_MODE={value}").into()),
        }
    }

    fn parse_huge_mode() -> Result<HugeMode, Box<dyn Error>> {
        match std::env::var("KIDDO_PROFILE_HUGE_PAGES")
            .unwrap_or_else(|_| "collapse".to_owned())
            .as_str()
        {
            "off" => Ok(HugeMode::Off),
            "nohuge" | "no-huge" => Ok(HugeMode::NoHuge),
            "advise" => Ok(HugeMode::Advise),
            "collapse" => Ok(HugeMode::Collapse),
            value => Err(format!("unsupported KIDDO_PROFILE_HUGE_PAGES={value}").into()),
        }
    }

    fn parse_query_kind() -> Result<QueryKind, Box<dyn Error>> {
        match std::env::var("KIDDO_PROFILE_QUERY_KIND")
            .unwrap_or_else(|_| "nearest-one".to_owned())
            .as_str()
        {
            "nearest-one" => Ok(QueryKind::NearestOne),
            "approx-nearest-one" => Ok(QueryKind::ApproxNearestOne),
            value => Err(format!("unsupported KIDDO_PROFILE_QUERY_KIND={value}").into()),
        }
    }

    fn load_archive(path: &Path, mode: LoadMode) -> Result<ArchiveBytes, Box<dyn Error>> {
        match mode {
            LoadMode::Mmap => Ok(ArchiveBytes::Mmap(MmapArchive::new(path)?)),
            LoadMode::Owned => {
                let bytes = std::fs::read(path)?;
                let mut aligned = AlignedVec::<128>::with_capacity(bytes.len());
                aligned.extend_from_slice(&bytes);
                Ok(ArchiveBytes::Owned(aligned))
            }
        }
    }

    fn apply_huge_pages(bytes: &[u8], mode: HugeMode) -> Result<(), Box<dyn Error>> {
        match mode {
            HugeMode::Off => {
                eprintln!("huge_page_advice=off");
                Ok(())
            }
            HugeMode::NoHuge => {
                let report = prepare_archived_bytes(bytes, HugePageMode::NoHuge)?;
                eprintln!("huge_page_advice={report:?}");
                Ok(())
            }
            HugeMode::Advise => {
                let report = prepare_archived_bytes(bytes, HugePageMode::Advise)?;
                eprintln!("huge_page_advice={report:?}");
                Ok(())
            }
            HugeMode::Collapse => {
                let report = prepare_archived_bytes(bytes, HugePageMode::Collapse)?;
                eprintln!("huge_page_advice={report:?}");
                Ok(())
            }
        }
    }

    fn print_mapping_report(label: &str, report: &HugePageMappingReport) {
        eprintln!(
            "{label}: vmas={} size_kb={} anon_huge_kb={} file_pmd_kb={} shmem_pmd_kb={} total_huge_kb={} kernel_page_kb={:?} mmu_page_kb={:?}",
            report.vma_count,
            report.size_kb,
            report.anon_huge_pages_kb,
            report.file_pmd_mapped_kb,
            report.shmem_pmd_mapped_kb,
            report.total_huge_kb(),
            report.kernel_page_size_kb,
            report.mmu_page_size_kb
        );
    }

    fn os_page_size() -> usize {
        let raw = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        usize::try_from(raw)
            .ok()
            .filter(|value| *value != 0)
            .unwrap_or(4096)
    }

    fn prefault(bytes: &[u8]) -> u8 {
        let page_size = os_page_size();
        let mut checksum = 0u8;
        let mut offset = 0usize;

        while offset < bytes.len() {
            checksum ^= unsafe { std::ptr::read_volatile(bytes.as_ptr().add(offset)) };
            offset = offset.saturating_add(page_size);
        }

        if !bytes.is_empty() {
            checksum ^= unsafe { std::ptr::read_volatile(bytes.as_ptr().add(bytes.len() - 1)) };
        }

        checksum
    }

    fn run_nearest_one(
        tree: &ArchivedDonnellyPfTree,
        queries: &rkyv_08::vec::ArchivedVec<[f64; K]>,
        repeats: usize,
    ) -> RunResult {
        let start = Instant::now();
        let mut checksum_dist = 0.0f64;
        let mut checksum_item = 0u64;

        for _ in 0..repeats {
            for query in queries.iter() {
                let (dist, item) = tree.nearest_one::<SquaredEuclidean<f64>>(black_box(query));
                checksum_dist += dist;
                checksum_item = checksum_item.wrapping_add(item as u64);
            }
        }

        RunResult {
            elapsed_ns: start.elapsed().as_nanos() as f64,
            checksum_dist,
            checksum_item,
        }
    }

    fn run_approx_nearest_one(
        tree: &ArchivedDonnellyPfTree,
        queries: &rkyv_08::vec::ArchivedVec<[f64; K]>,
        repeats: usize,
    ) -> RunResult {
        let start = Instant::now();
        let mut checksum_dist = 0.0f64;
        let mut checksum_item = 0u64;

        for _ in 0..repeats {
            for query in queries.iter() {
                let (dist, item) =
                    tree.approx_nearest_one::<SquaredEuclidean<f64>>(black_box(query));
                checksum_dist += dist;
                checksum_item = checksum_item.wrapping_add(item as u64);
            }
        }

        RunResult {
            elapsed_ns: start.elapsed().as_nanos() as f64,
            checksum_dist,
            checksum_item,
        }
    }

    pub fn main() -> Result<(), Box<dyn Error>> {
        let prefix = PathBuf::from(
            std::env::var("KIDDO_PROFILE_ARCHIVE_PREFIX")
                .unwrap_or_else(|_| "./target/kiddo-hugepage-v6".to_owned()),
        );
        let tree_path = archive_path(&prefix, "donnelly-pf-tree");
        let queries_path = archive_path(&prefix, "queries");
        let load_mode = parse_load_mode()?;
        let huge_mode = parse_huge_mode()?;
        let query_kind = parse_query_kind()?;
        let repeats = read_usize_env("KIDDO_PROFILE_QUERY_BATCH_REPEATS", DEFAULT_REPEATS);
        let prefault_enabled = read_bool_env("KIDDO_PROFILE_PREFAULT", true);
        let start_delay_ms = read_u64_env("KIDDO_PROFILE_START_DELAY_MS", DEFAULT_START_DELAY_MS);

        eprintln!(
            "loading archives: tree={} queries={} load_mode={load_mode:?} huge_pages={huge_mode:?} query_kind={query_kind:?} repeats={repeats}",
            tree_path.display(),
            queries_path.display()
        );

        let load_start = Instant::now();
        let tree_bytes = load_archive(&tree_path, load_mode)?;
        let query_bytes = load_archive(&queries_path, load_mode)?;
        eprintln!(
            "loaded archive bytes in {:.0} ns: tree_bytes={} query_bytes={}",
            load_start.elapsed().as_nanos() as f64,
            tree_bytes.as_slice().len(),
            query_bytes.as_slice().len()
        );

        apply_huge_pages(tree_bytes.as_slice(), huge_mode)?;
        let tree_mapping = mapping_report_for_slice(tree_bytes.as_slice())?;
        print_mapping_report("tree_mapping_after_advice", &tree_mapping);

        if prefault_enabled {
            let start = Instant::now();
            let checksum = prefault(tree_bytes.as_slice()) ^ prefault(query_bytes.as_slice());
            eprintln!(
                "prefaulted archives in {:.0} ns checksum={checksum}",
                start.elapsed().as_nanos() as f64
            );
        }

        let tree = rkyv_08::access::<ArchivedDonnellyPfTree, rkyv_08::rancor::Error>(
            tree_bytes.as_slice(),
        )?;
        let queries = rkyv_08::access::<rkyv_08::vec::ArchivedVec<[f64; K]>, rkyv_08::rancor::Error>(
            query_bytes.as_slice(),
        )?;

        eprintln!(
            "zero_copy_access: tree_size={} leaf_count={} max_stem_level={} max_leaf_len={} queries={}",
            tree.size(),
            tree.leaf_count(),
            tree.max_stem_level(),
            tree.max_leaf_len(),
            queries.len()
        );

        if start_delay_ms != 0 {
            eprintln!("sleeping {start_delay_ms} ms before query loop");
            std::thread::sleep(Duration::from_millis(start_delay_ms));
        }

        let total_queries = queries.len() * repeats;
        let result = match query_kind {
            QueryKind::NearestOne => run_nearest_one(tree, queries, repeats),
            QueryKind::ApproxNearestOne => run_approx_nearest_one(tree, queries, repeats),
        };

        println!(
            "result query_kind={query_kind:?} total_queries={} ns_per_query={:.2} elapsed_ns={:.0} checksum_dist={:.17e} checksum_item={}",
            total_queries,
            result.elapsed_ns / total_queries as f64,
            result.elapsed_ns,
            result.checksum_dist,
            result.checksum_item
        );

        let tree_mapping = mapping_report_for_slice(tree_bytes.as_slice())?;
        print_mapping_report("tree_mapping_after_queries", &tree_mapping);

        Ok(())
    }
}

#[cfg(target_os = "linux")]
fn main() {
    if let Err(err) = linux::main() {
        eprintln!("archived huge-page profile failed: {err}");
        std::process::exit(1);
    }
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("profile_v6_archived_huge_pages is currently Linux-only");
    std::process::exit(1);
}
