//! Best-effort Transparent Huge Page helpers for owned and archived tree storage.

use std::fmt;

#[cfg(all(feature = "huge_pages", target_os = "linux"))]
use std::ffi::c_void;
#[cfg(all(feature = "huge_pages", target_os = "linux"))]
use std::sync::OnceLock;

#[cfg(all(feature = "huge_pages", target_os = "linux"))]
const THP_CANDIDATE_MIN_BYTES: usize = 2 * 1024 * 1024;

/// The huge-page advice operation to apply to a byte range.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HugePageMode {
    /// Prevent the range from being backed by Transparent Huge Pages.
    NoHuge,
    /// Mark the range as Transparent Huge Page friendly.
    Advise,
    /// Attempt synchronous THP collapse, falling back to `MADV_HUGEPAGE` on failure.
    Collapse,
}

/// Result of applying huge-page advice to a range.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HugePageReport {
    /// Requested input byte length.
    pub requested_bytes: usize,
    /// Page-aligned address passed to `madvise`, or zero when unsupported/not applicable.
    pub aligned_addr: usize,
    /// Page-aligned byte length passed to `madvise`.
    pub aligned_bytes: usize,
    /// Operating-system base page size used for alignment.
    pub page_size: usize,
    /// Operation requested by the caller.
    pub mode: HugePageMode,
    /// Whether a `MADV_COLLAPSE` call was attempted.
    pub collapse_attempted: bool,
    /// Whether `MADV_COLLAPSE` succeeded.
    pub collapse_succeeded: bool,
    /// Whether a `MADV_HUGEPAGE` call was attempted.
    pub hugepage_attempted: bool,
    /// Whether `MADV_HUGEPAGE` succeeded.
    pub hugepage_succeeded: bool,
    /// Whether a `MADV_NOHUGEPAGE` call was attempted.
    pub nohuge_attempted: bool,
    /// Whether `MADV_NOHUGEPAGE` succeeded.
    pub nohuge_succeeded: bool,
}

impl HugePageReport {
    #[cfg(all(feature = "huge_pages", target_os = "linux"))]
    fn skipped(requested_bytes: usize, mode: HugePageMode) -> Self {
        Self {
            requested_bytes,
            aligned_addr: 0,
            aligned_bytes: 0,
            page_size: 0,
            mode,
            collapse_attempted: false,
            collapse_succeeded: false,
            hugepage_attempted: false,
            hugepage_succeeded: false,
            nohuge_attempted: false,
            nohuge_succeeded: false,
        }
    }
}

/// `/proc/self/smaps` summary for a byte range.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct HugePageMappingReport {
    /// Number of virtual memory areas overlapping the requested range.
    pub vma_count: usize,
    /// Total overlapped mapping size reported by `smaps`, in KiB.
    pub size_kb: usize,
    /// Anonymous huge pages reported by `smaps`, in KiB.
    pub anon_huge_pages_kb: usize,
    /// File-backed PMD mappings reported by `smaps`, in KiB.
    pub file_pmd_mapped_kb: usize,
    /// Shmem PMD mappings reported by `smaps`, in KiB.
    pub shmem_pmd_mapped_kb: usize,
    /// Kernel page sizes seen in overlapping VMAs, in KiB.
    pub kernel_page_size_kb: Vec<usize>,
    /// MMU page sizes seen in overlapping VMAs, in KiB.
    pub mmu_page_size_kb: Vec<usize>,
}

impl HugePageMappingReport {
    /// Returns the total huge-page-backed mapping amount reported by `smaps`, in KiB.
    pub fn total_huge_kb(&self) -> usize {
        self.anon_huge_pages_kb + self.file_pmd_mapped_kb + self.shmem_pmd_mapped_kb
    }
}

/// Error returned by huge-page advice and diagnostics.
#[derive(Debug)]
pub enum HugePageError {
    /// Huge-page advice is not available for this target or build.
    Unsupported,
    /// The operating system page size could not be determined.
    PageSizeUnavailable,
    /// The requested range length overflowed.
    RangeOverflow,
    /// `madvise` failed.
    Madvise {
        /// Advice operation that failed.
        mode: HugePageMode,
        /// Source OS error.
        source: std::io::Error,
    },
    /// Reading `/proc/self/smaps` failed.
    Smaps(std::io::Error),
}

impl fmt::Display for HugePageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported => f.write_str("huge-page advice is unsupported for this build"),
            Self::PageSizeUnavailable => f.write_str("could not determine OS page size"),
            Self::RangeOverflow => f.write_str("huge-page range length overflowed"),
            Self::Madvise { mode, source } => write!(f, "madvise({mode:?}) failed: {source}"),
            Self::Smaps(source) => write!(f, "could not read /proc/self/smaps: {source}"),
        }
    }
}

impl std::error::Error for HugePageError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Madvise { source, .. } | Self::Smaps(source) => Some(source),
            Self::Unsupported | Self::PageSizeUnavailable | Self::RangeOverflow => None,
        }
    }
}

#[cfg(all(feature = "huge_pages", target_os = "linux"))]
#[derive(Clone, Copy, Debug)]
struct HugePageRange {
    addr: *mut c_void,
    addr_usize: usize,
    len: usize,
    page_size: usize,
    requested_bytes: usize,
}

#[cfg(all(feature = "huge_pages", target_os = "linux"))]
#[inline]
fn os_page_size() -> Option<usize> {
    use libc::sysconf;
    use libc::_SC_PAGESIZE;

    static PAGE_SIZE: OnceLock<Option<usize>> = OnceLock::new();

    *PAGE_SIZE.get_or_init(|| {
        let raw = unsafe { sysconf(_SC_PAGESIZE) };
        usize::try_from(raw).ok().filter(|size| *size != 0)
    })
}

#[cfg(all(feature = "huge_pages", target_os = "linux"))]
#[inline]
fn huge_page_range_bytes(
    ptr: *const u8,
    len: usize,
) -> Result<Option<HugePageRange>, HugePageError> {
    if len == 0 || ptr.is_null() {
        return Ok(None);
    }

    if len < THP_CANDIDATE_MIN_BYTES {
        return Ok(None);
    }

    let page_size = os_page_size().ok_or(HugePageError::PageSizeUnavailable)?;
    let start = ptr as usize;
    let end = start.checked_add(len).ok_or(HugePageError::RangeOverflow)?;
    let aligned_start = start / page_size * page_size;
    let aligned_end = end
        .checked_add(page_size - 1)
        .ok_or(HugePageError::RangeOverflow)?
        / page_size
        * page_size;
    let aligned_len = aligned_end.saturating_sub(aligned_start);

    if aligned_len < THP_CANDIDATE_MIN_BYTES {
        return Ok(None);
    }

    Ok(Some(HugePageRange {
        addr: aligned_start as *mut c_void,
        addr_usize: aligned_start,
        len: aligned_len,
        page_size,
        requested_bytes: len,
    }))
}

#[cfg(all(feature = "huge_pages", target_os = "linux"))]
#[inline]
fn huge_page_range<T>(ptr: *const T, len: usize) -> Result<Option<HugePageRange>, HugePageError> {
    if len == 0 || std::mem::size_of::<T>() == 0 || ptr.is_null() {
        return Ok(None);
    }

    let byte_len = std::mem::size_of::<T>()
        .checked_mul(len)
        .ok_or(HugePageError::RangeOverflow)?;
    huge_page_range_bytes(ptr.cast::<u8>(), byte_len)
}

#[cfg(all(feature = "huge_pages", target_os = "linux"))]
#[inline]
fn madvise_range(range: HugePageRange, mode: HugePageMode) -> Result<(), HugePageError> {
    let advice = match mode {
        HugePageMode::NoHuge => libc::MADV_NOHUGEPAGE,
        HugePageMode::Advise => libc::MADV_HUGEPAGE,
        HugePageMode::Collapse => libc::MADV_COLLAPSE,
    };

    let rc = unsafe { libc::madvise(range.addr, range.len, advice) };
    if rc == 0 {
        Ok(())
    } else {
        Err(HugePageError::Madvise {
            mode,
            source: std::io::Error::last_os_error(),
        })
    }
}

#[cfg(all(feature = "huge_pages", target_os = "linux"))]
#[inline]
fn apply_huge_pages(
    ptr: *const u8,
    len: usize,
    mode: HugePageMode,
) -> Result<HugePageReport, HugePageError> {
    let Some(range) = huge_page_range_bytes(ptr, len)? else {
        return Ok(HugePageReport::skipped(len, mode));
    };

    let mut report = HugePageReport {
        requested_bytes: range.requested_bytes,
        aligned_addr: range.addr_usize,
        aligned_bytes: range.len,
        page_size: range.page_size,
        mode,
        collapse_attempted: false,
        collapse_succeeded: false,
        hugepage_attempted: false,
        hugepage_succeeded: false,
        nohuge_attempted: false,
        nohuge_succeeded: false,
    };

    match mode {
        HugePageMode::NoHuge => {
            report.nohuge_attempted = true;
            madvise_range(range, HugePageMode::NoHuge)?;
            report.nohuge_succeeded = true;
        }
        HugePageMode::Advise => {
            report.hugepage_attempted = true;
            madvise_range(range, HugePageMode::Advise)?;
            report.hugepage_succeeded = true;
        }
        HugePageMode::Collapse => {
            report.collapse_attempted = true;
            match madvise_range(range, HugePageMode::Collapse) {
                Ok(()) => {
                    report.collapse_succeeded = true;
                }
                Err(_) => {
                    report.hugepage_attempted = true;
                    madvise_range(range, HugePageMode::Advise)?;
                    report.hugepage_succeeded = true;
                }
            }
        }
    }

    Ok(report)
}

/// Prevent an archived byte range from being backed by Transparent Huge Pages.
///
/// This is primarily useful for establishing a comparative baseline on systems that promote
/// large file mappings without an explicit application hint.
pub fn no_huge_pages(bytes: &[u8]) -> Result<HugePageReport, HugePageError> {
    #[cfg(all(feature = "huge_pages", target_os = "linux"))]
    {
        apply_huge_pages(bytes.as_ptr(), bytes.len(), HugePageMode::NoHuge)
    }

    #[cfg(not(all(feature = "huge_pages", target_os = "linux")))]
    {
        let _ = bytes;
        Err(HugePageError::Unsupported)
    }
}

/// Mark an archived byte range as Transparent Huge Page friendly.
///
/// This is intended for already-loaded rkyv archive bytes, including bytes backed by a file
/// mapping. On Linux this calls `MADV_HUGEPAGE`.
pub fn advise_huge_pages(bytes: &[u8]) -> Result<HugePageReport, HugePageError> {
    #[cfg(all(feature = "huge_pages", target_os = "linux"))]
    {
        apply_huge_pages(bytes.as_ptr(), bytes.len(), HugePageMode::Advise)
    }

    #[cfg(not(all(feature = "huge_pages", target_os = "linux")))]
    {
        let _ = bytes;
        Err(HugePageError::Unsupported)
    }
}

/// Attempt to synchronously collapse an archived byte range into Transparent Huge Pages.
///
/// On Linux this first tries `MADV_COLLAPSE` and falls back to `MADV_HUGEPAGE`.
pub fn collapse_huge_pages(bytes: &[u8]) -> Result<HugePageReport, HugePageError> {
    #[cfg(all(feature = "huge_pages", target_os = "linux"))]
    {
        apply_huge_pages(bytes.as_ptr(), bytes.len(), HugePageMode::Collapse)
    }

    #[cfg(not(all(feature = "huge_pages", target_os = "linux")))]
    {
        let _ = bytes;
        Err(HugePageError::Unsupported)
    }
}

/// Apply the selected huge-page mode to an archived byte range.
pub fn prepare_archived_bytes(
    bytes: &[u8],
    mode: HugePageMode,
) -> Result<HugePageReport, HugePageError> {
    match mode {
        HugePageMode::NoHuge => no_huge_pages(bytes),
        HugePageMode::Advise => advise_huge_pages(bytes),
        HugePageMode::Collapse => collapse_huge_pages(bytes),
    }
}

/// Read `/proc/self/smaps` and summarize huge-page backing for a byte range.
#[cfg(target_os = "linux")]
pub fn mapping_report_for_slice(bytes: &[u8]) -> Result<HugePageMappingReport, HugePageError> {
    if bytes.is_empty() {
        return Ok(HugePageMappingReport::default());
    }

    let start = bytes.as_ptr() as usize;
    let end = start
        .checked_add(bytes.len())
        .ok_or(HugePageError::RangeOverflow)?;
    let smaps = std::fs::read_to_string("/proc/self/smaps").map_err(HugePageError::Smaps)?;
    Ok(parse_smaps_for_range(&smaps, start, end))
}

/// Read `/proc/self/smaps` and summarize huge-page backing for a byte range.
#[cfg(not(target_os = "linux"))]
pub fn mapping_report_for_slice(_bytes: &[u8]) -> Result<HugePageMappingReport, HugePageError> {
    Err(HugePageError::Unsupported)
}

#[cfg(target_os = "linux")]
fn parse_smaps_for_range(
    smaps: &str,
    target_start: usize,
    target_end: usize,
) -> HugePageMappingReport {
    let mut report = HugePageMappingReport::default();
    let mut current: Option<VmaSmaps> = None;

    for line in smaps.lines() {
        if let Some((start, end)) = parse_smaps_header(line) {
            if let Some(vma) = current.take() {
                add_vma_if_overlaps(&mut report, vma, target_start, target_end);
            }
            current = Some(VmaSmaps::new(start, end));
            continue;
        }

        let Some(vma) = current.as_mut() else {
            continue;
        };

        if let Some(value) = parse_kb_field(line, "Size:") {
            vma.size_kb = value;
        } else if let Some(value) = parse_kb_field(line, "AnonHugePages:") {
            vma.anon_huge_pages_kb = value;
        } else if let Some(value) = parse_kb_field(line, "FilePmdMapped:") {
            vma.file_pmd_mapped_kb = value;
        } else if let Some(value) = parse_kb_field(line, "ShmemPmdMapped:") {
            vma.shmem_pmd_mapped_kb = value;
        } else if let Some(value) = parse_kb_field(line, "KernelPageSize:") {
            vma.kernel_page_size_kb = Some(value);
        } else if let Some(value) = parse_kb_field(line, "MMUPageSize:") {
            vma.mmu_page_size_kb = Some(value);
        }
    }

    if let Some(vma) = current {
        add_vma_if_overlaps(&mut report, vma, target_start, target_end);
    }

    report
}

#[cfg(target_os = "linux")]
#[derive(Clone, Debug)]
struct VmaSmaps {
    start: usize,
    end: usize,
    size_kb: usize,
    anon_huge_pages_kb: usize,
    file_pmd_mapped_kb: usize,
    shmem_pmd_mapped_kb: usize,
    kernel_page_size_kb: Option<usize>,
    mmu_page_size_kb: Option<usize>,
}

#[cfg(target_os = "linux")]
impl VmaSmaps {
    fn new(start: usize, end: usize) -> Self {
        Self {
            start,
            end,
            size_kb: 0,
            anon_huge_pages_kb: 0,
            file_pmd_mapped_kb: 0,
            shmem_pmd_mapped_kb: 0,
            kernel_page_size_kb: None,
            mmu_page_size_kb: None,
        }
    }
}

#[cfg(target_os = "linux")]
fn add_vma_if_overlaps(
    report: &mut HugePageMappingReport,
    vma: VmaSmaps,
    target_start: usize,
    target_end: usize,
) {
    if vma.end <= target_start || vma.start >= target_end {
        return;
    }

    report.vma_count += 1;
    report.size_kb += vma.size_kb;
    report.anon_huge_pages_kb += vma.anon_huge_pages_kb;
    report.file_pmd_mapped_kb += vma.file_pmd_mapped_kb;
    report.shmem_pmd_mapped_kb += vma.shmem_pmd_mapped_kb;

    if let Some(value) = vma.kernel_page_size_kb {
        push_unique(&mut report.kernel_page_size_kb, value);
    }
    if let Some(value) = vma.mmu_page_size_kb {
        push_unique(&mut report.mmu_page_size_kb, value);
    }
}

#[cfg(target_os = "linux")]
fn push_unique(values: &mut Vec<usize>, value: usize) {
    if !values.contains(&value) {
        values.push(value);
    }
}

#[cfg(target_os = "linux")]
fn parse_smaps_header(line: &str) -> Option<(usize, usize)> {
    let first = line.split_ascii_whitespace().next()?;
    let (start, end) = first.split_once('-')?;
    let start = usize::from_str_radix(start, 16).ok()?;
    let end = usize::from_str_radix(end, 16).ok()?;
    Some((start, end))
}

#[cfg(target_os = "linux")]
fn parse_kb_field(line: &str, key: &str) -> Option<usize> {
    let rest = line.strip_prefix(key)?;
    rest.split_ascii_whitespace().next()?.parse().ok()
}

/// Best-effort hint that a large contiguous allocation would benefit from THP.
///
/// This is used internally on mutating/growing allocations and intentionally ignores failures.
#[inline]
pub(crate) fn maybe_advise_slice_huge_pages<T>(ptr: *const T, len: usize) {
    #[cfg(all(feature = "huge_pages", target_os = "linux"))]
    if let Ok(Some(range)) = huge_page_range(ptr, len) {
        let _ = madvise_range(range, HugePageMode::Advise);
    }

    #[cfg(not(all(feature = "huge_pages", target_os = "linux")))]
    let _ = (ptr, len);
}

/// Best-effort attempt to synchronously collapse a large allocation into THPs.
///
/// This is used internally for long-lived buffers after construction and intentionally ignores
/// failures so construction behavior stays unchanged.
#[inline]
pub(crate) fn maybe_collapse_slice_huge_pages<T>(ptr: *const T, len: usize) {
    #[cfg(all(feature = "huge_pages", target_os = "linux"))]
    if let Ok(Some(range)) = huge_page_range(ptr, len) {
        if madvise_range(range, HugePageMode::Collapse).is_err() {
            let _ = madvise_range(range, HugePageMode::Advise);
        }
    }

    #[cfg(not(all(feature = "huge_pages", target_os = "linux")))]
    let _ = (ptr, len);
}

#[cfg(test)]
mod tests {
    use super::{HugePageError, HugePageMappingReport};
    use std::error::Error;

    #[cfg(not(all(feature = "huge_pages", target_os = "linux")))]
    use super::{
        advise_huge_pages, collapse_huge_pages, no_huge_pages, prepare_archived_bytes, HugePageMode,
    };
    #[cfg(target_os = "linux")]
    use super::{mapping_report_for_slice, parse_smaps_for_range};
    #[cfg(target_os = "linux")]
    use super::{parse_kb_field, parse_smaps_header, push_unique};

    #[test]
    fn huge_page_mapping_report_total_huge_kb_sums_components() {
        let report = HugePageMappingReport {
            anon_huge_pages_kb: 1,
            file_pmd_mapped_kb: 2,
            shmem_pmd_mapped_kb: 3,
            ..HugePageMappingReport::default()
        };
        assert_eq!(report.total_huge_kb(), 6);
    }

    #[test]
    fn huge_page_error_display_and_source_are_wired() {
        let err = HugePageError::Unsupported;
        assert_eq!(
            err.to_string(),
            "huge-page advice is unsupported for this build"
        );
        assert!(err.source().is_none());

        let io = std::io::Error::other("boom");
        let err = HugePageError::Smaps(io);
        assert!(err.to_string().contains("could not read /proc/self/smaps"));
        assert!(err.source().is_some());
    }

    #[test]
    fn huge_page_public_helpers_cover_non_feature_paths() {
        #[cfg(not(all(feature = "huge_pages", target_os = "linux")))]
        {
            let bytes = [0u8; 16];

            assert!(matches!(
                no_huge_pages(&bytes),
                Err(HugePageError::Unsupported)
            ));
            assert!(matches!(
                advise_huge_pages(&bytes),
                Err(HugePageError::Unsupported)
            ));
            assert!(matches!(
                collapse_huge_pages(&bytes),
                Err(HugePageError::Unsupported)
            ));
            assert!(matches!(
                prepare_archived_bytes(&bytes, HugePageMode::NoHuge),
                Err(HugePageError::Unsupported)
            ));
            assert!(matches!(
                prepare_archived_bytes(&bytes, HugePageMode::Advise),
                Err(HugePageError::Unsupported)
            ));
            assert!(matches!(
                prepare_archived_bytes(&bytes, HugePageMode::Collapse),
                Err(HugePageError::Unsupported)
            ));
        }

        #[cfg(target_os = "linux")]
        {
            assert_eq!(
                mapping_report_for_slice(&[]).unwrap(),
                HugePageMappingReport::default()
            );
        }
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn smaps_parser_sums_overlapping_vmas() {
        let smaps = "\
1000-2000 r--p 00000000 00:00 0
Size:                  4 kB
KernelPageSize:        4 kB
MMUPageSize:           4 kB
AnonHugePages:         0 kB
FilePmdMapped:         0 kB
ShmemPmdMapped:        0 kB
2000-6000 r--p 00000000 00:00 0
Size:                 16 kB
KernelPageSize:     2048 kB
MMUPageSize:        2048 kB
AnonHugePages:      2048 kB
FilePmdMapped:         0 kB
ShmemPmdMapped:        0 kB
";

        let report = parse_smaps_for_range(smaps, 0x1800, 0x3000);

        assert_eq!(
            report,
            HugePageMappingReport {
                vma_count: 2,
                size_kb: 20,
                anon_huge_pages_kb: 2048,
                file_pmd_mapped_kb: 0,
                shmem_pmd_mapped_kb: 0,
                kernel_page_size_kb: vec![4, 2048],
                mmu_page_size_kb: vec![4, 2048],
            }
        );
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn smaps_parser_helpers_parse_and_deduplicate() {
        assert_eq!(
            parse_smaps_header("1000-2000 r--p 00000000 00:00 0"),
            Some((0x1000, 0x2000))
        );
        assert_eq!(parse_smaps_header("not-a-header"), None);

        assert_eq!(
            parse_kb_field("Size:                 16 kB", "Size:"),
            Some(16)
        );
        assert_eq!(parse_kb_field("AnonHugePages:      2048 kB", "Size:"), None);

        let mut values = vec![4, 2048];
        push_unique(&mut values, 2048);
        push_unique(&mut values, 64);
        assert_eq!(values, vec![4, 2048, 64]);
    }
}
