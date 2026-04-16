//! Best-effort transparent huge page hints for large contiguous allocations.

#[cfg(all(feature = "huge_pages", target_os = "linux"))]
use std::ffi::c_void;
#[cfg(all(feature = "huge_pages", target_os = "linux"))]
use std::sync::OnceLock;

#[cfg(all(feature = "huge_pages", target_os = "linux"))]
const THP_CANDIDATE_MIN_BYTES: usize = 2 * 1024 * 1024;

#[cfg(all(feature = "huge_pages", target_os = "linux"))]
#[inline]
fn huge_page_range<T>(ptr: *const T, len: usize) -> Option<(*mut c_void, usize)> {
    use libc::sysconf;
    use libc::_SC_PAGESIZE;

    static PAGE_SIZE: OnceLock<Option<usize>> = OnceLock::new();

    if len == 0 || std::mem::size_of::<T>() == 0 || ptr.is_null() {
        return None;
    }

    let byte_len = std::mem::size_of::<T>().checked_mul(len)?;
    if byte_len < THP_CANDIDATE_MIN_BYTES {
        return None;
    }

    let page_size = (*PAGE_SIZE.get_or_init(|| {
        let raw = unsafe { sysconf(_SC_PAGESIZE) };
        usize::try_from(raw).ok().filter(|size| *size != 0)
    }))?;

    let start = ptr.cast::<u8>() as usize;
    let end = start.checked_add(byte_len)?;
    let aligned_start = start / page_size * page_size;
    let aligned_end = end
        .checked_add(page_size - 1)
        .map(|value| value / page_size * page_size)?;
    let aligned_len = aligned_end.saturating_sub(aligned_start);

    if aligned_len < THP_CANDIDATE_MIN_BYTES {
        return None;
    }

    Some((aligned_start as *mut c_void, aligned_len))
}

#[cfg(all(feature = "huge_pages", target_os = "linux"))]
#[inline]
fn madvise_hugepage(addr: *mut c_void, len: usize) {
    unsafe {
        let _ = libc::madvise(addr, len, libc::MADV_HUGEPAGE);
    }
}

/// Best-effort hint that a large contiguous allocation would benefit from THP.
///
/// This is the cheap advisory path. It marks the range as THP-friendly but does not
/// synchronously force collapse. Use this on mutating/growing allocations.
#[inline]
pub(crate) fn maybe_advise_slice_huge_pages<T>(ptr: *const T, len: usize) {
    #[cfg(all(feature = "huge_pages", target_os = "linux"))]
    if let Some((addr, aligned_len)) = huge_page_range(ptr, len) {
        madvise_hugepage(addr, aligned_len);
    }

    #[cfg(not(all(feature = "huge_pages", target_os = "linux")))]
    let _ = (ptr, len);
}

/// Best-effort attempt to synchronously collapse a large allocation into THPs.
///
/// On Linux, this first tries `MADV_COLLAPSE` and falls back to `MADV_HUGEPAGE` if the
/// kernel cannot collapse the region immediately. This is intended for long-lived buffers
/// after construction, not for hot mutating growth paths.
#[inline]
pub(crate) fn maybe_collapse_slice_huge_pages<T>(ptr: *const T, len: usize) {
    #[cfg(all(feature = "huge_pages", target_os = "linux"))]
    if let Some((addr, aligned_len)) = huge_page_range(ptr, len) {
        unsafe {
            if libc::madvise(addr, aligned_len, libc::MADV_COLLAPSE) != 0 {
                madvise_hugepage(addr, aligned_len);
            }
        }
    }

    #[cfg(not(all(feature = "huge_pages", target_os = "linux")))]
    let _ = (ptr, len);
}
