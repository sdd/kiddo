#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub(crate) unsafe fn prefetch_t0(ptr: *const u8) {
    use core::arch::aarch64::{_prefetch, _PREFETCH_LOCALITY3, _PREFETCH_READ};
    _prefetch::<_PREFETCH_READ, _PREFETCH_LOCALITY3>(ptr as *const i8);
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub(crate) unsafe fn prefetch_t1(ptr: *const u8) {
    use core::arch::aarch64::{_prefetch, _PREFETCH_LOCALITY2, _PREFETCH_READ};
    _prefetch::<_PREFETCH_READ, _PREFETCH_LOCALITY2>(ptr as *const i8);
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub(crate) unsafe fn prefetch_t0(ptr: *const u8) {
    use core::arch::x86_64::_mm_prefetch;
    _mm_prefetch::<_MM_HINT_T0>(ptr as *const i8);
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub(crate) unsafe fn prefetch_t1(ptr: *const u8) {
    use core::arch::x86_64::_mm_prefetch;
    _mm_prefetch::<_MM_HINT_T1>(ptr as *const i8);
}
