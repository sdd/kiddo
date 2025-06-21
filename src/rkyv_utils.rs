use std::mem::{align_of, size_of};

/// Just a wrapper around `std::mem::transmute`. Used to map rkyv Archived primitive types back to
/// their un-archived versions. For primitive types such as f32 / f64 / u32 / u64, or arrays of
/// them, this should be fine assuming that you are not compiling for a target architecture with a
/// different endianness from the archived data. Where this is used in a non-rkyv context, T will
/// equal U and so this invocation of std::mem::transmute will have no effect at all.
pub(crate) fn transform<T, U>(item: &U) -> &T {
    debug_assert_eq!(
        size_of::<T>(),
        size_of::<U>(),
        "size of {} does not match size of {}",
        std::any::type_name::<T>(),
        std::any::type_name::<U>()
    );

    debug_assert!(
        align_of::<T>() <= align_of::<U>(),
        "alignment of {} ({}) is greater than alignment of {} ({})",
        std::any::type_name::<T>(),
        align_of::<T>(),
        std::any::type_name::<U>(),
        align_of::<U>()
    );

    unsafe { std::mem::transmute::<&U, &T>(item) }
}
