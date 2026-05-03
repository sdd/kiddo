use std::mem::{align_of, size_of};

/// Just a wrapper around `std::mem::transmute`. Used to map rkyv Archived primitive types back to
/// their un-archived versions. For primitive types such as f32 / f64 / u32 / u64, or arrays of
/// them, this should be fine assuming that you are not compiling for a target architecture with a
/// different endianness from the archived data. Where this is used in a non-rkyv context, T will
/// equal U and so this invocation of std::mem::transmute will have no effect at all.
#[allow(unused)]
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

/// Converts a slice of one type (`&[U]`) into a slice of another type (`&[T]`) without
/// performing any data transformation but rather reinterpreting the raw memory.
///
/// Used to reinterpret slices of rkyv-Archived primitive types into their original form, where valid to do so.
///
/// # Type Parameters
/// - `T`: Target type to convert to.
/// - `U`: Source type to convert from.
///
/// # Parameters
/// - `items`: A reference to a slice of type `U` which will be reinterpreted as a slice of type `T`.
///
/// # Returns
/// A slice of type `T` with the same length and memory layout as the input slice `&[U]`.
///
/// # Panics
/// This function will panic in debug mode if:
/// - The size of `T` does not match the size of `U`.
/// - The alignment of `T` is greater than the alignment of `U`.
///
/// # Safety
/// This function is marked as `unsafe` because:
/// - It performs a reinterpretation of the raw memory of the input slice.
/// - You must ensure that the memory layout of `U` is compatible with `T` to avoid undefined behavior.
/// - Misusing this function with incompatible types can lead to data corruption, undefined behavior, or program crashes.
///
/// # Examples
/// ```ignore
/// // Transforming a &[Archived<u32>] slice back into a &[u32]-compatible view
/// let bytes: &[u8] = &[0x12, 0x34, 0x56, 0x78];
/// let words: &[u32] = transform_slice(bytes);
///
/// assert_eq!(words.len(), 1);
/// assert_eq!(words[0], 0x78563412);
/// ```
///
/// Be cautious while using this function, as improper usage may result in undefined behavior.
pub(crate) fn transform_slice<T, U>(items: &[U]) -> &[T] {
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

    unsafe { std::slice::from_raw_parts(items.as_ptr() as *const T, items.len()) }
}
