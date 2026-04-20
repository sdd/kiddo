#![cfg(feature = "rkyv_08")]

use core::alloc::{Layout, LayoutError};

use aligned_vec::{AVec, CACHELINE_ALIGN};
use bytecheck::CheckBytes;
use nonmax::NonMaxUsize;
use ptr_meta::Pointee;
use rkyv_08::boxed::{ArchivedBox, BoxResolver};
use rkyv_08::primitive::ArchivedUsize;
use rkyv_08::rancor::{Fallible, Trace};
use rkyv_08::ser::{Allocator, Writer, WriterExt};
use rkyv_08::traits::{ArchivePointee, ArchiveUnsized, LayoutRaw};
use rkyv_08::with::{ArchiveWith, DeserializeWith, SerializeWith};
use rkyv_08::{
    Archive, Archived, ArchivedMetadata, Deserialize, Place, Serialize, SerializeUnsized,
};

/// Archived backing storage for cacheline-sensitive stem and arena buffers.
///
/// This is intentionally over-aligned so the archived slice payload itself begins on a cacheline
/// boundary, preserving the performance invariant expected by Donnelly-style stem layouts.
#[derive(rkyv_08::Portable)]
#[rkyv(crate = rkyv_08)]
#[repr(C, align(128))]
pub struct AlignedArchiveSlice<T: ?Sized> {
    tail: T,
}

impl<T: ?Sized> AlignedArchiveSlice<T> {
    #[inline(always)]
    pub fn tail(&self) -> &T {
        &self.tail
    }
}

impl<T> AlignedArchiveSlice<[T]> {
    #[inline(always)]
    pub fn as_slice(&self) -> &[T] {
        &self.tail
    }
}

unsafe impl<T> Pointee for AlignedArchiveSlice<[T]> {
    type Metadata = <[T] as Pointee>::Metadata;
}

impl<T> ArchivePointee for AlignedArchiveSlice<[T]> {
    type ArchivedMetadata = <[T] as ArchivePointee>::ArchivedMetadata;

    #[inline(always)]
    fn pointer_metadata(metadata: &Self::ArchivedMetadata) -> <Self as Pointee>::Metadata {
        <[T] as ArchivePointee>::pointer_metadata(metadata)
    }
}

impl<T> LayoutRaw for AlignedArchiveSlice<[T]> {
    #[inline]
    fn layout_raw(metadata: <Self as Pointee>::Metadata) -> Result<Layout, LayoutError> {
        Layout::array::<T>(metadata)?.align_to(CACHELINE_ALIGN)
    }
}

impl<T: Archive> ArchiveUnsized for AlignedArchiveSlice<[T]> {
    type Archived = AlignedArchiveSlice<[Archived<T>]>;

    #[inline(always)]
    fn archived_metadata(&self) -> ArchivedMetadata<Self> {
        self.tail.archived_metadata()
    }
}

// SAFETY: `AlignedArchiveSlice<[T]>` is a repr(C) wrapper with a single trailing slice field at
// offset 0. Validating the wrapped slice validates the entire archived payload.
unsafe impl<T, C> CheckBytes<C> for AlignedArchiveSlice<[T]>
where
    [T]: CheckBytes<C>,
    C: Fallible + ?Sized,
    C::Error: Trace,
{
    #[inline]
    unsafe fn check_bytes(value: *const Self, context: &mut C) -> Result<(), C::Error> {
        let (data_address, len) = ptr_meta::to_raw_parts(value);
        let slice_ptr = ptr_meta::from_raw_parts::<[T]>(data_address, len);
        <[T]>::check_bytes(slice_ptr, context)
    }
}

impl<T, S> SerializeUnsized<S> for AlignedArchiveSlice<[T]>
where
    T: Serialize<S>,
    S: Fallible + Writer + ?Sized,
{
    fn serialize_unsized(&self, serializer: &mut S) -> Result<usize, S::Error> {
        let mut resolvers = Vec::with_capacity(self.tail.len());
        for value in self.tail.iter() {
            resolvers.push(value.serialize(serializer)?);
        }

        let result = align_writer_to(serializer, CACHELINE_ALIGN)?;

        serializer.align_for::<Archived<T>>()?;
        for (value, resolver) in self.tail.iter().zip(resolvers.into_iter()) {
            unsafe {
                serializer.resolve_aligned(value, resolver)?;
            }
        }

        Ok(result)
    }
}

/// Archives an [`AVec`] into an aligned boxed slice whose payload begins on a cacheline boundary.
pub struct AsAlignedCachelineABox;

impl<T: Archive> ArchiveWith<AVec<T>> for AsAlignedCachelineABox {
    type Archived = ArchivedBox<AlignedArchiveSlice<[Archived<T>]>>;
    type Resolver = BoxResolver;

    #[inline(always)]
    fn resolve_with(field: &AVec<T>, resolver: Self::Resolver, out: Place<Self::Archived>) {
        ArchivedBox::resolve_from_raw_parts(resolver, field.as_slice().archived_metadata(), out);
    }
}

impl<T, S> SerializeWith<AVec<T>, S> for AsAlignedCachelineABox
where
    T: Serialize<S> + Archive,
    S: Fallible + Allocator + Writer + ?Sized,
{
    #[inline(always)]
    fn serialize_with(field: &AVec<T>, serializer: &mut S) -> Result<Self::Resolver, S::Error> {
        let aligned = unsafe {
            &*ptr_meta::from_raw_parts::<AlignedArchiveSlice<[T]>>(
                field.as_ptr().cast(),
                field.len(),
            )
        };

        ArchivedBox::serialize_from_ref(aligned, serializer)
    }
}

impl<T, D> DeserializeWith<ArchivedBox<AlignedArchiveSlice<[Archived<T>]>>, AVec<T>, D>
    for AsAlignedCachelineABox
where
    T: Archive,
    Archived<T>: Deserialize<T, D>,
    D: Fallible + ?Sized,
{
    #[inline]
    fn deserialize_with(
        field: &ArchivedBox<AlignedArchiveSlice<[Archived<T>]>>,
        deserializer: &mut D,
    ) -> Result<AVec<T>, D::Error> {
        let slice = field.get().as_slice();
        let mut result = AVec::with_capacity(CACHELINE_ALIGN, slice.len());
        for value in slice.iter() {
            result.push(value.deserialize(deserializer)?);
        }
        Ok(result)
    }
}

/// Archives `Option<NonMaxUsize>` as a plain archived usize using `usize::MAX` as the sentinel.
pub struct OptionNonMaxUsizeAsUsize;

impl ArchiveWith<Option<NonMaxUsize>> for OptionNonMaxUsizeAsUsize {
    type Archived = ArchivedUsize;
    type Resolver = ();

    #[inline(always)]
    fn resolve_with(field: &Option<NonMaxUsize>, _: Self::Resolver, out: Place<Self::Archived>) {
        let value = field.map_or(usize::MAX, |n| n.get());
        usize::resolve(&value, (), out);
    }
}

impl<S: Fallible + ?Sized> SerializeWith<Option<NonMaxUsize>, S> for OptionNonMaxUsizeAsUsize {
    #[inline(always)]
    fn serialize_with(
        _field: &Option<NonMaxUsize>,
        _serializer: &mut S,
    ) -> Result<Self::Resolver, S::Error> {
        Ok(())
    }
}

impl<D: Fallible + ?Sized> DeserializeWith<ArchivedUsize, Option<NonMaxUsize>, D>
    for OptionNonMaxUsizeAsUsize
{
    #[inline(always)]
    fn deserialize_with(
        field: &ArchivedUsize,
        _deserializer: &mut D,
    ) -> Result<Option<NonMaxUsize>, D::Error> {
        let value = field.to_native() as usize;
        Ok((value != usize::MAX).then(|| {
            NonMaxUsize::new(value).expect("archived nonmax leaf idx used usize::MAX sentinel")
        }))
    }
}

#[inline]
fn align_writer_to<S, E>(serializer: &mut S, align: usize) -> Result<usize, E>
where
    S: Writer<E> + ?Sized,
{
    debug_assert!(align.is_power_of_two());
    const ZEROS: [u8; CACHELINE_ALIGN] = [0; CACHELINE_ALIGN];

    let padding = (align - (serializer.pos() & (align - 1))) & (align - 1);
    let mut remaining = padding;
    while remaining != 0 {
        let chunk = remaining.min(ZEROS.len());
        serializer.write(&ZEROS[..chunk])?;
        remaining -= chunk;
    }

    Ok(serializer.pos())
}
