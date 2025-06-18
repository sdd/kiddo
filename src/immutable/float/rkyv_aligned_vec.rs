use aligned_vec::{AVec, CACHELINE_ALIGN};
use rkyv_08::rancor::Fallible;
use rkyv_08::with::{ArchiveWith, DeserializeWith, SerializeWith};
use rkyv_08::{
    ser::{Allocator, Writer},
    vec::{ArchivedVec, VecResolver},
    Archive, Place, Serialize,
};
use std::marker::PhantomData;

pub(crate) struct EncodeAVec<T> {
    _p: PhantomData<T>,
}

pub(crate) struct AVecResolver {
    len: usize,
    inner: VecResolver,
}

impl<T> ArchiveWith<AVec<T>> for EncodeAVec<T> {
    type Archived = ArchivedVec<T>;
    type Resolver = AVecResolver;

    fn resolve_with(_: &AVec<T>, resolver: Self::Resolver, out: Place<Self::Archived>) {
        ArchivedVec::resolve_from_len(resolver.len, resolver.inner, out);
    }
}

impl<T: Serialize<S>, S> SerializeWith<AVec<T>, S> for EncodeAVec<T>
where
    S: Fallible + Allocator + Writer + ?Sized,
{
    fn serialize_with(avec: &AVec<T>, serializer: &mut S) -> Result<Self::Resolver, S::Error> {
        Ok(AVecResolver {
            len: avec.len(),
            inner: ArchivedVec::serialize_from_slice(avec.as_slice(), serializer)?,
        })
    }
}

impl<T, D> DeserializeWith<ArchivedVec<T>, AVec<T>, D> for EncodeAVec<T>
where
    T: Archive + Clone,
    D: Fallible + ?Sized,
{
    fn deserialize_with(vec: &ArchivedVec<T>, _: &mut D) -> Result<AVec<T>, D::Error> {
        let mut result = AVec::with_capacity(CACHELINE_ALIGN, vec.len());

        for item in vec.as_slice() {
            result.push(item.clone());
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use aligned_vec::{AVec, CACHELINE_ALIGN};
    use std::ops::Rem;

    use rkyv_08::{
        access_unchecked, deserialize, rancor::Error, Archive, Archived, Deserialize, Serialize,
    };

    use crate::immutable::float::rkyv_aligned_vec::EncodeAVec;

    #[test]
    fn roundtrip_avec_deserialized() {
        #[derive(Archive, Debug, Serialize, Deserialize, PartialEq)]
        #[rkyv(crate=rkyv_08)]
        struct Obj {
            #[rkyv(with = EncodeAVec<i32>)]
            pub inner: AVec<i32>,
        }

        let original = Obj {
            inner: AVec::from_slice(CACHELINE_ALIGN, &[10, 20, 30, 40]),
        };

        let buf = rkyv_08::to_bytes::<Error>(&original).unwrap();

        // check that the deserialized values are the same
        let archived: &ArchivedObj = unsafe { access_unchecked::<Archived<Obj>>(buf.as_ref()) };
        let deserialized: Obj = deserialize::<Obj, Error>(archived).unwrap();
        assert_eq!(original, deserialized);

        // check that the deserialized AVec inside Obj has the alignment we expect
        let inner_ptr_adr_deser = deserialized.inner.as_ptr() as *const () as usize;
        assert_eq!(inner_ptr_adr_deser.rem(CACHELINE_ALIGN), 0);
    }

    #[ignore = "Fails. See comment at bottom of test"]
    #[test]
    fn roundtrip_avec_archived() {
        #[derive(Archive, Debug, Serialize, Deserialize, PartialEq)]
        #[rkyv(crate=rkyv_08)]
        struct Obj {
            #[rkyv(with = EncodeAVec<i32>)]
            pub inner: AVec<i32>,
        }

        let original = Obj {
            inner: AVec::from_slice(CACHELINE_ALIGN, &[10, 20, 30, 40]),
        };

        let buf = rkyv_08::to_bytes::<Error>(&original).unwrap();

        // check that the archived values are the same
        let archived: &ArchivedObj = unsafe { access_unchecked::<Archived<Obj>>(buf.as_ref()) };
        assert_eq!(original.inner.as_slice(), archived.inner.as_slice());

        // check that the archived AVec has the alignment we expect
        // NOTE: right now this works sometimes but not others - it
        // seems like the Archived AVec items are 32-byte aligned rather than 128
        let inner_ptr_adr_archived = archived.inner.as_ptr() as *const () as usize;
        assert_eq!(inner_ptr_adr_archived.rem(CACHELINE_ALIGN), 0);
    }
}
