use std::fs::File;
use std::io::Write;
use std::path::Path;
use az::Cast;
use memmap::MmapOptions;
use rkyv::ser::serializers::{AlignedSerializer, BufferScratch, CompositeSerializer};
use rkyv::ser::Serializer;
use rkyv::{AlignedVec, Deserialize, Infallible};

use crate::float::kdtree::{Axis,ArchivedKdTree,KdTree};
use crate::types::{Content, Index};

impl<A: Axis, T: Content, const K: usize, const B: usize, IDX: Index<T = IDX>>
    KdTree<A, T, K, B, IDX>
where
    usize: Cast<IDX>,
{
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P, buffer_len: usize, scratch_len: usize) -> Result<(), std::io::Error> {
        let mut file: File = File::create(path)?;

        let mut serialize_buffer = AlignedVec::with_capacity(buffer_len);
        let mut serialize_scratch = AlignedVec::with_capacity(scratch_len);

        unsafe {
            serialize_scratch.set_len(scratch_len);
        }
        serialize_buffer.clear();

        let mut serializer = CompositeSerializer::new(
            AlignedSerializer::new(&mut serialize_buffer),
            BufferScratch::new(&mut serialize_scratch),
            Infallible,
        );

        serializer
            .serialize_value(self)?;

        let buf = serializer.into_serializer().into_inner();
        Ok(file.write(&buf)?)
    }
}

impl<A: Axis + rkyv::Archive<Archived = A>, T: Content + rkyv::Archive<Archived = T>, const K: usize, const B: usize, IDX: Index<T = IDX> + rkyv::Archive<Archived = IDX>>
ArchivedKdTree<A, T, K, B, IDX>
    where
        usize: Cast<IDX>,
{
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Self {
        let mmap = unsafe { MmapOptions::new().map(&File::open(path).expect("could not open file")).expect("Could not map file") };
        *unsafe { rkyv::archived_root::<KdTree<A, T, K, B, IDX>>(&mmap) }
    }
}
