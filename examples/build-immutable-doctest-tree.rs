// This example builds /examples/immutable-doctest-tree.rkyv, which is needed
// if you want to run the ArchivedImmutableTree doctests
use std::error::Error;
use std::fs::File;
use std::io::Write;

use rkyv::ser::serializers::{AlignedSerializer, BufferScratch, CompositeSerializer};
use rkyv::ser::Serializer;
use rkyv::{AlignedVec, Infallible};

use kiddo::ImmutableKdTree;

const BUFFER_LEN: usize = 300_000;
const SCRATCH_LEN: usize = 300_000;

fn main() -> Result<(), Box<dyn Error>> {
    // build and serialize small tree for ArchivedImmutableKdTree doctests
    let content: Vec<[f64; 3]> = vec![[1.0, 2.0, 5.0], [2.0, 3.0, 6.0]];
    let tree: ImmutableKdTree<f64, 3> = ImmutableKdTree::new_from_slice(&content);

    let mut serialize_buffer = AlignedVec::with_capacity(BUFFER_LEN);
    let mut serialize_scratch = AlignedVec::with_capacity(SCRATCH_LEN);
    unsafe {
        serialize_scratch.set_len(SCRATCH_LEN);
    }
    serialize_buffer.clear();
    let mut serializer = CompositeSerializer::new(
        AlignedSerializer::new(&mut serialize_buffer),
        BufferScratch::new(&mut serialize_scratch),
        Infallible,
    );
    serializer
        .serialize_value(&tree)
        .expect("Could not serialize with rkyv");

    let buf = serializer.into_serializer().into_inner();
    let mut file = File::create("./examples/immutable-doctest-tree.rkyv")?;
    file.write_all(buf)
        .expect("Could not write serialized rkyv to file");

    Ok(())
}
