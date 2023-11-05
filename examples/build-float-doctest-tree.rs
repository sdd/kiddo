// This example builds /examples/float-doctest-tree.rkyv, which is needed
// if you want to run the float ArchivedTree doctests
use std::error::Error;
use std::fs::File;
use std::io::Write;

use rkyv::ser::serializers::{AlignedSerializer, BufferScratch, CompositeSerializer};
use rkyv::ser::Serializer;
use rkyv::{AlignedVec, Infallible};

use kiddo::KdTree;

const BUFFER_LEN: usize = 300_000;
const SCRATCH_LEN: usize = 300_000;

fn main() -> Result<(), Box<dyn Error>> {
    // build and serialize small tree for ArchivedKdTree doctests
    let mut tree: KdTree<f64, 3> = KdTree::new();
    tree.add(&[1.0, 2.0, 5.0], 100);
    tree.add(&[2.0, 3.0, 6.0], 101);

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
    let mut file = File::create("./examples/float-doctest-tree.rkyv")?;
    file.write_all(buf)
        .expect("Could not write serialized rkyv to file");

    Ok(())
}
