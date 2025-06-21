// This example builds /examples/float-doctest-tree-rkyv_08.rkyv, which is needed
// if you want to run the float ArchivedTree doctests
use std::error::Error;
use std::fs::File;
use std::io::Write;

use rkyv_08::{rancor::Error as RkyvError, to_bytes};

use kiddo::KdTree;

fn main() -> Result<(), Box<dyn Error>> {
    // build and serialize small tree for ArchivedKdTree doctests
    let mut tree: KdTree<f64, 3> = KdTree::new();
    tree.add(&[1.0, 2.0, 5.0], 100);
    tree.add(&[2.0, 3.0, 6.0], 101);

    let buf = to_bytes::<RkyvError>(&tree)?;

    let mut file = File::create("./examples/float-doctest-tree-rkyv_08.rkyv")?;
    file.write_all(&buf)
        .expect("Could not write serialized rkyv to file");

    Ok(())
}
