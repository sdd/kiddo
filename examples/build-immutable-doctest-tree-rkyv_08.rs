use kiddo::immutable::float::kdtree::ImmutableKdTree;
use rkyv_08::{rancor::Error as RkyvError, to_bytes};
use std::error::Error;
use std::fs::File;
use std::io::Write;

fn main() -> Result<(), Box<dyn Error>> {
    // build and serialize small tree for ArchivedImmutableKdTree doctests
    let content: Vec<[f64; 3]> = vec![[1.0, 2.0, 5.0], [2.0, 3.0, 6.0]];
    let tree: ImmutableKdTree<f64, u32, 3, 256> = ImmutableKdTree::new_from_slice(&content);

    let buf = to_bytes::<RkyvError>(&tree)?;

    let mut file = File::create("./examples/immutable-doctest-tree_rkyv08.rkyv")?;
    file.write_all(&buf)
        .expect("Could not write serialized rkyv to file");

    Ok(())
}
