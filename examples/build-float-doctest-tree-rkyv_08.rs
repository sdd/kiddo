// This example builds /examples/float-doctest-tree-rkyv_08.rkyv for ArchivedKdTree doctests.
use std::error::Error;
use std::fs::File;
use std::io::Write;

use kiddo::kd_tree::leaf_strategies::VecOfArenas;
use kiddo::kd_tree::KdTree;
use kiddo::stem_strategies::EytzingerPf;
use rkyv_08::{rancor::Error as RkyvError, to_bytes};

type Tree = KdTree<f64, u32, EytzingerPf<3, 8>, VecOfArenas<f64, u32, 3, 256>, 3, 256>;

fn main() -> Result<(), Box<dyn Error>> {
    let points: Vec<[f64; 3]> = vec![[1.0, 2.0, 5.0], [2.0, 3.0, 6.0]];
    let tree: Tree = KdTree::new_from_slice(&points);

    let buf = to_bytes::<RkyvError>(&tree)?;
    let mut file = File::create("./examples/float-doctest-tree-rkyv_08.rkyv")?;
    file.write_all(&buf)?;

    Ok(())
}
