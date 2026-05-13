/// Kiddo example: Pointcloud LAS file
///
/// Populates a v6 `KdTree` from a pointcloud LAS/LAZ file and performs a query.
/// An example LAZ (compressed LAS) file can be found here:
/// https://cesium.com/public/learn/House.laz
use elapsed::ElapsedDuration;
use std::error::Error;
use std::time::Instant;

use kiddo::dist::SquaredEuclidean;
use kiddo::kd_tree::leaf_strategies::FlatVec;
use kiddo::kd_tree::KdTree;
use kiddo::Eytzinger;
use las::Reader;

type Tree = KdTree<f32, u32, Eytzinger<3>, FlatVec<f32, u32, 3, 64>, 3, 64>;

fn main() -> Result<(), Box<dyn Error>> {
    let start = Instant::now();
    let mut reader = Reader::from_path("./examples/House.laz")?;

    let points: Vec<[f32; 3]> = reader
        .points()
        .map(|point| {
            let point = point.unwrap();
            [point.x as f32, point.y as f32, point.z as f32]
        })
        .collect();

    println!("Points loaded from LAZ file. Count: {}", points.len());

    let kdtree: Tree = KdTree::new_from_slice(&points)?;

    println!(
        "Populated k-d tree with {} items. Took {}",
        kdtree.size(),
        ElapsedDuration::new(start.elapsed())
    );

    let query = [0.123f32, 0.456f32, 0.789f32];
    let (distance, item) = kdtree.nearest_one::<SquaredEuclidean<f32>>(&query);
    println!("Nearest item to query: index={item}, distance={distance:?}");

    Ok(())
}
