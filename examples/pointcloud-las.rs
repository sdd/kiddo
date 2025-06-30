/// Kiddo example: Pointcloud LAS file
///
/// Populates a Kiddo KdTree from a pointcloud LAS file
/// An example LAZ (compressed LAS) file can be found here:
/// https://cesium.com/public/learn/House.laz
use elapsed::ElapsedDuration;
use std::error::Error;
use std::time::Instant;

use kiddo::SquaredEuclidean;
use las::Reader;

use kiddo::immutable::float::kdtree::ImmutableKdTree;
use tracing::Level;
use tracing_subscriber::fmt;

type Tree = ImmutableKdTree<f32, u32, 3, 64>;

fn main() -> Result<(), Box<dyn Error>> {
    #[cfg(feature = "tracing")]
    let subscriber = fmt().with_max_level(Level::WARN).without_time().finish();
    #[cfg(feature = "tracing")]
    tracing::subscriber::set_global_default(subscriber)?;

    // create a tree populated with random points
    let start = Instant::now();

    let mut reader = Reader::from_path("./House.laz")?;

    let points: Vec<[f32; 3]> = reader
        .points()
        .map(|point| {
            let point = point.unwrap();
            [point.x as f32, point.y as f32, point.z as f32]
        })
        .collect();

    println!("Points loaded from LAZ file. Count: {}", points.len());

    let kdtree: Tree = (&*points).into();

    println!(
        "Populated k-d tree with {} items. Took {}",
        kdtree.size(),
        ElapsedDuration::new(start.elapsed())
    );

    // Test query on the newly created tree
    let query = [0.123f32, 0.456f32, 0.789f32];
    let nearest_neighbour = kdtree.nearest_one::<SquaredEuclidean>(&query);
    println!("Nearest item to query: {:?}", nearest_neighbour.item);

    Ok(())
}
