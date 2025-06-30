/// Kiddo example: Pointcloud LAS file
///
/// Populates a Kiddo KdTree from a pointcloud LAS file
/// An example LAZ (compressed LAS) file can be found here:
/// https://cesium.com/public/learn/House.laz
use elapsed::ElapsedDuration;
use std::error::Error;
use std::time::Instant;

use csv::Reader;
use kiddo::immutable::float::kdtree::ImmutableKdTree;
use kiddo::SquaredEuclidean;

use serde::Deserialize;
use tracing::Level;
use tracing_subscriber::fmt;

#[derive(Debug, Deserialize)]
struct Point {
    x: f64,
    y: f64,
    z: f64,
}

type Tree = ImmutableKdTree<f64, u32, 3, 64>;

fn main() -> Result<(), Box<dyn Error>> {
    #[cfg(feature = "tracing")]
    let subscriber = fmt().with_max_level(Level::WARN).without_time().finish();
    #[cfg(feature = "tracing")]
    tracing::subscriber::set_global_default(subscriber)?;

    // create a tree populated with random points
    let start = Instant::now();

    let mut reader = Reader::from_path("./points.csv")?;

    let points: Vec<[f64; 3]> = reader
        .deserialize()
        .map(|point| {
            let point: Point = point.unwrap();
            [point.x, point.y, point.z]
        })
        .collect();

    println!("Points loaded from CSV file. Count: {}", points.len());

    let kdtree: Tree = (&*points).into();

    println!(
        "Populated k-d tree with {} items. Took {}",
        kdtree.size(),
        ElapsedDuration::new(start.elapsed())
    );

    // Test query on the newly created tree
    let query = [0.123f64, 0.456f64, 0.789f64];
    let nearest_neighbour = kdtree.nearest_one::<SquaredEuclidean>(&query);
    println!("Nearest item to query: {:?}", nearest_neighbour.item);

    Ok(())
}
