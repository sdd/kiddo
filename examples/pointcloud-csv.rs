/// Kiddo example: Pointcloud CSV file
///
/// Populates a v6 `KdTree` from a pointcloud CSV file and performs a query.
use elapsed::ElapsedDuration;
use std::error::Error;
use std::time::Instant;

use csv::Reader;
use kiddo::dist::SquaredEuclidean;
use kiddo::kd_tree::leaf_strategies::FlatVec;
use kiddo::kd_tree::KdTree;
use kiddo::Eytzinger;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Point {
    x: f64,
    y: f64,
    z: f64,
}

type Tree = KdTree<f64, u32, Eytzinger<3>, FlatVec<f64, u32, 3, 64>, 3, 64>;

fn main() -> Result<(), Box<dyn Error>> {
    let start = Instant::now();
    let mut reader = Reader::from_path("./examples/points.csv")?;

    let points: Vec<[f64; 3]> = reader
        .deserialize()
        .map(|point| {
            let point: Point = point.unwrap();
            [point.x, point.y, point.z]
        })
        .collect();

    println!("Points loaded from CSV file. Count: {}", points.len());

    let kdtree: Tree = KdTree::new_from_slice(&points)?;

    println!(
        "Populated k-d tree with {} items. Took {}",
        kdtree.size(),
        ElapsedDuration::new(start.elapsed())
    );

    let query = [0.123f64, 0.456f64, 0.789f64];
    let (distance, item) = kdtree.nearest_one::<SquaredEuclidean<f64>>(&query);
    println!("Nearest item to query: index={item}, distance={distance:?}");

    Ok(())
}
