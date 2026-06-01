/// Kiddo example 3: Serde
///
/// This example extends the Serde deserialization from Example 1
/// by demonstrating serialization to/from gzipped Postcard.
mod cities;

use std::error::Error;
use std::fs::File;
use std::io::{Read, Write};

use elapsed::ElapsedDuration;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use std::time::Instant;

use serde::Deserialize;

use cities::{degrees_lat_lng_to_unit_sphere, parse_csv_file};
use kiddo::dist::SquaredEuclidean;
use kiddo::{Eytzinger, KdTree, VecOfArenas, VecOfArrays};

/// Each `CityCsvRecord` corresponds to 1 row in our city source data CSV.
///
/// Serde uses this to deserialize the CSV into a convenient format for us to work with.
#[derive(Debug, Deserialize)]
pub struct CityCsvRecord {
    #[allow(dead_code)]
    name: String,

    #[serde(rename = "latitude")]
    lat: f32,
    #[serde(rename = "longitude")]
    lng: f32,
}

impl CityCsvRecord {
    pub fn as_xyz(&self) -> [f32; 3] {
        degrees_lat_lng_to_unit_sphere(self.lat, self.lng)
    }
}

// We use a larger bucket size for the mutable tree in this example because the
// GeoNames dataset contains many near-duplicate coordinates.
const BUCKET_SIZE: usize = 1024;

type Tree =
    KdTree<f32, usize, Eytzinger<3>, VecOfArrays<f32, usize, 3, BUCKET_SIZE>, 3, BUCKET_SIZE>;

fn main() -> Result<(), Box<dyn Error>> {
    // Load in the cities data from the CSV and use it to populate a k-d tree, as per
    // the cities.rs example
    let start = Instant::now();
    let cities: Vec<CityCsvRecord> = parse_csv_file("./examples/geonames.csv")?;
    println!(
        "Parsed {} rows from the CSV: ({})",
        cities.len(),
        ElapsedDuration::new(start.elapsed())
    );

    let city_entries: Vec<(usize, [f32; 3])> = cities
        .iter()
        .enumerate()
        .map(|(idx, city)| (idx, city.as_xyz()))
        .collect();

    let start = Instant::now();
    let kdtree: Tree = KdTree::new_from_entries(&city_entries)?;
    println!(
        "Populated k-d tree with {} items ({})",
        kdtree.size(),
        ElapsedDuration::new(start.elapsed())
    );

    // Test query on the newly created tree
    let query = degrees_lat_lng_to_unit_sphere(52.5f32, -1.9f32);
    let nearest_neighbour = kdtree
        .query(&query)
        .nearest_one::<SquaredEuclidean<f32>>()
        .execute();
    let nearest_city = &cities[nearest_neighbour.item as usize];
    println!("\nNearest city to 52.5N, 1.9W: {nearest_city:?}");

    let start = Instant::now();
    let file = File::create("./examples/geonames-tree.postcard.gz")?;
    let mut encoder = GzEncoder::new(file, Compression::default());
    let serialized = postcard::to_allocvec(&kdtree)?;
    encoder.write_all(&serialized)?;
    encoder.finish()?;
    println!(
        "Serialized k-d tree to gzipped postcard file ({})",
        ElapsedDuration::new(start.elapsed())
    );

    let start = Instant::now();
    let file = File::open("./examples/geonames-tree.postcard.gz")?;
    let mut decompressor = GzDecoder::new(file);
    let mut bytes = Vec::new();
    decompressor.read_to_end(&mut bytes)?;
    let deserialized_tree: Tree = postcard::from_bytes(&bytes)?;
    println!(
        "Deserialized gzipped postcard file back into a k-d tree ({})",
        ElapsedDuration::new(start.elapsed())
    );

    // Test that the deserialization worked
    let query = degrees_lat_lng_to_unit_sphere(52.5f32, -1.9f32);
    let nearest_neighbour = deserialized_tree
        .query(&query)
        .nearest_one::<SquaredEuclidean<f32>>()
        .execute();
    let nearest_city = &cities[nearest_neighbour.item as usize];
    println!("\nNearest city to 52.5N, 1.9W: {nearest_city:?}");

    let city_points: Vec<_> = cities.iter().map(|city| city.as_xyz()).collect();
    println!("Building an ImmutableKdTree...");
    // Build an ImmutableKdTree
    let start = Instant::now();
    let kdtree: KdTree<f32, u32, Eytzinger<3>, VecOfArenas<f32, u32, 3, 32>, 3, 32> =
        KdTree::new_from_slice(&city_points).unwrap();
    println!(
        "Built an ImmutableKdTree ({})",
        ElapsedDuration::new(start.elapsed())
    );

    let original_nearest_neighbour_result = kdtree
        .query(&query)
        .nearest_one::<SquaredEuclidean<f32>>()
        .execute();

    let start = Instant::now();
    let file = File::create("./examples/geonames-immutable-tree.postcard.gz")?;
    let mut encoder = GzEncoder::new(file, Compression::default());
    let serialized = postcard::to_allocvec(&kdtree)?;
    encoder.write_all(&serialized)?;
    encoder.finish()?;
    println!(
        "Serialized k-d tree to gzipped postcard file ({})",
        ElapsedDuration::new(start.elapsed())
    );

    let start = Instant::now();
    let file = File::open("./examples/geonames-immutable-tree.postcard.gz")?;
    let mut decompressor = GzDecoder::new(file);
    let mut bytes = Vec::new();
    decompressor.read_to_end(&mut bytes)?;
    let deserialized_tree: KdTree<f32, u32, Eytzinger<3>, VecOfArenas<f32, u32, 3, 32>, 3, 32> =
        postcard::from_bytes(&bytes)?;
    println!(
        "Deserialized gzipped postcard file back into a k-d tree ({})",
        ElapsedDuration::new(start.elapsed())
    );

    // Test that the deserialization worked
    let query = degrees_lat_lng_to_unit_sphere(52.5f32, -1.9f32);
    let nearest_neighbour_result = deserialized_tree
        .query(&query)
        .nearest_one::<SquaredEuclidean<f32>>()
        .execute();
    assert_eq!(
        nearest_neighbour_result.distance,
        original_nearest_neighbour_result.distance
    );
    assert_eq!(
        nearest_neighbour_result.item,
        original_nearest_neighbour_result.item
    );
    let nearest = &cities[nearest_neighbour_result.item as usize];
    println!("\nNearest city to 52.5N, 1.9W: {nearest:?}");

    Ok(())
}
