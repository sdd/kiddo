/// Kiddo example 2: Serde
///
/// This example extends the Serde deserialization from Example 1
/// by demonstrating serialization to/from JSON and gzipped Bincode
mod cities;

use std::error::Error;
use std::fs::File;
use std::io::{Read, Write};
use std::time::Instant;
use elapsed::ElapsedDuration;




use kiddo::{float::distance::squared_euclidean, KdTree};

use rkyv::{AlignedVec, Deserialize, Infallible};
use rkyv::ser::Serializer;
use rkyv::ser::serializers::{AlignedSerializer, BufferScratch, CompositeSerializer};

use cities::{degrees_lat_lng_to_unit_sphere, parse_csv_file};

const BUFFER_LEN: usize = 300_000_000;
const SCRATCH_LEN: usize = 300_000_000;

/// Each `CityCsvRecord` corresponds to 1 row in our city source data CSV.
///
/// Serde uses this to deserialize the CSV into a convenient format for us to work with.
#[derive(Debug,serde::Deserialize)]
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

fn main() -> Result<(), Box<dyn Error>> {

    // Load in the cities data from the CSV and use it to populate a kd-tree, as per
    // the cities.rs example
    let start = Instant::now();
    let cities: Vec<CityCsvRecord> = parse_csv_file("./examples/geonames.csv")?;
    println!("Parsed {} rows from the CSV: ({})", cities.len(), ElapsedDuration::new(start.elapsed()));

    let start = Instant::now();
    let mut kdtree: KdTree<f32, u32, 3, 32, u32> = KdTree::with_capacity(cities.len());
    cities.iter().enumerate().for_each(|(idx, city)| {
        kdtree.add(&city.as_xyz(), idx as u32);
    });
    println!("Populated kd-tree with {} items ({})", kdtree.size(), ElapsedDuration::new(start.elapsed()));

    // Test query on the newly created tree
    let query = degrees_lat_lng_to_unit_sphere(52.5f32, -1.9f32);
    let (_, nearest_idx) = kdtree.nearest_one(&query, &squared_euclidean);
    let nearest = &cities[nearest_idx as usize];
    println!("\nNearest city to 52.5N, 1.9W: {:?}", nearest);

    let start = Instant::now();
    let mut file = File::create("./examples/geonames-tree.rkyv")?;
    serialize_to_rkyv(&mut file, kdtree);
    println!("Serialized kd-tree to rkyv file ({})", ElapsedDuration::new(start.elapsed()));

    let start = Instant::now();
    let file = File::open("./examples/geonames-tree.rkyv")?;
    let deserialized_tree: KdTree<f32, u32, 3, 32, u32> = deserialize_from_rkyv(file)?;
    println!("Deserialized rkyv file back into a kd-tree ({})", ElapsedDuration::new(start.elapsed()));

    // Test that the deserialization worked
    let query = degrees_lat_lng_to_unit_sphere(52.5f32, -1.9f32);
    let (_, nearest_idx) = deserialized_tree.nearest_one(&query, &squared_euclidean);
    let nearest = &cities[nearest_idx as usize];
    println!("\nNearest city to 52.5N, 1.9W: {:?}", nearest);

    Ok(())
}

fn serialize_to_rkyv(file: &mut File, tree: KdTree<f32, u32, 3, 32, u32>) {
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
    serializer.serialize_value(&tree).expect("Could not serialize with rkyv");
    let buf = serializer.into_serializer().into_inner();
    file.write(&buf).expect("Could not write serialized rkyv to file");
}

fn deserialize_from_rkyv(mut file: File) -> Result<KdTree<f32, u32, 3, 32, u32>, Box<dyn Error>> {
    let mut buffer: Vec<u8> = Vec::new();
    file.read_to_end(&mut buffer)?;

    let archived = unsafe { rkyv::archived_root::<KdTree<f32, u32, 3, 32, u32>>(&buffer) };
    let tree: KdTree<f32, u32, 3, 32, u32> = archived.deserialize(&mut rkyv::Infallible).unwrap();

    Ok(tree)
}
