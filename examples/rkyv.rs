/// Kiddo example 2: Rkyv
///
/// Creates a large tree of random data and serialize it with Rkyv
/// to a file.
///
/// Deserializes it back into a KdTree using four different approaches:
/// memmapped & ZC, ZC without memmmap, Memmapped without ZC, and niether memmapped nor ZC.
mod cities;

use elapsed::ElapsedDuration;
use memmap::MmapOptions;
use std::error::Error;
use std::fs::File;
use std::io::{Read, Write};
use std::time::Instant;

use kiddo::{float::distance::SquaredEuclidean, float::kdtree::KdTree};

use rkyv::ser::serializers::{AlignedSerializer, BufferScratch, CompositeSerializer};
use rkyv::ser::Serializer;
use rkyv::{AlignedVec, Deserialize, Infallible};

use cities::{degrees_lat_lng_to_unit_sphere, parse_csv_file};

const BUFFER_LEN: usize = 300_000_000;
const SCRATCH_LEN: usize = 300_000_000;

// We need a large bucket size for this dataset as there are 11m items but
// the positional precision of the source dataset is only 4DP in degrees
// of lat / lon and so there are large numbers of points with the same value
// on some axes. All values that are the same in one axis must fit in one bucket.
const BUCKET_SIZE: usize = 1024;

type Tree = KdTree<f32, u64, 3, BUCKET_SIZE, u32>;

/// Each `CityCsvRecord` corresponds to 1 row in our city source data CSV.
///
/// Serde uses this to deserialize the CSV into a convenient format for us to work with.
#[derive(Debug, serde::Deserialize)]
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
    println!(
        "Parsed {} rows from the CSV: ({})",
        cities.len(),
        ElapsedDuration::new(start.elapsed())
    );

    let start = Instant::now();
    let mut kdtree: Tree = KdTree::with_capacity(cities.len());
    cities.iter().enumerate().for_each(|(idx, city)| {
        //println!("Adding #{} ({:?})", idx, &city);
        kdtree.add(&city.as_xyz(), idx as u64);
    });
    println!(
        "Populated kd-tree with {} items ({})",
        kdtree.size(),
        ElapsedDuration::new(start.elapsed())
    );

    // Test query on the newly created tree
    let query = degrees_lat_lng_to_unit_sphere(52.5f32, -1.9f32);
    let nearest_neighbour = kdtree.nearest_one::<SquaredEuclidean>(&query);
    let nearest_city = &cities[nearest_neighbour.item as usize];
    println!("\nNearest city to 52.5N, 1.9W: {:?}", nearest_city);

    let start = Instant::now();
    let mut file: File = File::create("./examples/geonames-tree.rkyv")?;
    serialize_to_rkyv(&mut file, kdtree);
    println!(
        "Serialized kd-tree to rkyv file ({})\n\n",
        ElapsedDuration::new(start.elapsed())
    );

    /////// zero-copy memmapped deserialization ////////////////////////////////
    let start = Instant::now();
    let mmap = unsafe { MmapOptions::new().map(&File::open("./examples/geonames-tree.rkyv")?)? };
    let mm_zc_deserialized_tree = unsafe { rkyv::archived_root::<Tree>(&mmap) };
    println!(
        "Memmap ZC Deserialized rkyv file back into a kd-tree ({})",
        ElapsedDuration::new(start.elapsed())
    );

    let query = degrees_lat_lng_to_unit_sphere(52.5f32, -1.9f32);
    let nearest_neighbour = mm_zc_deserialized_tree.nearest_one::<SquaredEuclidean>(&query);
    let nearest_city = &cities[nearest_neighbour.item as usize];
    println!("\nNearest city to 52.5N, 1.9W: {:?}", nearest_city);
    println!(
        "total elapsed: {}\n\n",
        ElapsedDuration::new(start.elapsed())
    );

    /////// zero-copy non-memmapped deserialization ////////////////////////////////
    let start = Instant::now();
    let mut buffer: Vec<u8> = Vec::new();
    File::open("./examples/geonames-tree.rkyv")?.read_to_end(&mut buffer)?;
    let zc_deserialized_tree = unsafe { rkyv::archived_root::<Tree>(&buffer) };
    println!(
        "ZC Deserialized rkyv file back into a kd-tree ({})",
        ElapsedDuration::new(start.elapsed())
    );

    let query = degrees_lat_lng_to_unit_sphere(52.5f32, -1.9f32);
    let nearest_neighbour = zc_deserialized_tree.nearest_one::<SquaredEuclidean>(&query);
    let nearest_city = &cities[nearest_neighbour.item as usize];
    println!("\nNearest city to 52.5N, 1.9W: {:?}", nearest_city);
    println!(
        "total elapsed: {}\n\n",
        ElapsedDuration::new(start.elapsed())
    );

    /////// non-zero-copy memmapped deserialization ////////////////////////////////
    let start = Instant::now();
    let mmap = unsafe { MmapOptions::new().map(&File::open("./examples/geonames-tree.rkyv")?)? };
    let archived = unsafe { rkyv::archived_root::<Tree>(&mmap) };
    let mm_deserialized_tree: Tree = archived.deserialize(&mut rkyv::Infallible).unwrap();
    println!(
        "Memmap Deserialized rkyv file back into a kd-tree ({})",
        ElapsedDuration::new(start.elapsed())
    );

    let query = degrees_lat_lng_to_unit_sphere(52.5f32, -1.9f32);
    let nearest_neighbour = mm_deserialized_tree.nearest_one::<SquaredEuclidean>(&query);
    let nearest_city = &cities[nearest_neighbour.item as usize];
    println!("\nNearest city to 52.5N, 1.9W: {:?}", nearest_city);
    println!(
        "total elapsed: {}\n\n",
        ElapsedDuration::new(start.elapsed())
    );

    ///////// non-zero-copy, non-memmapped deserialization ///////////////////////
    let start = Instant::now();
    let mut file = File::open("./examples/geonames-tree.rkyv")?;
    let mut buffer: Vec<u8> = Vec::new();
    file.read_to_end(&mut buffer)?;
    let archived = unsafe { rkyv::archived_root::<Tree>(&buffer) };
    let deserialized_tree: Tree = archived.deserialize(&mut rkyv::Infallible).unwrap();
    println!(
        "Deserialized rkyv file back into a kd-tree ({})",
        ElapsedDuration::new(start.elapsed())
    );

    let query = degrees_lat_lng_to_unit_sphere(52.5f32, -1.9f32);
    let nearest_neighbour = deserialized_tree.nearest_one::<SquaredEuclidean>(&query);
    let nearest_city = &cities[nearest_neighbour.item as usize];
    println!("\nNearest city to 52.5N, 1.9W: {:?}", nearest_city);
    println!(
        "total elapsed: {}\n\n",
        ElapsedDuration::new(start.elapsed())
    );

    Ok(())
}

fn serialize_to_rkyv(file: &mut File, tree: Tree) {
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
    let _ = file
        .write(buf)
        .expect("Could not write serialized rkyv to file");
}
