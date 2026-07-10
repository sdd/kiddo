mod geonames_embedded_types;

use csv::Reader;
use elapsed::ElapsedDuration;
use rkyv_08::{rancor::Error as RkyvError, to_bytes};
use serde::Deserialize;
use std::error::Error;
use std::fs;
use std::path::Path;
use std::time::Instant;

use geonames_embedded_types::{EmbeddedCity, EmbeddedGeoNames, Tree};

const CSV_PATH: &str = "./examples/data/geonames.csv";
const ARCHIVE_PATH: &str = "./examples/data/geonames-embedded.rkyv";

#[derive(Debug, Deserialize)]
struct GeoNameRecord {
    name: String,
    #[serde(rename = "latitude")]
    lat: String,
    #[serde(rename = "longitude")]
    lng: String,
    #[serde(rename = "feature class")]
    feature_class: String,
    #[serde(rename = "country code")]
    country_code: String,
    population: String,
}

fn main() -> Result<(), Box<dyn Error>> {
    if !Path::new(CSV_PATH).exists() {
        return Err(format!(
            "{} not found. Run `cargo run --example download-geonames-data --features csv` first.",
            CSV_PATH
        )
        .into());
    }

    let start = Instant::now();
    let mut rdr = Reader::from_path(CSV_PATH)?;
    let mut cities = Vec::new();
    for record in rdr.deserialize::<GeoNameRecord>() {
        let record = match record {
            Ok(record) => record,
            Err(_) => continue,
        };
        if record.feature_class != "P" {
            continue;
        }

        let lat: f32 = match record.lat.parse() {
            Ok(lat) => lat,
            Err(_) => continue,
        };
        let lng: f32 = match record.lng.parse() {
            Ok(lng) => lng,
            Err(_) => continue,
        };
        let population: u32 = match record.population.parse() {
            Ok(population) => population,
            Err(_) => continue,
        };

        cities.push(EmbeddedCity {
            name: record.name,
            country_code: record.country_code,
            lat,
            lng,
            population,
        });
    }
    println!(
        "Filtered {} populated places from {} ({})",
        cities.len(),
        CSV_PATH,
        ElapsedDuration::new(start.elapsed())
    );

    cities.sort_by(|a, b| {
        b.population
            .cmp(&a.population)
            .then_with(|| a.name.cmp(&b.name))
    });

    let start = Instant::now();
    let entries: Vec<(u32, [f32; 3])> = cities
        .iter()
        .enumerate()
        .map(|(idx, city)| {
            (
                idx as u32,
                degrees_lat_lng_to_unit_sphere(city.lat, city.lng),
            )
        })
        .collect();
    let tree: Tree = Tree::new_from_entries(&entries)?;
    println!(
        "Built embedded tree with {} items ({})",
        tree.size(),
        ElapsedDuration::new(start.elapsed())
    );

    let start = Instant::now();
    let embedded = EmbeddedGeoNames { tree, cities };
    let bytes = to_bytes::<RkyvError>(&embedded)?;
    fs::write(ARCHIVE_PATH, &bytes)?;
    println!(
        "Serialized archive to {} ({} bytes, {})",
        ARCHIVE_PATH,
        bytes.len(),
        ElapsedDuration::new(start.elapsed())
    );

    Ok(())
}

fn degrees_lat_lng_to_unit_sphere(lat: f32, lng: f32) -> [f32; 3] {
    let lat = lat.to_radians();
    let lng = lng.to_radians();
    [lat.cos() * lng.cos(), lat.cos() * lng.sin(), lat.sin()]
}
