use csv::Writer;
use elapsed::ElapsedDuration;
use reqwest::blocking::Client;
use std::error::Error;
use std::fs::{self, File};
use std::io::{copy, BufWriter};
use std::path::Path;
use std::time::Instant;
use zip::ZipArchive;

const GEONAMES_URL: &str = "https://download.geonames.org/export/dump/allCountries.zip";
const DATA_DIR: &str = "./examples/data";
const ZIP_PATH: &str = "./examples/data/allCountries.zip";
const CSV_PATH: &str = "./examples/data/geonames.csv";

const HEADER: [&str; 19] = [
    "geonameid",
    "name",
    "asciiname",
    "alternatenames",
    "latitude",
    "longitude",
    "feature class",
    "feature code",
    "country code",
    "cc2",
    "admin1 code",
    "admin2 code",
    "admin3 code",
    "admin4 code",
    "population",
    "elevation",
    "dem",
    "timezone",
    "modification date",
];

fn main() -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(DATA_DIR)?;

    if Path::new(CSV_PATH).exists() {
        println!("GeoNames CSV already present at {}", CSV_PATH);
        return Ok(());
    }

    ensure_zip_downloaded()?;
    convert_zip_to_csv()?;
    Ok(())
}

fn ensure_zip_downloaded() -> Result<(), Box<dyn Error>> {
    if Path::new(ZIP_PATH).exists() {
        println!("GeoNames zip already present at {}", ZIP_PATH);
        return Ok(());
    }

    let start = Instant::now();
    let client = Client::builder().build()?;
    let mut response = client.get(GEONAMES_URL).send()?.error_for_status()?;
    let mut file = BufWriter::new(File::create(ZIP_PATH)?);
    copy(&mut response, &mut file)?;

    println!(
        "Downloaded {} to {} ({})",
        GEONAMES_URL,
        ZIP_PATH,
        ElapsedDuration::new(start.elapsed())
    );

    Ok(())
}

fn convert_zip_to_csv() -> Result<(), Box<dyn Error>> {
    let start = Instant::now();
    let file = File::open(ZIP_PATH)?;
    let mut archive = ZipArchive::new(file)?;
    let zipped = archive.by_name("allCountries.txt")?;

    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .has_headers(false)
        .flexible(true)
        .from_reader(zipped);
    let mut wtr = Writer::from_path(CSV_PATH)?;
    wtr.write_record(HEADER)?;

    let mut rows = 0usize;
    for record in rdr.records() {
        let record = record?;
        if record.len() >= HEADER.len() {
            wtr.write_record(record.iter().take(HEADER.len()))?;
            rows += 1;
        }
    }
    wtr.flush()?;

    println!(
        "Wrote {} rows to {} ({})",
        rows,
        CSV_PATH,
        ElapsedDuration::new(start.elapsed())
    );

    Ok(())
}
