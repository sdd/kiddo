mod geonames_embedded_types;

use std::error::Error;
use std::fmt::Write as _;
use std::num::NonZeroUsize;
use std::time::Instant;

use elapsed::ElapsedDuration;
use kiddo::SquaredEuclidean;
use rkyv_08::access_unchecked;

use geonames_embedded_types::ArchivedEmbeddedGeoNames;

const EARTH_RADIUS_IN_KM: f32 = 6371.0;

#[repr(align(128))]
struct AlignedBytes<const N: usize>([u8; N]);

static EMBEDDED_ARCHIVE_BYTES: AlignedBytes<
    { include_bytes!("data/geonames-embedded.rkyv").len() },
> = AlignedBytes(*include_bytes!("data/geonames-embedded.rkyv"));

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let lat_arg = args.next().ok_or("missing latitude arg, e.g. `52.5N`")?;
    let lon_arg = args.next().ok_or("missing longitude arg, e.g. `1.9W`")?;
    if args.next().is_some() {
        return Err("expected exactly two args: `<lat> <lon>`".into());
    }

    let lat = parse_lat(&lat_arg)?;
    let lon = parse_lon(&lon_arg)?;
    let query = degrees_lat_lng_to_unit_sphere(lat, lon);

    let bytes = &EMBEDDED_ARCHIVE_BYTES.0[..];

    let start = Instant::now();
    let archived_unchecked = unsafe { access_unchecked::<ArchivedEmbeddedGeoNames>(bytes) };
    let unchecked_elapsed = start.elapsed();
    println!(
        "zero-copy deserialized embedded tree: city_count={} ({})",
        archived_unchecked.cities.len(),
        ElapsedDuration::new(unchecked_elapsed)
    );

    let start = Instant::now();
    let nearest = archived_unchecked
        .tree
        .query(&query)
        .nearest_one::<SquaredEuclidean<f32>>()
        .execute();
    let query_elapsed = start.elapsed();

    let city = archived_unchecked
        .cities
        .get(nearest.item as usize)
        .ok_or("nearest item index out of bounds for embedded city data")?;
    println!(
        "Nearest city to {} {}: {}, {} ({:.4}, {:.4}), pop. {}, {:.1}km ({})",
        format_coord(lat, true),
        format_coord(lon, false),
        city.name.as_str(),
        city.country_code.as_str(),
        city.lat,
        city.lng,
        city.population,
        unit_sphere_squared_euclidean_to_kilometres(nearest.distance),
        ElapsedDuration::new(query_elapsed)
    );

    let start = Instant::now();
    let nearest_five = archived_unchecked
        .tree
        .query(&query)
        .nearest_n::<SquaredEuclidean<f32>>(NonZeroUsize::new(5).unwrap())
        .execute();
    let knn_elapsed = start.elapsed();

    println!("Nearest 5 cities ({})", ElapsedDuration::new(knn_elapsed));
    for result in nearest_five {
        let city = archived_unchecked
            .cities
            .get(result.item as usize)
            .ok_or("nearest_n item index out of bounds for embedded city data")?;
        println!(
            "  - {}, {} ({:.1}km)",
            city.name.as_str(),
            city.country_code.as_str(),
            unit_sphere_squared_euclidean_to_kilometres(result.distance)
        );
    }

    Ok(())
}

fn parse_lat(raw: &str) -> Result<f32, Box<dyn Error>> {
    parse_coord(raw, true)
}

fn parse_lon(raw: &str) -> Result<f32, Box<dyn Error>> {
    parse_coord(raw, false)
}

fn parse_coord(raw: &str, is_lat: bool) -> Result<f32, Box<dyn Error>> {
    let trimmed = raw.trim();
    let mut chars = trimmed.chars();
    let last = chars.next_back().ok_or("empty coordinate")?;

    let (number_str, sign) = match last {
        'N' | 'n' if is_lat => (chars.as_str(), 1.0),
        'S' | 's' if is_lat => (chars.as_str(), -1.0),
        'E' | 'e' if !is_lat => (chars.as_str(), 1.0),
        'W' | 'w' if !is_lat => (chars.as_str(), -1.0),
        _ => (trimmed, 1.0),
    };

    let value: f32 = number_str.trim().parse()?;
    Ok(value * sign)
}

fn format_coord(value: f32, is_lat: bool) -> String {
    let suffix = if is_lat {
        if value >= 0.0 {
            'N'
        } else {
            'S'
        }
    } else if value >= 0.0 {
        'E'
    } else {
        'W'
    };
    let mut out = String::new();
    let _ = write!(&mut out, "{:.4}{}", value.abs(), suffix);
    out
}

fn degrees_lat_lng_to_unit_sphere(lat: f32, lng: f32) -> [f32; 3] {
    let lat = lat.to_radians();
    let lng = lng.to_radians();
    [lat.cos() * lng.cos(), lat.cos() * lng.sin(), lat.sin()]
}

fn unit_sphere_squared_euclidean_to_kilometres(sq_euc_dist: f32) -> f32 {
    sq_euc_dist.sqrt() * EARTH_RADIUS_IN_KM
}
