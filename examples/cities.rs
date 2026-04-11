/// Kiddo example 1: Cities
///
/// This example walks through the basics of using the v6 `KdTree` to
/// populate a tree from CSV data and then use it to perform
/// nearest-neighbour, k-nearest-neighbour, radius, and best-within queries.
use csv::Reader;
use kiddo::dist::SquaredEuclidean;
use kiddo::kd_tree::leaf_strategies::FlatVec;
use kiddo::kd_tree::KdTree;
use kiddo::Eytzinger;
use serde::Deserialize;
use std::error::Error;
use std::fmt::Formatter;
use std::fs::File;
use std::num::NonZeroUsize;

#[allow(dead_code)]
pub const EARTH_RADIUS_IN_KM: f32 = 6371.0;

/// Each `CityCsvRecord` corresponds to 1 row in our city source data CSV.
///
/// Serde uses this to deserialize the CSV into a convenient format for us to work with.
#[derive(Debug, Deserialize)]
pub struct CityCsvRecord {
    #[allow(dead_code)]
    #[serde(rename = "city")]
    name: String,

    lat: f32,
    lng: f32,

    country: String,
    population: u32,
}

impl std::fmt::Display for CityCsvRecord {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}, {} (pop. {})",
            self.name, self.country, self.population
        )
    }
}

impl CityCsvRecord {
    pub fn as_xyz(&self) -> [f32; 3] {
        degrees_lat_lng_to_unit_sphere(self.lat, self.lng)
    }
}

/// converts Earth surface co-ordinates in degrees of latitude and longitude to 3D cartesian coordinates on a unit sphere
///
/// We use this when populating our tree, to convert from the `f32` lat/lng data into `f32` (x,y,z) co-ordinates to store in our tree, as well as
/// allowing us to query the created tree using lat/lng query points.
pub fn degrees_lat_lng_to_unit_sphere(lat: f32, lng: f32) -> [f32; 3] {
    // convert from degrees to radians
    let lat = lat.to_radians();
    let lng = lng.to_radians();

    // convert from ra/dec to xyz coords on unit sphere
    [lat.cos() * lng.cos(), lat.cos() * lng.sin(), lat.sin()]
}

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn Error>> {
    let cities: Vec<CityCsvRecord> = parse_csv_file("./examples/worldcities.csv")?;
    let city_points: Vec<[f32; 3]> = cities.iter().map(CityCsvRecord::as_xyz).collect();

    type Tree = KdTree<f32, u32, Eytzinger<3>, FlatVec<f32, u32, 3, 32>, 3, 32>;
    let kdtree: Tree = KdTree::new_from_slice(&city_points);

    println!("Loaded {} items into Kiddo k-d tree", kdtree.size());

    // ### find the nearest city to 52.5N, 1.9W
    // First, let's try a nearest-neighbour query. Let's say we want to
    // find the nearest city to a point with latitude 52.5 degrees north,
    // longitude 1.9 degrees west.

    // since our positions are stored as XYZ on a unit sphere, we
    // first need to convert our query point into this co-ordinate scheme:
    let query = degrees_lat_lng_to_unit_sphere(52.5f32, -1.9f32);

    let (nearest_dist, nearest_idx) = kdtree.nearest_one::<SquaredEuclidean<f32>>(&query);

    let nearest = &cities[nearest_idx as usize];
    println!(
        "\nNearest city to 52.5N, 1.9W: {} ({:.1})km",
        nearest,
        unit_sphere_squared_euclidean_to_kilometres(nearest_dist)
    );

    // ### Find the nearest five cities to 52.5N, 1.9W
    // Let's try something similar, but a K-nearest-neighbour (KNN) query instead.
    // This allows us to find, for example, the five nearest cities to a specified
    // point, sorted in order of distance.
    let query = degrees_lat_lng_to_unit_sphere(52.5f32, -1.9f32);
    let nearest_5_idx =
        kdtree.nearest_n::<SquaredEuclidean<f32>>(&query, NonZeroUsize::new(5).unwrap(), true);

    let nearest_5 = nearest_5_idx
        .into_iter()
        .map(|neighbour| {
            (
                &cities[neighbour.item as usize].name,
                format!(
                    "{dist:.1}km",
                    dist = unit_sphere_squared_euclidean_to_kilometres(neighbour.distance)
                ),
            )
        })
        .collect::<Vec<_>>();
    println!("\nNearest 5 cities to 52.5N, 1.9W: {nearest_5:?}");

    // ### Find all cities within 1000km of 0N, 0E
    // Kiddo's `within` method returns the indices of all points within
    // a certain distance of a query point, alongside the distance from that
    // point, as an iterator. Here we use `within` to find all cities within
    // 1000km of 0N 0W.
    let query = degrees_lat_lng_to_unit_sphere(0f32, 0f32);
    let dist = kilometres_to_unit_sphere_squared_euclidean(1000.0);
    let all_within = kdtree
        .within::<SquaredEuclidean<f32>>(&query, dist)
        .iter()
        .map(|neighbour| &cities[neighbour.item as usize].name)
        .collect::<Vec<_>>();
    println!("\nAll cities within 1000km of 0N, 0W: {all_within:?}");

    // ### Find the most populous 3 cities within 500km of 0N, 0E
    // Kiddo provides a `best_n_within` method that can provide great performance
    // for certain types of queries. We're going to use it here in order to find
    // the three most populous cities within 500km of a query point.
    // In order for this to work, your indices need to be ordered, with "best"
    // (highest population, in this case) having lower-numbered indices. Since
    // our input CSV is ordered by decending population, this works for us.
    // Without this method, users would need to call Kiddo::within` to find
    // all cities within 500km and then sort the results before taking the
    // first three, which is significantly slower.
    let query = degrees_lat_lng_to_unit_sphere(0f32, 0f32);
    let dist = kilometres_to_unit_sphere_squared_euclidean(1000.0);
    let best_3_iter =
        kdtree.best_n_within::<SquaredEuclidean<f32>>(&query, dist, NonZeroUsize::new(3).unwrap());
    let best_3 = best_3_iter
        .into_iter()
        .map(|neighbour| &cities[neighbour.item as usize].name)
        .collect::<Vec<_>>();
    println!("\nMost populous 3 cities within 1000km of 0N, 0W: {best_3:?}");

    Ok(())
}

/// Converts a squared euclidean unit sphere distance (like what we'd get back from
/// our k-d tree) into kilometres for user convenience.
#[allow(dead_code)]
pub fn unit_sphere_squared_euclidean_to_kilometres(sq_euc_dist: f32) -> f32 {
    sq_euc_dist.sqrt() * EARTH_RADIUS_IN_KM
}

/// Converts a value in km to squared euclidean distance on a unit sphere representing Earth.
///
/// This allows us to query using kilometres as distances in our k-d tree.
pub fn kilometres_to_unit_sphere_squared_euclidean(km_dist: f32) -> f32 {
    (km_dist / EARTH_RADIUS_IN_KM).powi(2)
}

/// Parses CSV data from `file` into a `Vec` of `R`
pub fn parse_csv_file<R: for<'de> serde::Deserialize<'de>>(
    filename: &str,
) -> Result<Vec<R>, std::io::Error> {
    let file = File::open(filename)?;
    let cities: Vec<R> = Reader::from_reader(file)
        .deserialize()
        .filter_map(Result::ok)
        .collect();

    println!("Cities successfully parsed from CSV: {:?}", cities.len());

    Ok(cities)
}
