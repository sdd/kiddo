/// Kiddo example 1: Cities
///
/// This example walks through the basics of using Kiddo to
/// populate a kd-tree from CSV data and then use it to perform
/// nearest neighbour (NN) and k-nearest-neighbour (KNN) queries.
use std::error::Error;
use std::fmt::Formatter;
use std::fs::File;

use csv::Reader;
use kiddo::float::{distance::squared_euclidean, kdtree::KdTree};
use serde::Deserialize;

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
    // parse the CSV file into a `Vec` of `CityCsvRecord`s. We'll keep hold of this Vec,
    // since Kiddo is intended to store indexes alongside points, rather than storing
    // all the data for each point. Our queries will return indexes into this `Vec`.
    // This design choice helps keep the size of the internal leaf nodes down within
    // Kiddo, which helps with performance.
    let cities: Vec<CityCsvRecord> = parse_csv_file("./examples/worldcities.csv")?;

    // Construct an empty KdTree to hold the positions of the cities.
    // The positions of our Cities will be converted from lat/lng to 3D positions
    // on a unit sphere. This avoids the singularities at the poles and makes
    // distance measurement faster. Since we're storing our position values as floats,
    // we use kiddo::KdTree rather than kiddo:FixedKdTree. The five generic parameters
    // are explained as follows:
    // 1) A: `f32` - this specifies the type of the floating point position data that we
    //    are storing in the tree. This can be either `f32` or `f64`. `f32` provides
    //    around 7 significant figures of precision. Since our source data is around 7
    //    significant figures, and we are only interested in a single decimal place (and
    //    a max of 4 or 5 significant figures),  `f32` is sufficient for our use case.
    //    `f32` consumes half the space of `f64`, and using it can improve performance
    //    since more of the internal stem nodes in the kd-tree can fit in the CPU
    //    cache. If you need more precision than 7 significant figures, Kiddo supports `f64`.
    //    If your data covers a narrow dynamic range (e.g. all of the values are within
    //    a few orders of magnitude of each other) and performance (or memory usage)
    //    is a key concern, then you may want to experiment with converting your
    //    positions into 16-bit fixed point and using a kiddo::FixedKdTree.
    // 2) `T`: `u16` - this specifies the type of the indices that are stored in the tree.
    //    A `u16` will suffice for up to 2^16 (approx 65k) different items. A `u32` will allow you
    //    to store 2^32 (around 4 billion) different indices. Using a `u16` may
    //    result in your tree using less memory, and again this can help with performance
    //    by allowing more stem nodes to fit in the CPU cache.
    // 3) `K`: 3 - the number of dimensions that your position data has, i.e. the `k` in kd-tree.
    //    this of course depends solely on the data that you are storing. Kd tree NN and KNN
    //    queries work best with lower numbers of dimensions.
    // 4) `B`: 32 - the "bucket size". Kiddo stores points on the leaf nodes of its tree,
    //    and this value determines how many entries are stored on each leaf. 32 is a
    //    good starting point - feel free to experiment with this if you want to eke
    //    out the most performance, but I've found 32 to be a good choice most of the time.
    // 5) `IDX`: `u16` - IDX is the type used internally to index the nodes. `u16` will permit
    //    `32768 * B` items to be stored in the tree. `u32` will allow up to `~2 billion * B` items.
    //    If you want to eke out the max possible performance, benchmark with different
    //    values of this. smaller types will allow a little bit more nodes to be able to
    //    fit in the cache, but this may not necessarily equate to any improvement; YMMV.
    let mut kdtree: KdTree<f32, usize, 3, 32, u16> = KdTree::with_capacity(cities.len());

    // Now we populate the newly-created empty kd tree with data about our cities.
    // Our `CityCsvRecord`s store their position as latitude and longitude, so we
    // first use the `as_xyz` method we defined in order to convert the lat/lng values
    // into 3D xyz co-ordinates. Our kd-tree uses `u16` indices so we need to cast
    // our `usize`s from `enumerate` into `u16`s for storage into the tree.
    cities.iter().enumerate().for_each(|(idx, city)| {
        kdtree.add(&city.as_xyz(), idx);
    });

    println!("Loaded {} items into Kiddo kd-tree", kdtree.size());

    // ### find the nearest city to 52.5N, 1.9W
    // First, let's try a nearest-neighbour query. Let's say we want to
    // find the nearest city to a point with latitude 52.5 degrees north,
    // longitude 1.9 degrees west.

    // since our positions are stored as XYZ on a unit sphere, we
    // first need to convert our query point into this co-ordinate scheme:
    let query = degrees_lat_lng_to_unit_sphere(52.5f32, -1.9f32);

    // now we perform the actual nearest neighbour query. We need to specify a
    // distance metric: we use `squared_euclidean` in this case, which is a good
    // default. See the `distance` module docs for a discussion o distance metrics.
    let (_, nearest_idx) = kdtree.nearest_one(&query, &squared_euclidean);

    // since the result of the query is an index, we need to use this index
    // on the `cities` `Vec` in order to retrieve the original record.
    let nearest = &cities[nearest_idx as usize];
    println!("\nNearest city to 52.5N, 1.9W: {}", nearest);

    // ### Find the nearest five cities to 52.5N, 1.9W
    // Let's try something similar, but a K-nearest-neighbour (KNN) query instead.
    // This allows us to find, for example, the five nearest cities to a specified
    // point, sorted in order of distance.
    let query = degrees_lat_lng_to_unit_sphere(52.5f32, -1.9f32);
    let nearest_5_idx = kdtree.nearest_n(&query, 5, &squared_euclidean);

    // `kdtree::nearest_n` returns an `Iterator`, rather than a `Vec`. This
    // gives callers the flexibility of deciding how to process and store the
    // results, possibly avoiding the memory allocation required to store
    // the results in a new `Vec`. We're just `collect`ing into a `Vec` here.
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
    println!("\nNearest 5 cities to 52.5N, 1.9W: {:?}", nearest_5);

    // ### Find all cities within 1000km of 0N, 0E
    // Kiddo's `within` method returns the indices of all points within
    // a certain distance of a query point, alongside the distance from that
    // point, as an iterator. Here we use `within` to find all cities within
    // 1000km of 0N 0W.
    let query = degrees_lat_lng_to_unit_sphere(0f32, 0f32);
    let dist = kilometres_to_unit_sphere_squared_euclidean(1000.0);
    let all_within = kdtree
        .within(&query, dist, &squared_euclidean)
        .iter()
        .map(|neighbour| &cities[neighbour.item as usize].name)
        .collect::<Vec<_>>();
    println!("\nAll cities within 1000km of 0N, 0W: {:?}", all_within);

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
    let best_3_iter = kdtree.best_n_within(&query, dist, 3, &squared_euclidean);
    let best_3 = best_3_iter
        .map(|idx| (&cities[idx as usize].name))
        .collect::<Vec<_>>();
    println!(
        "\nMost populous 3 cities within 1000km of 0N, 0W: {:?}",
        best_3
    );

    Ok(())
}

/// Converts a squared euclidean unit sphere distance (like what we'd get back from
/// our kd-tree) into kilometres for user convenience.
#[allow(dead_code)]
pub fn unit_sphere_squared_euclidean_to_kilometres(sq_euc_dist: f32) -> f32 {
    sq_euc_dist.sqrt() * EARTH_RADIUS_IN_KM
}

/// Converts a value in km to squared euclidean distance on a unit sphere representing Earth.
///
/// This allows us to query using kilometres as distances in our kd-tree.
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
